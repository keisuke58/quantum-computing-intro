"""
Allen-Cahn (Raissi et al. 2019, Eq. 12) with fixed features + least squares, marched in time.

    u_t - D u_xx + 5 u^3 - 5 u = 0,  x in [-1, 1],  t in [0, 1],  u(0, x) = x^2 cos(pi x),  periodic.

Space: u(t, x) = sum_k c_k(t) phi_k(x). Periodicity is built in by feeding (cos pi x, sin pi x) to the features.
Time: BDF2 (first step backward Euler); each step is solved by Newton, every Newton iteration is one
least-squares solve at fixed collocation points. The feature matrices are computed once.
Reference: Fourier pseudo-spectral (ETDRK4, Kassam-Trefethen contour integral).

  python 52_allen_cahn_qerc.py --D 1e-4 --N 8 --seeds 2
"""
import argparse, json, time
import numpy as np
import jax, jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# ---------------- features of x through z = (cos pi x, sin pi x) ----------------
def _z(x):
    return jnp.stack([jnp.cos(jnp.pi * x), jnp.sin(jnp.pi * x)])


def make_qerc(N, seed, scale):
    g = np.random.default_rng(1000 + seed)
    W = jnp.asarray(g.normal(size=(2, 2 * N)) * scale); b = jnp.asarray(g.normal(size=2 * N))
    z = (g.normal(size=(2**N,) * 2) + 1j * g.normal(size=(2**N,) * 2)) / np.sqrt(2)
    q, r = np.linalg.qr(z); U = jnp.asarray(q * (np.diag(r) / np.abs(np.diag(r))))   # Haar

    def feat(x):
        ang = jnp.pi * (jnp.tanh(_z(x) @ W + b) + 1) / 2
        psi = jnp.ones((1,), complex)
        for l in range(N):
            psi = jnp.kron(psi, jnp.stack([jnp.cos(ang[l] / 2), jnp.exp(1j * ang[N + l]) * jnp.sin(ang[l] / 2)]))
        return jnp.abs(U @ psi) ** 2
    return feat


def make_tanh(m, seed, scale):
    g = np.random.default_rng(seed)
    W = jnp.asarray(g.normal(size=(2, m)) * scale); b = jnp.asarray(g.normal(size=m))
    return lambda x: jnp.tanh(_z(x) @ W + b)


def make_fourier(m, seed, scale):
    g = np.random.default_rng(seed)
    W = jnp.asarray(g.normal(size=(2, m)) * scale); b = jnp.asarray(g.random(m) * 2 * np.pi)
    return lambda x: jnp.cos(_z(x) @ W + b)


def feature_mats(feat, x):
    f = jax.vmap(feat)
    fxx = jax.vmap(jax.jacfwd(jax.jacfwd(feat)))
    return np.asarray(f(jnp.asarray(x))), np.asarray(fxx(jnp.asarray(x)))


def grad_x(u):
    """spectral x-derivative of samples on the uniform periodic grid over [-1, 1]"""
    M = len(u); k = np.fft.fftfreq(M, d=2 / M) * 2 * np.pi
    return np.real(np.fft.ifft(1j * k * np.fft.fft(u)))


def energy_ac(u, D):
    """free energy E = int (D/2 u_x^2 + 5/4 (u^2 - 1)^2) dx; Allen-Cahn is its L2 gradient flow"""
    h = 2 / len(u)
    return float(h * np.sum(0.5 * D * grad_x(u) ** 2 + 1.25 * (u**2 - 1) ** 2))


def energy_metrics(Es, E_ref_T):
    """relative error of E at the final time, and the largest increase of E between steps (relative to E(0));
    a gradient flow must have E non-increasing, so E_up > 0 means the scheme broke dissipation"""
    Es = np.asarray(Es)
    return {"E_rel_T": float(abs(Es[-1] - E_ref_T) / abs(E_ref_T)),
            "E_up": float(max(0.0, np.max(np.diff(Es))) / abs(Es[0]))}


def lsq(A, b, ridge=1e-12):
    sc = np.linalg.norm(A, axis=0) + 1e-30
    As = A / sc; G = As.T @ As; G[np.diag_indices_from(G)] += ridge * np.trace(G) / len(G)
    return np.linalg.solve(G, As.T @ b) / sc


@jax.jit
def lsq_j(A, b, ridge=1e-12):
    """same as lsq, on the JAX device (GPU when available); float64"""
    sc = jnp.linalg.norm(A, axis=0) + 1e-30
    As = A / sc; G = As.T @ As
    G = G + ridge * jnp.trace(G) / G.shape[0] * jnp.eye(G.shape[0])
    return jnp.linalg.solve(G, As.T @ b) / sc


@jax.jit
def _ac_newton_iter(Phi, Phixx, uk, a, rhs0, D):
    A = (a + 15 * uk**2 - 5)[:, None] * Phi - D * Phixx
    un = Phi @ lsq_j(A, rhs0 + 10 * uk**3)
    return un, jnp.max(jnp.abs(un - uk))


# ---------------- time marching with fixed features ----------------
def march(Phi, Phixx, x, D, dt, T, t_out, newton=6, tol=1e-10, efun=None, u0=None):
    u0 = x**2 * np.cos(np.pi * x) if u0 is None else u0(x)
    c = lsq(Phi, u0); u_prev = None; u = Phi @ c
    err_ic = np.linalg.norm(u - u0) / np.linalg.norm(u0)
    Es = [efun(u)] if efun else None
    out = {}; n_steps = int(round(T / dt)); want = {int(round(t / dt)): t for t in t_out}
    Pj, Pxxj = jnp.asarray(Phi), jnp.asarray(Phixx)          # the time loop runs on the JAX device
    u = jnp.asarray(u)
    for n in range(1, n_steps + 1):
        if u_prev is None:                     # backward Euler for the first step
            a, rhs0 = 1.0 / dt, u / dt
        else:                                  # BDF2
            a, rhs0 = 1.5 / dt, (4 * u - u_prev) / (2 * dt)
        uk = u
        for _ in range(newton):
            uk, d = _ac_newton_iter(Pj, Pxxj, uk, a, rhs0, D)
            if float(d) < tol:
                break
        u_prev, u = u, uk
        un = np.asarray(u)
        if not np.all(np.isfinite(un)) or np.max(np.abs(un)) > 10:
            return err_ic, None
        if efun:
            Es.append(efun(un))
        if n in want:
            out[want[n]] = un.copy()
    if efun:
        out["E"] = Es
    return err_ic, out


# ---------------- reference: Fourier spectral + ETDRK4 ----------------
def reference(D, x_eval, T_out, n_modes=1024, dt=1e-5, u0=None):
    xs = -1 + 2 * np.arange(n_modes) / n_modes
    k = np.fft.fftfreq(n_modes, d=2 / n_modes) * 2 * np.pi
    L = -D * k**2 + 5.0
    E, E2 = np.exp(dt * L), np.exp(dt * L / 2)
    M = 32; r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = dt * L[:, None] + r[None, :]
    Q = dt * np.real(np.mean((np.exp(LR / 2) - 1) / LR, 1))
    f1 = dt * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3, 1))
    f2 = dt * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, 1))
    f3 = dt * np.real(np.mean((-4 - 3 * LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3, 1))
    Nf = lambda v: np.fft.fft(-5 * np.real(np.fft.ifft(v))**3)
    v = np.fft.fft(xs**2 * np.cos(np.pi * xs) if u0 is None else u0(xs)); out = {}
    want = {int(round(t / dt)): t for t in T_out}
    for n in range(1, int(round(max(T_out) / dt)) + 1):
        Nv = Nf(v); a = E2 * v + Q * Nv; Na = Nf(a); b = E2 * v + Q * Na; Nb = Nf(b)
        c = E2 * a + Q * (2 * Nb - Nv); Nc = Nf(c)
        v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
        if n in want:
            # evaluate the Fourier series at x_eval
            coef = v / n_modes
            out[want[n]] = np.real(np.exp(1j * np.outer(x_eval + 1, k)) @ coef)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=float, default=1e-4)
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--M", type=int, default=0, help="collocation points (default 4 * 2^N)")
    ap.add_argument("--dt", type=float, default=1e-3)
    ap.add_argument("--models", default="QERC-Haar,tanh-RF,Fourier")
    ap.add_argument("--scales", default=None)
    ap.add_argument("--out", default="results/allen_cahn.json")
    a = ap.parse_args()
    m = 2**a.N; M = a.M or 4 * m
    x = -1 + 2 * (np.arange(M) + 0.5) / M
    t_out = [0.25, 0.5, 1.0]
    t0 = time.time(); ref = reference(a.D, x, t_out); print(f"reference {time.time()-t0:.0f}s", flush=True)
    E_ref = {t: energy_ac(ref[t], a.D) for t in t_out}
    print("reference energy: " + " ".join(f"t={t}:{E_ref[t]:.5f}" for t in t_out), flush=True)
    grid = {
        "QERC-Haar": ([2, 4, 8, 16, 32], lambda sd, s: make_qerc(a.N, sd, s)),
        "tanh-RF": ([8, 16, 32, 64], lambda sd, s: make_tanh(m, sd, s)),
        "Fourier": ([8, 16, 32, 64], lambda sd, s: make_fourier(m, sd, s)),
    }
    grid = {k: v for k, v in grid.items() if k in a.models.split(",")}
    if a.scales:
        sc = [float(s) for s in a.scales.split(",")]
        grid = {k: (sc, v[1]) for k, v in grid.items()}
    rows = []
    for name, (scales, ctor) in grid.items():
        for sd in range(a.seeds):
            for s in scales:
                t0 = time.time()
                Phi, Phixx = feature_mats(ctor(sd, s), x)
                e_ic, out = march(Phi, Phixx, x, a.D, a.dt, 1.0, t_out, efun=lambda u: energy_ac(u, a.D))
                errs = {t: (float(np.linalg.norm(out[t] - ref[t]) / np.linalg.norm(ref[t])) if out else float("nan"))
                        for t in t_out}
                em = energy_metrics(out["E"], E_ref[1.0]) if out else {"E_rel_T": float("nan"), "E_up": float("nan")}
                rows.append({"D": a.D, "N": a.N, "model": name, "seed": sd, "scale": s, "ic": float(e_ic),
                             **{f"t{t}": e for t, e in errs.items()}, **em})
                print(f"{name:10s} seed={sd} x{s:<5} ic={e_ic:.1e} " +
                      " ".join(f"t={t}:{e:.1e}" for t, e in errs.items()) +
                      f" E_err={em['E_rel_T']:.1e} E_up={em['E_up']:.1e}  {time.time()-t0:.0f}s", flush=True)
                json.dump(rows, open(a.out, "w"), indent=1)
    print("\n=== best scale per seed (by t=1 error), geometric mean ===")
    for name in grid:
        v = []
        for sd in range(a.seeds):
            r = [r for r in rows if r["model"] == name and r["seed"] == sd and np.isfinite(r["t1.0"])]
            if r:
                b = min(r, key=lambda r: r["t1.0"]); v.append(b)
        if v:
            print(f"{name:10s} t=1: {np.exp(np.mean([np.log(r['t1.0']) for r in v])):.2e}  "
                  f"t=0.5: {np.exp(np.mean([np.log(r['t0.5']) for r in v])):.2e}  scales {[r['scale'] for r in v]}")


if __name__ == "__main__":
    main()
