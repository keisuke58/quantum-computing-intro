"""
2D Allen-Cahn, shrinking circle, with fixed features + least squares (same solver as 52, in 2D).

    u_t = D Lap u + u - u^3,  (x, y) in [-1, 1]^2, periodic,  D = eps^2,
    u(0) = tanh((R0 - r) / (sqrt(2) eps)),  R0 = 0.5,  eps = 0.05.
Sharp-interface limit: the circle shrinks as R(t)^2 = R0^2 - 2 D t.

Features of z = (cos pi x, sin pi x, cos pi y, sin pi y) (periodic by construction); BDF2 + Newton, one least-squares
solve per Newton iteration, on the JAX device. Reference: 2D Fourier pseudo-spectral + ETDRK4.

  python 59_ac2d_qerc.py --check                  # exact Fourier basis vs reference (time-step error)
  python 59_ac2d_qerc.py --N 8 --seeds 2
"""
import argparse, json, time
import numpy as np
import jax, jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
EPS, R0 = 0.05, 0.5
D = EPS**2


def u_init(X, Y):
    return np.tanh((R0 - np.hypot(X, Y)) / (np.sqrt(2) * EPS))


def _z(p):
    return jnp.stack([jnp.cos(jnp.pi * p[0]), jnp.sin(jnp.pi * p[0]), jnp.cos(jnp.pi * p[1]), jnp.sin(jnp.pi * p[1])])


def make_qerc(N, seed, scale):
    g = np.random.default_rng(1000 + seed)
    W = jnp.asarray(g.normal(size=(4, 2 * N)) * scale); b = jnp.asarray(g.normal(size=2 * N))
    z = (g.normal(size=(2**N,) * 2) + 1j * g.normal(size=(2**N,) * 2)) / np.sqrt(2)
    q, r = np.linalg.qr(z); U = jnp.asarray(q * (np.diag(r) / np.abs(np.diag(r))))

    def feat(p):
        ang = jnp.pi * (jnp.tanh(_z(p) @ W + b) + 1) / 2
        psi = jnp.ones((1,), complex)
        for l in range(N):
            psi = jnp.kron(psi, jnp.stack([jnp.cos(ang[l] / 2), jnp.exp(1j * ang[N + l]) * jnp.sin(ang[l] / 2)]))
        return jnp.abs(U @ psi) ** 2
    return feat


def make_tanh(m, seed, scale):
    g = np.random.default_rng(seed)
    W = jnp.asarray(g.normal(size=(4, m)) * scale); b = jnp.asarray(g.normal(size=m))
    return lambda p: jnp.tanh(_z(p) @ W + b)


def make_fourier(m, seed, scale):
    g = np.random.default_rng(seed)
    W = jnp.asarray(g.normal(size=(4, m)) * scale); b = jnp.asarray(g.random(m) * 2 * np.pi)
    return lambda p: jnp.cos(_z(p) @ W + b)


def feature_mats(feat, P, chunk=512):
    f = jax.jit(jax.vmap(feat))
    lap = jax.jit(jax.vmap(lambda p: jnp.trace(jax.jacfwd(jax.jacfwd(feat))(p), axis1=1, axis2=2)))
    Ph, L = [], []
    for i in range(0, len(P), chunk):
        q = jnp.asarray(P[i:i + chunk]); Ph.append(np.asarray(f(q))); L.append(np.asarray(lap(q)))
    return np.concatenate(Ph), np.concatenate(L)


@jax.jit
def lsq_j(A, b, ridge=1e-12):
    sc = jnp.linalg.norm(A, axis=0) + 1e-30
    As = A / sc; G = As.T @ As
    G = G + ridge * jnp.trace(G) / G.shape[0] * jnp.eye(G.shape[0])
    return jnp.linalg.solve(G, As.T @ b) / sc


@jax.jit
def newton_iter(Phi, L, uk, a, rhs0):
    A = (a + 3 * uk**2 - 1)[:, None] * Phi - D * L
    un = Phi @ lsq_j(A, rhs0 + 2 * uk**3)
    return un, jnp.max(jnp.abs(un - uk))


def energy(u, n):
    """E = int (D/2 |grad u|^2 + 1/4 (u^2 - 1)^2), u on the n x n periodic grid"""
    U = u.reshape(n, n); h = 2 / n
    k = np.fft.fftfreq(n, d=h) * 2 * np.pi
    Uh = np.fft.fft2(U)
    ux = np.real(np.fft.ifft2(1j * k[None, :] * Uh)); uy = np.real(np.fft.ifft2(1j * k[:, None] * Uh))
    return float(h * h * np.sum(0.5 * D * (ux**2 + uy**2) + 0.25 * (U**2 - 1) ** 2))


def radius(u, n):
    return float(np.sqrt(np.sum(u > 0) * (2 / n) ** 2 / np.pi))


def march(Phi, L, u0, dt, T, t_out, n, newton=6, tol=1e-10):
    c = np.linalg.lstsq(Phi, u0, rcond=None)[0]; u = Phi @ c
    err_ic = float(np.linalg.norm(u - u0) / np.linalg.norm(u0))
    Es = [energy(u, n)]
    Pj, Lj = jnp.asarray(Phi), jnp.asarray(L)
    u = jnp.asarray(u); u_prev = None; out = {}
    want = {int(round(t / dt)): t for t in t_out}
    for k in range(1, int(round(T / dt)) + 1):
        if u_prev is None:
            a, rhs0 = 1.0 / dt, u / dt
        else:
            a, rhs0 = 1.5 / dt, (4 * u - u_prev) / (2 * dt)
        uk = u
        for _ in range(newton):
            uk, d = newton_iter(Pj, Lj, uk, a, rhs0)
            if float(d) < tol:
                break
        u_prev, u = u, uk
        un = np.asarray(u)
        if not np.all(np.isfinite(un)) or np.max(np.abs(un)) > 10:
            return err_ic, None
        Es.append(energy(un, n))
        if k in want:
            out[want[k]] = un.copy()
    out["E"] = Es
    return err_ic, out


def reference(n_eval, T_out, n_modes=128, dt=1e-3):
    xs = -1 + 2 * np.arange(n_modes) / n_modes
    X, Y = np.meshgrid(xs, xs)
    k = np.fft.fftfreq(n_modes, d=2 / n_modes) * 2 * np.pi
    K2 = k[None, :] ** 2 + k[:, None] ** 2
    Lop = -D * K2 + 1.0
    E, E2 = np.exp(dt * Lop), np.exp(dt * Lop / 2)
    M = 32; r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = dt * Lop[..., None] + r
    Q = dt * np.real(np.mean((np.exp(LR / 2) - 1) / LR, -1))
    f1 = dt * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3, -1))
    f2 = dt * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, -1))
    f3 = dt * np.real(np.mean((-4 - 3 * LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3, -1))
    Nf = lambda v: np.fft.fft2(-np.real(np.fft.ifft2(v)) ** 3)
    v = np.fft.fft2(u_init(X, Y)); out = {}
    want = {int(round(t / dt)): t for t in T_out}
    step = n_modes // n_eval
    for i in range(1, int(round(max(T_out) / dt)) + 1):
        Nv = Nf(v); a = E2 * v + Q * Nv; Na = Nf(a); b = E2 * v + Q * Na; Nb = Nf(b)
        c = E2 * a + Q * (2 * Nb - Nv); Nc = Nf(c)
        v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
        if i in want:
            out[want[i]] = np.real(np.fft.ifft2(v))[::step, ::step].reshape(-1)   # on the n_eval grid
    return out


def fourier_basis(P, K):
    cols, lap = [], []
    for kx in range(-K, K + 1):
        for ky in range(-K, K + 1):
            if kx * kx + ky * ky > K * K:
                continue
            ph = np.pi * (kx * P[:, 0] + ky * P[:, 1]); w2 = np.pi**2 * (kx * kx + ky * ky)
            for f in ((np.cos(ph), -w2 * np.cos(ph)), (np.sin(ph), -w2 * np.sin(ph))):
                if np.any(f[0] != 0):
                    cols.append(f[0]); lap.append(f[1])
    return np.array(cols).T, np.array(lap).T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--n", type=int, default=64, help="collocation grid n x n")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--T", type=float, default=20.0)
    ap.add_argument("--models", default="QERC-Haar,tanh-RF,Fourier")
    ap.add_argument("--scales", default=None)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default="results/ac2d.json")
    a = ap.parse_args()
    n = a.n; xs = -1 + 2 * np.arange(n) / n
    X, Y = np.meshgrid(xs, xs); P = np.c_[X.reshape(-1), Y.reshape(-1)]
    u0 = u_init(X, Y).reshape(-1)
    t_out = [round(a.T * f, 6) for f in (0.25, 0.5, 1.0)]
    t0 = time.time(); ref = reference(n, t_out); print(f"reference {time.time()-t0:.0f}s", flush=True)
    rel = lambda u, t: float(np.linalg.norm(u - ref[t]) / np.linalg.norm(ref[t]))
    E_ref = {t: energy(ref[t], n) for t in t_out}
    print("reference: R " + " ".join(f"t={t}:{radius(ref[t], n):.3f} (sharp-limit {np.sqrt(R0**2 - 2 * D * t):.3f})"
                                     for t in t_out) + " | E " + " ".join(f"{E_ref[t]:.4f}" for t in t_out), flush=True)
    if a.check:
        for K in (12, 20):
            Ph, L = fourier_basis(P, K)
            for dt in (a.dt, a.dt / 2):
                e, out = march(Ph, L, u0, dt, a.T, t_out, n)
                print(f"Fourier basis K={K} ({Ph.shape[1]} fns) dt={dt}: ic {e:.1e} " +
                      " ".join(f"t={t}:{rel(out[t], t):.1e}" for t in t_out), flush=True)
        return
    m = 2**a.N
    grid = {
        "QERC-Haar": ([1, 2, 4, 8], lambda sd, s: make_qerc(a.N, sd, s)),
        "tanh-RF": ([2, 4, 8, 16, 32], lambda sd, s: make_tanh(m, sd, s)),
        "Fourier": ([2, 4, 8, 16, 32], lambda sd, s: make_fourier(m, sd, s)),
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
                Ph, L = feature_mats(ctor(sd, s), P)
                e_ic, out = march(Ph, L, u0, a.dt, a.T, t_out, n)
                errs = {t: (rel(out[t], t) if out else float("nan")) for t in t_out}
                tl = t_out[-1]
                extra = ({"R_T": radius(out[tl], n), "E_rel_T": abs(out["E"][-1] - E_ref[tl]) / abs(E_ref[tl]),
                          "E_up": float(max(0.0, np.max(np.diff(out["E"]))) / abs(out["E"][0]))} if out
                         else {"R_T": float("nan"), "E_rel_T": float("nan"), "E_up": float("nan")})
                rows.append({"N": a.N, "model": name, "seed": sd, "scale": s, "ic": e_ic,
                             **{f"t{t}": e for t, e in errs.items()}, **extra})
                print(f"{name:10s} seed={sd} x{s:<4} ic={e_ic:.1e} " + " ".join(f"t={t}:{e:.1e}" for t, e in errs.items()) +
                      f" R={extra['R_T']:.3f} E_up={extra['E_up']:.0e}  {time.time()-t0:.0f}s", flush=True)
                json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
