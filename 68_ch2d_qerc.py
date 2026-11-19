"""
2D Cahn-Hilliard with fixed features + least squares (mixed form), same solver family as 54 (1D CH) and 59 (2D AC).

    phi_t = Lap mu,  mu = phi^3 - phi - eps^2 Lap phi,  (x, y) in [-1, 1]^2, periodic,  eps = 0.05,
    phi(0) = tanh((1 - sqrt((x/a)^2 + (y/b)^2)) * sqrt(a b) / (sqrt(2) eps)),  a = 0.7, b = 0.35:
an elliptic droplet that relaxes to a circle of the same area (mass is conserved). A deterministic, smooth problem:
spinodal decomposition from noise amplifies any error exponentially, so it cannot rank the methods.

phi and mu are expanded in the same features of z = (cos pi x, sin pi x, cos pi y, sin pi y). BDF2 + Newton; each
Newton iteration is one least-squares solve for both coefficient vectors, on the JAX device.
Reference: 2D Fourier pseudo-spectral + ETDRK4 with linear stabilisation (as in 65_ch2d_numerics.py).

Checked on CPU (2026-11-17) with the truncated Fourier basis K = 12 (881 functions) on the 64 x 64 grid:
  eps = 0.1 (resolved: fit of phi(0) 1.5e-3):  error at t = 0.2 is 4e-4 / 6e-4 / 1e-3 for dt = 8e-3 / 4e-3 / 2e-3
  eps = 0.05 (not resolved: fit of phi(0) 1.6e-2): error at t = 0.4 is 5.9e-2 / 8.7e-2 for dt = 8e-3 / 4e-3
The error stays at the level of what the basis can represent and grows slowly with the number of steps (each step
is a least-squares projection), so it is not reduced by a smaller dt. All models use the same dt = 4e-3.

  python 68_ch2d_qerc.py --check            # exact Fourier basis vs reference (time-step error), reference checks
  python 68_ch2d_qerc.py --N 8 --seeds 3
"""
import argparse, json, os, time, importlib.util
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("ac2", os.path.join(H, "59_ac2d_qerc.py"))
ac2 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ac2)
jax, jnp = ac2.jax, ac2.jnp
EPS, A_EL, B_EL, S_STAB = 0.05, 0.7, 0.35, 2.0
EPS2 = EPS**2


def phi_init(X, Y):
    r = np.sqrt((X / A_EL) ** 2 + (Y / B_EL) ** 2)
    return np.tanh((1 - r) * np.sqrt(A_EL * B_EL) / (np.sqrt(2) * EPS))


def energy(p, n):
    U = p.reshape(n, n); h = 2 / n
    k = np.fft.fftfreq(n, d=h) * 2 * np.pi
    Uh = np.fft.fft2(U)
    ux = np.real(np.fft.ifft2(1j * k[None, :] * Uh)); uy = np.real(np.fft.ifft2(1j * k[:, None] * Uh))
    return float(h * h * np.sum(0.5 * EPS2 * (ux**2 + uy**2) + 0.25 * (U**2 - 1) ** 2))


def aspect(p, n):
    """ratio of the second moments of (phi + 1) / 2 along x and y: (a / b)^2 for an ellipse, 1 for a circle"""
    xs = -1 + 2 * np.arange(n) / n; w = (p.reshape(n, n) + 1) / 2
    X, Y = np.meshgrid(xs, xs)
    return float(np.sum(w * X**2) / np.sum(w * Y**2))


@jax.jit
def newton_iter(Phi, L, pk, c0, rhs0, dt):
    # eq 1 (times dt): c0 phi - dt Lap mu = rhs0;   eq 2: mu - (3 pk^2 - 1) phi + eps2 Lap phi = -2 pk^3
    A = jnp.block([[c0 * Phi, -dt * L],
                   [-(3 * pk**2 - 1)[:, None] * Phi + EPS2 * L, Phi]])
    b = jnp.concatenate([rhs0, -2 * pk**3])
    pn = Phi @ ac2.lsq_j(A, b)[:Phi.shape[1]]
    return pn, jnp.max(jnp.abs(pn - pk))


def march(Phi, L, u0, dt, T, t_out, n, newton=8, tol=1e-10):
    c = np.linalg.lstsq(Phi, u0, rcond=None)[0]; p = Phi @ c
    err_ic = float(np.linalg.norm(p - u0) / np.linalg.norm(u0))
    Es = [energy(p, n)]
    Pj, Lj = jnp.asarray(Phi), jnp.asarray(L)
    p = jnp.asarray(p); p_prev = None; out = {}
    want = {int(round(t / dt)): t for t in t_out}
    for k in range(1, int(round(T / dt)) + 1):
        if p_prev is None:
            c0, rhs0 = 1.0, p
        else:
            c0, rhs0 = 1.5, (4 * p - p_prev) / 2
        pk = p
        for _ in range(newton):
            pk, d = newton_iter(Pj, Lj, pk, c0, rhs0, dt)
            if float(d) < tol:
                break
        p_prev, p = p, pk
        pn = np.asarray(p)
        if not np.all(np.isfinite(pn)) or np.max(np.abs(pn)) > 10:
            return err_ic, None
        Es.append(energy(pn, n))
        if k in want:
            out[want[k]] = pn.copy()
    out["E"] = Es
    return err_ic, out


def reference(n_eval, T_out, n_modes=128, dt=1e-4):
    xs = -1 + 2 * np.arange(n_modes) / n_modes
    X, Y = np.meshgrid(xs, xs)
    k = np.fft.fftfreq(n_modes, d=2 / n_modes) * 2 * np.pi
    k2 = k[None, :] ** 2 + k[:, None] ** 2
    Lop = -S_STAB * k2 - EPS2 * k2**2
    E, E2 = np.exp(dt * Lop), np.exp(dt * Lop / 2)
    M = 32; r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = dt * Lop[..., None] + r
    Q = dt * np.real(np.mean((np.exp(LR / 2) - 1) / LR, -1))
    f1 = dt * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3, -1))
    f2 = dt * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, -1))
    f3 = dt * np.real(np.mean((-4 - 3 * LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3, -1)); del LR

    def Nf(v):
        p = np.real(np.fft.ifft2(v))
        return -k2 * np.fft.fft2(p**3 - (1 + S_STAB) * p)
    v = np.fft.fft2(phi_init(X, Y)); out = {}
    want = {int(round(t / dt)): t for t in T_out}
    step = n_modes // n_eval
    for i in range(1, int(round(max(T_out) / dt)) + 1):
        Nv = Nf(v); a = E2 * v + Q * Nv; Na = Nf(a); b = E2 * v + Q * Na; Nb = Nf(b)
        c = E2 * a + Q * (2 * Nb - Nv); Nc = Nf(c)
        v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
        if i in want:
            out[want[i]] = np.real(np.fft.ifft2(v))[::step, ::step].reshape(-1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--n", type=int, default=64, help="collocation grid n x n")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--dt", type=float, default=4e-3)
    ap.add_argument("--T", type=float, default=0.4)
    ap.add_argument("--models", default="QERC-Haar,tanh-RF,Fourier")
    ap.add_argument("--scales", default=None)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default="results/ch2d.json")
    a = ap.parse_args()
    n = a.n; xs = -1 + 2 * np.arange(n) / n
    X, Y = np.meshgrid(xs, xs); P = np.c_[X.reshape(-1), Y.reshape(-1)]
    u0 = phi_init(X, Y).reshape(-1); h = 2 / n
    t_out = [round(a.T * f, 6) for f in (0.25, 0.5, 1.0)]
    cache = os.path.join(H, "results", f"ch2d_ref_n{n}_T{a.T:g}.npz")
    if os.path.exists(cache):
        z = np.load(cache); ref = {t: z[f"t{t}"] for t in t_out}
    else:
        t0 = time.time(); ref = reference(n, t_out); print(f"reference {time.time()-t0:.0f}s", flush=True)
        np.savez_compressed(cache, **{f"t{t}": ref[t] for t in t_out})
    rel = lambda u, t: float(np.linalg.norm(u - ref[t]) / np.linalg.norm(ref[t]))
    mass0 = float(np.sum(u0) * h * h)
    E_ref = {t: energy(ref[t], n) for t in t_out}
    print(f"reference: aspect t=0:{aspect(u0, n):.3f} " + " ".join(f"t={t}:{aspect(ref[t], n):.3f}" for t in t_out) +
          f" | E t=0:{energy(u0, n):.4f} " + " ".join(f"{E_ref[t]:.4f}" for t in t_out) +
          " | mass drift " + " ".join(f"{abs(np.sum(ref[t]) * h * h - mass0):.1e}" for t in t_out), flush=True)
    if a.check:
        r2 = reference(n, t_out, dt=2e-4); print("reference dt x2: " + " ".join(f"t={t}:{rel(r2[t], t):.1e}" for t in t_out), flush=True)
        r3 = reference(n, t_out, n_modes=256); print("reference modes x2: " + " ".join(f"t={t}:{rel(r3[t], t):.1e}" for t in t_out), flush=True)
        for K in (12, 20):
            Ph, L = ac2.fourier_basis(P, K)
            for dt in (a.dt, a.dt / 2):
                e, out = march(Ph, L, u0, dt, a.T, t_out, n)
                print(f"Fourier basis K={K} ({Ph.shape[1]} fns) dt={dt}: ic {e:.1e} " +
                      (" ".join(f"t={t}:{rel(out[t], t):.1e}" for t in t_out) if out else "diverged"), flush=True)
        return
    m = 2**a.N
    grid = {
        "QERC-Haar": ([0.25, 0.5, 1, 2, 4], lambda sd, s: ac2.make_qerc(a.N, sd, s)),
        "tanh-RF": ([0.5, 1, 2, 4, 8], lambda sd, s: ac2.make_tanh(m, sd, s)),
        "Fourier": ([0.5, 1, 2, 4, 8], lambda sd, s: ac2.make_fourier(m, sd, s)),
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
                Ph, L = ac2.feature_mats(ctor(sd, s), P)
                e_ic, out = march(Ph, L, u0, a.dt, a.T, t_out, n)
                errs = {t: (rel(out[t], t) if out else float("nan")) for t in t_out}
                tl = t_out[-1]
                extra = ({"aspect_T": aspect(out[tl], n), "E_rel_T": abs(out["E"][-1] - E_ref[tl]) / abs(E_ref[tl]),
                          "E_up": float(max(0.0, np.max(np.diff(out["E"]))) / abs(out["E"][0])),
                          "mass_drift": float(abs(np.sum(out[tl]) * h * h - mass0))} if out
                         else {"aspect_T": float("nan"), "E_rel_T": float("nan"), "E_up": float("nan"), "mass_drift": float("nan")})
                rows.append({"N": a.N, "model": name, "seed": sd, "scale": s, "ic": e_ic,
                             **{f"t{t}": e for t, e in errs.items()}, **extra})
                print(f"{name:10s} seed={sd} x{s:<4} ic={e_ic:.1e} " + " ".join(f"t={t}:{e:.1e}" for t, e in errs.items()) +
                      f" aspect={extra['aspect_T']:.3f} mass={extra['mass_drift']:.1e} E_up={extra['E_up']:.1e} {time.time()-t0:.0f}s", flush=True)
                os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
                json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
