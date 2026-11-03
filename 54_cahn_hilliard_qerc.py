"""
Cahn-Hilliard in 1D with fixed features + least squares, mixed form, marched in time.

    phi_t = mu_xx,   mu = g2 (phi^3 - phi) - eps2 * phi_xx,   x in [-1, 1],  periodic,
    phi(0, x) = cos(pi x) - exp(-4 (pi x)^2)   (--ic gauss, default; g2 = 1, eps2 = 0.02)
             or -cos(2 pi x)                   (--ic cos2: Wight & Zhao 2021, g2 = 0.01, eps2 = 1e-6)

Both phi and mu are expanded in the same fixed features (only second derivatives are needed).
Time: BDF2 (first step backward Euler); phi^3 by Newton; every Newton iteration is one least-squares solve
for the coefficients of phi and mu together. Reference: Fourier pseudo-spectral + ETDRK4.

  python 54_cahn_hilliard_qerc.py --eps2 0.02 --N 8 --seeds 2
  python 54_cahn_hilliard_qerc.py --eps2 1e-6 --g2 0.01 --ic cos2 --ref_modes 1024 --N 8   # Wight & Zhao
  python 54_cahn_hilliard_qerc.py --check      # exact Fourier basis: time-step error and reference check
"""
import argparse, json, os, time
import numpy as np
import importlib.util

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("ac", os.path.join(_here, "52_allen_cahn_qerc.py"))
ac = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ac)


G2 = 1.0          # set from --g2 in main()
IC = "gauss"      # set from --ic in main()


def phi0(x):
    if IC == "cos2":
        return -np.cos(2 * np.pi * x)
    return np.cos(np.pi * x) - np.exp(-4 * (np.pi * x) ** 2)


def energy_ch(p, eps2):
    """free energy E = int (eps2/2 phi_x^2 + g2/4 (phi^2 - 1)^2) dx; Cahn-Hilliard is its H^-1 gradient flow"""
    h = 2 / len(p)
    return float(h * np.sum(0.5 * eps2 * ac.grad_x(p) ** 2 + 0.25 * G2 * (p**2 - 1) ** 2))


@ac.jax.jit
def _ch_newton_iter(Phi, Phixx, pk, c0, rhs0, dt, eps2, g2):
    jnp = ac.jnp
    A = jnp.block([[c0 * Phi, -dt * Phixx],
                   [-(g2 * (3 * pk**2 - 1))[:, None] * Phi + eps2 * Phixx, Phi]])
    b = jnp.concatenate([rhs0, -2 * g2 * pk**3])
    pn = Phi @ ac.lsq_j(A, b)[:Phi.shape[1]]
    return pn, jnp.max(jnp.abs(pn - pk))


def march(Phi, Phixx, x, eps2, dt, T, t_out, newton=8, tol=1e-10, efun=None):
    u0 = phi0(x)
    a = ac.lsq(Phi, u0); p = Phi @ a; p_prev = None
    err_ic = np.linalg.norm(p - u0) / np.linalg.norm(u0)
    Es = [efun(p)] if efun else None
    out = {}; want = {int(round(t / dt)): t for t in t_out}
    Pj, Pxxj = ac.jnp.asarray(Phi), ac.jnp.asarray(Phixx)      # the time loop runs on the JAX device
    p = ac.jnp.asarray(p)
    for n in range(1, int(round(T / dt)) + 1):
        if p_prev is None:
            c0, rhs0 = 1.0, p
        else:
            c0, rhs0 = 1.5, (4 * p - p_prev) / 2
        pk = p
        for _ in range(newton):
            # eq 1 (times dt): c0 phi - dt mu_xx = rhs0
            # eq 2: mu - g2 (3 pk^2 - 1) phi + eps2 phi_xx = -2 g2 pk^3
            pk, d = _ch_newton_iter(Pj, Pxxj, pk, c0, rhs0, dt, eps2, G2)
            if float(d) < tol:
                break
        p_prev, p = p, pk
        pn = np.asarray(p)
        if not np.all(np.isfinite(pn)) or np.max(np.abs(pn)) > 10:
            return err_ic, None
        if efun:
            Es.append(efun(pn))
        if n in want:
            out[want[n]] = pn.copy()
    if efun:
        out["E"] = Es
    return err_ic, out


def reference(eps2, x_eval, T_out, n_modes=256, dt=1e-5):
    xs = -1 + 2 * np.arange(n_modes) / n_modes
    k = np.fft.fftfreq(n_modes, d=2 / n_modes) * 2 * np.pi
    L = G2 * k**2 - eps2 * k**4                # -k^2 (-g2 phi) - eps2 k^4 phi
    E, E2 = np.exp(dt * L), np.exp(dt * L / 2)
    M = 32; r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = dt * L[:, None] + r[None, :]
    Q = dt * np.real(np.mean((np.exp(LR / 2) - 1) / LR, 1))
    f1 = dt * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3, 1))
    f2 = dt * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, 1))
    f3 = dt * np.real(np.mean((-4 - 3 * LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3, 1))
    Nf = lambda v: -G2 * k**2 * np.fft.fft(np.real(np.fft.ifft(v))**3)
    v = np.fft.fft(phi0(xs)); out = {}
    want = {int(round(t / dt)): t for t in T_out}
    for n in range(1, int(round(max(T_out) / dt)) + 1):
        Nv = Nf(v); a = E2 * v + Q * Nv; Na = Nf(a); b = E2 * v + Q * Na; Nb = Nf(b)
        c = E2 * a + Q * (2 * Nb - Nv); Nc = Nf(c)
        v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
        if n in want:
            out[want[n]] = np.real(np.exp(1j * np.outer(x_eval + 1, k)) @ (v / n_modes))
    return out


def fourier_basis(x, K):
    cols, cxx = [np.ones_like(x)], [np.zeros_like(x)]
    for k in range(1, K + 1):
        w = k * np.pi
        cols += [np.cos(w * x), np.sin(w * x)]; cxx += [-w * w * np.cos(w * x), -w * w * np.sin(w * x)]
    return np.array(cols).T, np.array(cxx).T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps2", type=float, default=0.02)
    ap.add_argument("--g2", type=float, default=1.0)
    ap.add_argument("--ic", default="gauss", choices=["gauss", "cos2"])
    ap.add_argument("--ref_modes", type=int, default=256)
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--dt", type=float, default=1e-3)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--models", default="QERC-Haar,tanh-RF,Fourier")
    ap.add_argument("--scales", default=None)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default="results/cahn_hilliard.json")
    a = ap.parse_args()
    global G2, IC
    G2, IC = a.g2, a.ic
    m = 2**a.N; M = 4 * m
    x = -1 + 2 * (np.arange(M) + 0.5) / M; h = 2 / M
    t_out = [round(a.T * f, 6) for f in (0.25, 0.5, 1.0)]
    t0 = time.time(); ref = reference(a.eps2, x, t_out, n_modes=a.ref_modes); print(f"reference {time.time()-t0:.0f}s", flush=True)
    rel = lambda u, t: float(np.linalg.norm(u - ref[t]) / np.linalg.norm(ref[t]))
    mass0 = np.sum(phi0(x)) * h
    E_ref = {t: energy_ch(ref[t], a.eps2) for t in t_out}
    print(f"reference energy: t=0:{energy_ch(phi0(x), a.eps2):.5f} " +
          " ".join(f"t={t}:{E_ref[t]:.5f}" for t in t_out), flush=True)
    print("reference: max|phi| " + " ".join(f"t={t}:{np.abs(ref[t]).max():.3f}" for t in t_out) +
          f"  change from t=0 at T: {np.linalg.norm(ref[t_out[-1]] - phi0(x)) / np.linalg.norm(phi0(x)):.2f}" +
          f"  mass drift {abs(np.sum(ref[t_out[-1]]) * h - mass0):.1e}")
    if a.check:
        ref2 = reference(a.eps2, x, t_out, n_modes=a.ref_modes, dt=2e-5)
        print("reference dt x2:", " ".join(f"{rel(ref2[t], t):.1e}" for t in t_out))
        for K in (64, 128):
            P, Pxx = fourier_basis(x, K)
            for dt in (1e-3, 5e-4):
                e, out = march(P, Pxx, x, a.eps2, dt, a.T, t_out)
                print(f"Fourier basis K={K} dt={dt}: ic {e:.1e} " + " ".join(f"t={t}:{rel(out[t], t):.1e}" for t in t_out))
        return
    grid = {
        "QERC-Haar": ([1, 2, 4, 8, 16], lambda sd, s: ac.make_qerc(a.N, sd, s)),
        "tanh-RF": ([2, 4, 8, 16, 32], lambda sd, s: ac.make_tanh(m, sd, s)),
        "Fourier": ([2, 4, 8, 16, 32], lambda sd, s: ac.make_fourier(m, sd, s)),
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
                Phi, Phixx = ac.feature_mats(ctor(sd, s), x)
                e_ic, out = march(Phi, Phixx, x, a.eps2, a.dt, a.T, t_out, efun=lambda p: energy_ch(p, a.eps2))
                errs = {t: (rel(out[t], t) if out else float("nan")) for t in t_out}
                drift = float(abs(np.sum(out[t_out[-1]]) * h - mass0)) if out else float("nan")
                em = (ac.energy_metrics(out["E"], E_ref[t_out[-1]]) if out
                      else {"E_rel_T": float("nan"), "E_up": float("nan")})
                rows.append({"eps2": a.eps2, "g2": a.g2, "init": a.ic, "N": a.N, "model": name, "seed": sd, "scale": s, "ic": float(e_ic),
                             "mass_drift": drift, **{f"t{t}": e for t, e in errs.items()}, **em})
                print(f"{name:10s} seed={sd} x{s:<5} ic={e_ic:.1e} " +
                      " ".join(f"t={t}:{e:.1e}" for t, e in errs.items()) + f" mass {drift:.1e} E_err={em['E_rel_T']:.1e} E_up={em['E_up']:.1e}  {time.time()-t0:.0f}s",
                      flush=True)
                json.dump(rows, open(a.out, "w"), indent=1)
    tl = f"t{t_out[-1]}"
    print("\n=== best scale per seed (by final-time error), geometric mean ===")
    for name in grid:
        v = []
        for sd in range(a.seeds):
            r = [r for r in rows if r["model"] == name and r["seed"] == sd and np.isfinite(r[tl])]
            if r:
                v.append(min(r, key=lambda r: r[tl]))
        if v:
            print(f"{name:10s} {tl}: {np.exp(np.mean([np.log(r[tl]) for r in v])):.2e}  scales {[r['scale'] for r in v]}")


if __name__ == "__main__":
    main()
