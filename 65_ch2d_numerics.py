"""
2D Cahn-Hilliard, spinodal decomposition, by a standard numerical method only (2D Fourier pseudo-spectral +
ETDRK4 with a linear stabilisation term), for a progress slide.

    phi_t = Lap mu,  mu = phi^3 - phi - eps^2 Lap phi,  (x, y) in [-1, 1]^2, periodic,  eps = 0.05,
    phi(0) = small random noise with zero mean (fixed seed, defined by its Fourier modes).

Split: phi_t = L phi + N(phi) with L = -S k^2 - eps^2 k^4 and N = -k^2 FFT(phi^3 - (1 + S) phi), S = 2.
Panels: phi at three times, free energy E(t). Checks: dt x2, modes x2, mass drift.

  python 65_ch2d_numerics.py     -> figs/ch2d_numerics.png / .pdf  (fields cached in results/ch2d_numerics_fields.npz)
"""
import os, json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.family": ["IPAGothic", "DejaVu Sans"], "font.size": 11, "mathtext.fontset": "cm",
                     "axes.spines.top": False, "axes.spines.right": False})
EPS, S, T, N0 = 0.05, 2.0, 1.0, 128
T_SNAP = [0.05, 0.2, 1.0]


def init_hat(n):
    """Fourier coefficients of phi(0) on an n x n grid: N0 x N0 grid noise, zero-padded when n > N0"""
    g = np.random.default_rng(0)
    p = g.uniform(-0.05, 0.05, (N0, N0)); p -= p.mean()
    v = np.fft.fft2(p)
    if n == N0:
        return v
    w = np.zeros((n, n), complex); h = N0 // 2
    for a in (slice(0, h), slice(-h, None)):
        for b in (slice(0, h), slice(-h, None)):
            w[a, b] = v[a, b]
    return w * (n / N0) ** 2


def energy_hat(v, k2, n):
    p = np.real(np.fft.ifft2(v)); h = 2 / n
    grad2 = np.sum(k2 * np.abs(v) ** 2) / n**2 * h * h          # Parseval: int |grad phi|^2
    return float(0.5 * EPS**2 * grad2 + h * h * np.sum(0.25 * (p**2 - 1) ** 2))


def run(job):
    n, dt = job
    k = np.fft.fftfreq(n, d=2 / n) * 2 * np.pi
    k2 = k[None, :] ** 2 + k[:, None] ** 2
    L = -S * k2 - EPS**2 * k2**2
    E, E2 = np.exp(dt * L), np.exp(dt * L / 2)
    M = 32; r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = dt * L[..., None] + r
    Q = dt * np.real(np.mean((np.exp(LR / 2) - 1) / LR, -1))
    f1 = dt * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3, -1))
    f2 = dt * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, -1))
    f3 = dt * np.real(np.mean((-4 - 3 * LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3, -1)); del LR
    def Nf(v):
        p = np.real(np.fft.ifft2(v))
        return -k2 * np.fft.fft2(p**3 - (1 + S) * p)
    v = init_hat(n); steps = int(round(T / dt))
    rec = sorted(set(np.unique(np.round(np.logspace(0, np.log10(steps), 60)).astype(int))))
    snap = {int(round(t / dt)): t for t in T_SNAP}
    Es, ts, mass, out = [energy_hat(v, k2, n)], [0.0], [np.real(v[0, 0]) / n**2 * 4], {}
    sub = n // N0
    for i in range(1, steps + 1):
        Nv = Nf(v); a = E2 * v + Q * Nv; Na = Nf(a); b = E2 * v + Q * Na; Nb = Nf(b)
        c = E2 * a + Q * (2 * Nb - Nv); Nc = Nf(c)
        v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
        if i in rec:
            Es.append(energy_hat(v, k2, n)); ts.append(i * dt); mass.append(np.real(v[0, 0]) / n**2 * 4)
        if i in snap:
            out[snap[i]] = np.real(np.fft.ifft2(v))[::sub, ::sub]
    return {"snap": out, "t": np.array(ts), "E": np.array(Es), "mass": np.array(mass), "phi0": np.real(np.fft.ifft2(init_hat(N0)))}


def main():
    cache = os.path.join(H, "results", "ch2d_numerics_fields.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        R = {"snap": {t: z[f"s{t}"] for t in T_SNAP}, "t": z["t"], "E": z["E"], "mass": z["mass"]}
        c_dt, c_n = z["c_dt"], z["c_n"]
    else:
        with ProcessPoolExecutor(3) as ex:
            R, Rdt, Rn = ex.map(run, [(N0, 1e-4), (N0, 2e-4), (2 * N0, 1e-4)])
        c_dt, c_n = Rdt["snap"][T], Rn["snap"][T]
        np.savez_compressed(cache, t=R["t"], E=R["E"], mass=R["mass"], c_dt=c_dt, c_n=c_n,
                            **{f"s{t}": R["snap"][t] for t in T_SNAP})
    uT = R["snap"][T]
    rel = lambda w: float(np.linalg.norm(w - uT) / np.linalg.norm(uT))
    checks = {"dt x2": rel(c_dt), "modes x2": rel(c_n)}

    fig = plt.figure(figsize=(16.5, 4.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1.12, 1.1], wspace=0.48)
    for i, t in enumerate(T_SNAP):
        ax = fig.add_subplot(gs[i])
        im = ax.imshow(R["snap"][t], origin="lower", extent=[-1, 1, -1, 1], cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(f"({'abc'[i]}) $\\phi$（$t={t:g}$）", loc="left"); ax.set_aspect("equal")
        ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1]); ax.set_xlabel("$x$"); ax.set_ylabel("$y$", labelpad=-2)
        if i == 2:
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_ticks([-1, 0, 1]); cb.set_label("$\\phi$", labelpad=-2)
    ax = fig.add_subplot(gs[3])
    ax.semilogx(R["t"][1:], R["E"][1:], color="#1E2433", lw=2); ax.set_xlim(1e-3, 1.1)
    for t in T_SNAP:
        ax.axvline(t, color="#9AA5B8", lw=0.8, ls=":")
    ax.set_xlabel("時間 $t$（対数）"); ax.set_ylabel("自由エネルギー $E$"); ax.set_title("(d) エネルギーは単調に減少", loc="left")
    fig.savefig(os.path.join(H, "figs", "ch2d_numerics.png"), dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(H, "figs", "ch2d_numerics.pdf"), bbox_inches="tight")
    info = {"checks": checks, "E0": float(R["E"][0]), "E1": float(R["E"][-1]), "max_dE": float(np.diff(R["E"]).max()),
            "max_dmass": float(np.max(np.abs(R["mass"] - R["mass"][0]))), "minmax_T": [float(uT.min()), float(uT.max())]}
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
