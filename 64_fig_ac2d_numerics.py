"""
2D Allen-Cahn (shrinking circle) by a standard numerical method only (2D Fourier pseudo-spectral + ETDRK4),
for a progress slide: (a) u at t = 0, (b) u at t = T with the u = 0 contours over time,
(c) area / pi = R^2 against the sharp-interface law R0^2 - 2 D t, (d) free energy E(t); plus convergence checks.

  python 64_fig_ac2d_numerics.py     -> figs/ac2d_numerics.png / .pdf
"""
import os, importlib.util, json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("ac2", os.path.join(H, "59_ac2d_qerc.py"))
ac2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(ac2)
plt.rcParams.update({"font.family": ["IPAGothic", "DejaVu Sans"], "font.size": 11, "mathtext.fontset": "cm",
                     "axes.spines.top": False, "axes.spines.right": False})
T = 20.0
N_GRID = 128
TS = [round(0.5 * k, 6) for k in range(1, 41)]


def _run(job):
    modes, dt, ts = job
    return ac2.reference(N_GRID, ts, n_modes=modes, dt=dt)


def main():
    n = 128; xs = -1 + 2 * np.arange(n) / n; X, Y = np.meshgrid(xs, xs); h = 2 / n
    ts = TS
    cache = os.path.join(H, "results", "ac2d_numerics_fields.npz")
    if os.path.exists(cache):
        z = np.load(cache); u = {t: z["u"][i] for i, t in enumerate(ts)}; c_dt, c_n = z["c_dt"], z["c_n"]
    else:    # three reference runs in parallel: the field, dt x2, modes x2
        with ProcessPoolExecutor(3) as ex:
            u, r_dt, r_n = ex.map(_run, [(n, 1e-3, ts), (n, 2e-3, [T]), (2 * n, 1e-3, [T])])
        c_dt, c_n = r_dt[T], r_n[T]
        np.savez_compressed(cache, u=np.array([u[t] for t in ts]), c_dt=c_dt, c_n=c_n)
    U = [ac2.u_init(X, Y).reshape(-1)] + [u[t] for t in ts]; Tt = np.array([0.0] + ts)
    E = np.array([ac2.energy(v, n) for v in U])
    R2 = np.array([np.sum((v + 1) / 2) * h * h / np.pi for v in U])          # area of {u > 0} / pi, smooth version
    rel = lambda v: float(np.linalg.norm(v - u[T]) / np.linalg.norm(u[T]))
    checks = {"dt x2": rel(c_dt), "modes x2": rel(c_n)}

    fig = plt.figure(figsize=(16.5, 4.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1.12, 1.0, 1.0], wspace=0.34)
    for i, (k, ttl) in enumerate([(0, "(a) $u$（$t=0$）"), (len(Tt) - 1, f"(b) $u$（$t={T:g}$）と界面の動き")]):
        ax = fig.add_subplot(gs[i])
        im = ax.imshow(U[k].reshape(n, n), origin="lower", extent=[-1, 1, -1, 1], cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xlabel("$x$"); ax.set_ylabel("$y$", labelpad=-2); ax.set_title(ttl, loc="left"); ax.set_aspect("equal")
        ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
        if i == 1:
            for j, tt in enumerate([0, 5, 10, 15]):
                ax.contour(xs, xs, U[int(tt * 2)].reshape(n, n), [0], colors="white", linewidths=1.0,
                           linestyles="dashed")
            ax.text(0.0, -0.75, "白破線：界面 $u=0$（$t=0,5,10,15$）", ha="center", fontsize=9, color="white")
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_ticks([-1, 0, 1]); cb.set_label("$u$", labelpad=-2)
    ax = fig.add_subplot(gs[2])
    ax.plot(Tt, R2, color="#1E2433", lw=2, label="数値解")
    ax.plot(Tt, ac2.R0**2 - 2 * ac2.D * Tt, color="#E69F00", lw=1.4, ls="--", label="鋭い界面の理論 $R_0^2-2Dt$")
    ax.set_xlabel("時間 $t$"); ax.set_ylabel("$R^2$ = 面積 / $\\pi$"); ax.set_title("(c) 円の面積は直線的に減る", loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax = fig.add_subplot(gs[3])
    ax.plot(Tt, E, color="#1E2433", lw=2)
    ax.set_xlabel("時間 $t$"); ax.set_ylabel("自由エネルギー $E$"); ax.set_title("(d) エネルギーは単調に減少", loc="left")
    fig.savefig(os.path.join(H, "figs", "ac2d_numerics.png"), dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(H, "figs", "ac2d_numerics.pdf"), bbox_inches="tight")
    slope = np.polyfit(Tt[4:], R2[4:], 1)[0]
    info = {"checks": checks, "E0": float(E[0]), "E1": float(E[-1]), "max_dE": float(np.diff(E).max()),
            "R_T": float(np.sqrt(R2[-1])), "R_sharp_T": float(np.sqrt(ac2.R0**2 - 2 * ac2.D * T)),
            "slope_R2": float(slope), "slope_theory": -2 * ac2.D}
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
