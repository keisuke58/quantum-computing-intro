"""
Allen-Cahn by a standard numerical method only (Fourier pseudo-spectral + ETDRK4), for a progress slide:
(a) u(t, x), (b) snapshots, (c) free energy E(t), (d) the interface width over time; plus convergence checks.

  python 62_fig_ac_numerics.py     -> figs/ac_numerics.png / .pdf
"""
import os, importlib.util, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("ac", os.path.join(H, "52_allen_cahn_qerc.py"))
ac = importlib.util.module_from_spec(spec); spec.loader.exec_module(ac)
plt.rcParams.update({"font.family": ["IPAGothic", "DejaVu Sans"], "font.size": 11, "mathtext.fontset": "cm",
                     "axes.spines.top": False, "axes.spines.right": False})
D = 1e-4


def width(u, x):
    """interface width 2 / max|u_x| (for u = tanh(x/d) this is 2d)"""
    return float(2 / np.max(np.abs(ac.grad_x(u))))


def main():
    M = 1024; x = -1 + 2 * np.arange(M) / M
    ts = [round(t, 3) for t in np.arange(0.01, 1.0001, 0.01)]
    u = ac.reference(D, x, ts, n_modes=1024, dt=1e-5)
    u0 = x**2 * np.cos(np.pi * x)
    U = np.array([u0] + [u[t] for t in ts]); T = np.array([0.0] + ts)
    E = np.array([ac.energy_ac(v, D) for v in U])
    # convergence checks at t = 1
    c_dt = ac.reference(D, x, [1.0], n_modes=1024, dt=2e-5)[1.0]
    c_n = ac.reference(D, x, [1.0], n_modes=512, dt=1e-5)[1.0]
    rel = lambda v: float(np.linalg.norm(v - u[1.0]) / np.linalg.norm(u[1.0]))
    checks = {"dt x2": rel(c_dt), "modes /2": rel(c_n)}

    fig = plt.figure(figsize=(16.5, 4.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.2, 1.1, 0.95, 0.95], wspace=0.36)
    ax = fig.add_subplot(gs[0])
    im = ax.imshow(U.T, origin="lower", aspect="auto", extent=[0, 1, -1, 1], cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xlabel("時間 $t$"); ax.set_ylabel("位置 $x$"); ax.set_title("(a) $u(t,x)$", loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02); cb.set_ticks([-1, 0, 1]); cb.set_label("$u$", labelpad=-2)
    ax = fig.add_subplot(gs[1])
    for t, c in zip([0.0, 0.25, 0.5, 1.0], ["#C9D1DE", "#9AA5B8", "#5A6B8A", "#1E2433"]):
        ax.plot(x, U[int(round(t * 100))], color=c, lw=2, label=f"$t={t:g}$")
    ax.set_xlabel("位置 $x$"); ax.set_ylabel("$u$"); ax.set_title("(b) 解の時間変化", loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, 0.06), ncol=1)
    ax = fig.add_subplot(gs[2])
    ax.plot(T, E, color="#1E2433", lw=2)
    ax.set_xlabel("時間 $t$"); ax.set_ylabel("自由エネルギー $E$"); ax.set_title("(c) エネルギーは単調に減少", loc="left")
    ax = fig.add_subplot(gs[3])
    weq = 2 * np.sqrt(2 * D / 5)
    ax.semilogy(T, [width(v, x) for v in U], color="#1E2433", lw=2)
    ax.axhline(weq, color="#5A6273", ls="--", lw=1)
    ax.text(0.03, weq * 1.15, "平衡の界面の幅", ha="left", va="bottom", fontsize=9, color="#5A6273",
            transform=ax.get_yaxis_transform())
    ax.set_xlabel("時間 $t$"); ax.set_ylabel("界面の幅 $2/\\max|u_x|$"); ax.set_title("(d) 界面が鋭くなる", loc="left")
    fig.savefig(os.path.join(H, "figs", "ac_numerics.png"), dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(H, "figs", "ac_numerics.pdf"), bbox_inches="tight")
    dE = np.diff(E)
    info = {"checks": checks, "E0": float(E[0]), "E1": float(E[-1]), "max_dE": float(dE.max()),
            "width_t1": width(u[1.0], x), "width_eq": float(2 * np.sqrt(2 * D / 5))}
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
