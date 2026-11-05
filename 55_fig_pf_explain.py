"""
Explanatory figure for the phase-field slides, drawn from the computed reference solutions:
(a) double-well energy (two stable phases), (b) Allen-Cahn u(t, x), (c) the interface profile at t = 1,
(d) Cahn-Hilliard phi(t, x) snapshots with the conserved mean.

  python 55_fig_pf_explain.py        -> figs/pf_explain.png / .pdf
"""
import os, importlib.util
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

_here = os.path.dirname(os.path.abspath(__file__))


def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_here, fn))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


ac = _load("ac", "52_allen_cahn_qerc.py")
ch = _load("ch", "54_cahn_hilliard_qerc.py")

plt.rcParams.update({"font.family": ["IPAGothic", "DejaVu Sans"], "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})
INK, MUTED, A, B = "#1E2433", "#5A6273", "#C0392B", "#1F5FAD"


def main():
    x = -1 + 2 * (np.arange(1024) + 0.5) / 1024
    ts = [round(t, 3) for t in np.arange(0.0, 1.0001, 0.02)][1:]
    u = ac.reference(1e-4, x, ts, n_modes=512, dt=2e-5)
    U = np.array([x**2 * np.cos(np.pi * x)] + [u[t] for t in ts])
    tch = [0.002, 0.02, 0.1, 1.0]
    p = ch.reference(0.02, x, tch, n_modes=256, dt=1e-5)

    fig, axs = plt.subplots(2, 2, figsize=(11, 7.4), gridspec_kw={"hspace": 0.55, "wspace": 0.28})
    # (a) double well
    ax = axs[0, 0]; s = np.linspace(-1.5, 1.5, 300)
    ax.plot(s, 1.25 * (s**2 - 1) ** 2, color=INK, lw=2)
    ax.scatter([-1, 1], [0, 0], s=60, color=[B, A], zorder=3)
    ax.annotate("相 B（u = −1）", (-1, 0), (-1.45, 0.9), color=B, arrowprops=dict(arrowstyle="-", color=B))
    ax.annotate("相 A（u = +1）", (1, 0), (0.35, 0.9), color=A, arrowprops=dict(arrowstyle="-", color=A))
    ax.annotate("混ざった状態（u = 0）は不安定", (0, 1.25), (-0.95, 2.0), color=MUTED,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.set_ylim(-0.2, 2.6); ax.set_xlabel("秩序変数 u"); ax.set_ylabel("自由エネルギー f(u)")
    ax.set_title("(a) 2 つの安定な相（二重井戸）", loc="left")
    # (b) Allen-Cahn u(t, x)
    ax = axs[0, 1]
    im = ax.imshow(U.T, origin="lower", aspect="auto", extent=[0, 1, -1, 1], cmap="RdBu_r", vmin=-1, vmax=1)
    ax.text(0.80, 0.27, "相 A", color="white", ha="center", va="center", fontsize=12)
    ax.text(0.80, -0.27, "相 A", color="white", ha="center", va="center", fontsize=12)
    ax.text(0.80, 0.82, "相 B", color="white", ha="center", va="center", fontsize=12)
    ax.text(0.80, -0.82, "相 B", color="white", ha="center", va="center", fontsize=12)
    ax.annotate("界面", (0.97, 0.505), (0.45, 0.62), color=INK, fontsize=11,
                arrowprops=dict(arrowstyle="->", color=INK))
    ax.set_xlabel("時間 t"); ax.set_ylabel("位置 x"); ax.set_title("(b) Allen–Cahn：u(t, x)", loc="left")
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02, label="u")
    # (c) interface profile at t = 1
    ax = axs[1, 0]; u1 = u[1.0]
    i0 = np.where((np.sign(u1[:-1]) != np.sign(u1[1:])) & (x[:-1] > 0))[0][0]; x0 = x[i0]
    w = np.abs(x - x0) < 0.06
    ax.plot(x[w], u1[w], color=INK, lw=2)
    lo, hi = x[w][np.argmin(np.abs(u1[w] + 0.9))], x[w][np.argmin(np.abs(u1[w] - 0.9))]
    ax.annotate("", (lo, 0), (hi, 0), arrowprops=dict(arrowstyle="<->", color=MUTED))
    ax.text(0.442, -0.45, f"界面の幅 ≈ {abs(hi - lo):.2f}", ha="left", color=MUTED)
    ax.axhline(1, color=A, lw=0.8, ls=":"); ax.axhline(-1, color=B, lw=0.8, ls=":")
    ax.set_xlabel("位置 x"); ax.set_ylabel("u(1, x)")
    ax.set_title("(c) t = 1 の界面（拡大）", loc="left")
    # (d) Cahn-Hilliard snapshots
    ax = axs[1, 1]
    cols = ["#9AA5B8", "#6B7A93", "#3D4E6E", INK]
    h = x[1] - x[0]
    for t, c in zip(tch, cols):
        ax.plot(x, p[t], color=c, lw=1.8, label=f"t = {t:g}")
    ax.set_xlabel("位置 x"); ax.set_ylabel("φ(t, x)")
    ax.set_title("(d) Cahn–Hilliard：相の量を保ったまま分離", loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.savefig(os.path.join(_here, "figs", "pf_explain.png"), dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(_here, "figs", "pf_explain.pdf"), bbox_inches="tight")
    m0 = np.sum(x**2 * np.cos(np.pi * x)) * h / 2
    print(f"AC mean: t=0 {m0:+.4f}, t=1 {np.sum(u1) * h / 2:+.4f};  CH mean: " +
          " ".join(f"{np.sum(p[t]) * h / 2:+.4f}" for t in tch), f"interface at x={x0:.3f}")


if __name__ == "__main__":
    main()
