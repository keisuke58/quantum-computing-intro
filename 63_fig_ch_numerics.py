"""
Cahn-Hilliard by a standard numerical method only (Fourier pseudo-spectral + ETDRK4), for a progress slide:
(a) phi(t, x), (b) snapshots, (c) free energy E(t), (d) mass drift over time; plus convergence checks.
Default case: Wight & Zhao 2021 (eps2 = 1e-6, g2 = 0.01, phi0 = -cos 2 pi x).

  python 63_fig_ch_numerics.py                 -> figs/ch_numerics.png / .pdf
  python 63_fig_ch_numerics.py --case gauss    -> figs/ch_numerics_gauss.png / .pdf  (eps2 = 0.02, g2 = 1)
"""
import os, importlib.util, json, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("ch", os.path.join(H, "54_cahn_hilliard_qerc.py"))
ch = importlib.util.module_from_spec(spec); spec.loader.exec_module(ch)
plt.rcParams.update({"font.family": ["IPAGothic", "DejaVu Sans"], "font.size": 11, "mathtext.fontset": "cm",
                     "axes.spines.top": False, "axes.spines.right": False})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--case", default="wz", choices=["wz", "gauss"]); a = ap.parse_args()
    if a.case == "wz":
        ch.G2, ch.IC, eps2, tag = 0.01, "cos2", 1e-6, ""
    else:
        ch.G2, ch.IC, eps2, tag = 1.0, "gauss", 0.02, "_gauss"
    M = 1024; x = -1 + 2 * np.arange(M) / M; h = 2 / M
    ts = [round(t, 3) for t in np.arange(0.01, 1.0001, 0.01)]
    u = ch.reference(eps2, x, ts, n_modes=1024, dt=1e-5)
    U = np.array([ch.phi0(x)] + [u[t] for t in ts]); T = np.array([0.0] + ts)
    E = np.array([ch.energy_ch(v, eps2) for v in U])
    mass = U.sum(1) * h; dm = np.abs(mass - mass[0])
    c_dt = ch.reference(eps2, x, [1.0], n_modes=1024, dt=2e-5)[1.0]
    c_n = ch.reference(eps2, x, [1.0], n_modes=512, dt=1e-5)[1.0]
    rel = lambda v: float(np.linalg.norm(v - u[1.0]) / np.linalg.norm(u[1.0]))
    checks = {"dt x2": rel(c_dt), "modes /2": rel(c_n)}

    fig = plt.figure(figsize=(16.5, 4.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.2, 1.1, 0.95, 0.95], wspace=0.36)
    ax = fig.add_subplot(gs[0])
    im = ax.imshow(U.T, origin="lower", aspect="auto", extent=[0, 1, -1, 1], cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xlabel("時間 $t$"); ax.set_ylabel("位置 $x$"); ax.set_title("(a) $\\phi(t,x)$", loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02); cb.set_ticks([-1, 0, 1]); cb.set_label("$\\phi$", labelpad=-2)
    ax = fig.add_subplot(gs[1])
    for t, c in zip([0.0, 0.25, 0.5, 1.0], ["#C9D1DE", "#9AA5B8", "#5A6B8A", "#1E2433"]):
        ax.plot(x, U[int(round(t * 100))], color=c, lw=2, label=f"$t={t:g}$")
    ax.set_xlabel("位置 $x$"); ax.set_ylabel("$\\phi$"); ax.set_title("(b) 解の時間変化", loc="left")
    ax.set_ylim(-1.15, 1.55); ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.legend(frameon=False, fontsize=9, loc="upper center", ncol=4, handlelength=1.2, columnspacing=0.8)
    ax = fig.add_subplot(gs[2])
    ax.plot(T, E, color="#1E2433", lw=2)
    ax.set_xlabel("時間 $t$"); ax.set_ylabel("自由エネルギー $E$"); ax.set_title("(c) エネルギーは単調に減少", loc="left")
    ax = fig.add_subplot(gs[3])
    ax.semilogy(T[1:], dm[1:] + 1e-18, color="#1E2433", lw=2)
    ax.set_ylim(1e-18, 1e-10)
    ax.axhline(2.2e-16, color="#5A6273", ls="--", lw=1)
    ax.text(0.03, 2.2e-16 * 2, "倍精度の丸め誤差", ha="left", va="bottom", fontsize=9, color="#5A6273",
            transform=ax.get_yaxis_transform())
    ax.set_xlabel("時間 $t$"); ax.set_ylabel("質量の変化（絶対値）")
    ax.set_title("(d) 質量は保存される", loc="left")
    fig.savefig(os.path.join(H, "figs", f"ch_numerics{tag}.png"), dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(H, "figs", f"ch_numerics{tag}.pdf"), bbox_inches="tight")
    dE = np.diff(E)
    info = {"case": a.case, "checks": checks, "E0": float(E[0]), "E1": float(E[-1]), "max_dE": float(dE.max()),
            "mass0": float(mass[0]), "max_dmass": float(dm.max()), "min_phi_t1": float(u[1.0].min()), "max_phi_t1": float(u[1.0].max())}
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
