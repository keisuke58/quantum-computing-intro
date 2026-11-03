"""
Bar chart comparing all methods on the phase-field problems (log scale, smaller is better).
Reads results/server/allen_cahn_N8/N10.json, cahn_hilliard_wz_N8.json, ac_vqpinn.json.

  python 58_fig_pf_compare.py   -> figs/pf_compare.png / .pdf
"""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = os.path.dirname(os.path.abspath(__file__))
R = lambda f: json.load(open(os.path.join(H, "results", "server", f)))
plt.rcParams.update({"font.family": ["IPAGothic", "DejaVu Sans"], "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})


def best_gm(rows, model, key="t1.0"):
    seeds = sorted({r["seed"] for r in rows if r["model"] == model})
    v = [min(r[key] for r in rows if r["model"] == model and r["seed"] == s and np.isfinite(r[key])) for s in seeds]
    return float(np.exp(np.mean(np.log(v))))


def gm(rows, model, key="t1.0"):
    return float(np.exp(np.mean(np.log([r[key] for r in rows if r["model"] == model]))))


METHODS = [  # label, color, hatch (trained methods hatched)
    ("量子リザバー", "#0072B2", ""),
    ("tanh 乱数特徴量", "#E69F00", ""),
    ("Fourier 乱数特徴量", "#009E73", ""),
    ("古典 PINN（学習）", "#CC79A7", "//"),
    ("変分 QPINN（学習）", "#56B4E9", "//"),
]


def main():
    ac8, ac10, wz, vq = R("allen_cahn_N8.json"), R("allen_cahn_N10.json"), R("cahn_hilliard_wz_N8.json"), R("ac_vqpinn.json")
    groups = {
        "Allen–Cahn\n256 個（N = 8）": [best_gm(ac8, "QERC-Haar"), best_gm(ac8, "tanh-RF"), best_gm(ac8, "Fourier"),
                                       gm(vq, "PINN"), gm(vq, "VQPINN")],
        "Allen–Cahn\n1024 個（N = 10）": [best_gm(ac10, "QERC-Haar"), best_gm(ac10, "tanh-RF"), best_gm(ac10, "Fourier"),
                                         np.nan, np.nan],
        "Cahn–Hilliard\n256 個（N = 8）": [best_gm(wz, "QERC-Haar"), best_gm(wz, "tanh-RF"), best_gm(wz, "Fourier"),
                                          np.nan, np.nan],
    }
    ref_pinn = {"Allen–Cahn\n256 個（N = 8）": 2.33e-2, "Cahn–Hilliard\n256 個（N = 8）": 9.51e-3}
    fig, ax = plt.subplots(figsize=(10, 4.6))
    w = 0.16; xs = np.arange(len(groups))
    nval = [sum(np.isfinite(v) for v in groups[g]) for g in groups]      # center the bars that exist
    for j, (lab, col, hat) in enumerate(METHODS):
        vals = [groups[g][j] for g in groups]
        pos = np.array([xs[i] + (j - (nval[i] - 1) / 2) * w for i in range(len(xs))])
        ax.bar(pos, vals, w * 0.92, color=col, hatch=hat, edgecolor="white", linewidth=0.8, label=lab)
        for p, v in zip(pos, vals):
            if np.isfinite(v):
                ax.text(p, v * 1.25, f"{v:.0e}".replace("e-0", "e-"), ha="center", va="bottom", fontsize=8, color="#333333")
    for i, g in enumerate(groups):
        if g in ref_pinn:
            ax.hlines(ref_pinn[g], xs[i] - 0.45, xs[i] + 0.45, colors="#444444", linestyles="--", lw=1.2)
            if i == 0:   # above the line, left (bars on the left are below it)
                ax.text(xs[i] - 0.45, ref_pinn[g] * 1.35, "公表 PINN", ha="left", fontsize=8, color="#444444")
            else:        # below the line, right
                ax.text(xs[i] + 0.45, ref_pinn[g] / 1.5, "公表 PINN", ha="right", va="top", fontsize=8,
                        color="#444444")
    ax.set_yscale("log"); ax.set_ylim(1e-6, 3)
    ax.set_xticks(xs); ax.set_xticklabels(list(groups))
    ax.set_ylabel("t = 1 の相対 L2 誤差（小さいほど良い）")
    ax.grid(True, axis="y", which="major", color="#e5e5e5", lw=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=5, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.savefig(os.path.join(H, "figs", "pf_compare.png"), dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(H, "figs", "pf_compare.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    main()
