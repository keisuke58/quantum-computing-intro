# -*- coding: utf-8 -*-
"""35 - 34（TSV 周期配列の単位セル）の結果を図にする．(a) 配列ピッチ L ごとの u 誤差，(b) 最良モデルの σ 誤差"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})
src = sys.argv[1] if len(sys.argv) > 1 else "results/tsv_periodic.json"
rows = json.load(open(src))
MODELS = [m for m in ["QERC-Haar", "tensor-RP", "MLP2-RF", "tanh-RF", "Fourier"] if any(r["model"] == m for r in rows)]
COL = {"QERC-Haar": "#1F3A68", "tensor-RP": "#5B8DB8", "MLP2-RF": "#7A9A3A", "tanh-RF": "#B5651D", "Fourier": "#8E3B7A"}
Ls = sorted({r["L"] for r in rows})


def best(L, m, key):
    """シードごとに u が最小の倍率を選び，その倍率での key を返す"""
    out = []
    for sd in sorted({r["seed"] for r in rows}):
        rr = [r for r in rows if r["L"] == L and r["model"] == m and r["seed"] == sd]
        if rr:
            out.append(min(rr, key=lambda r: r["u"])[key])
    return np.array(out)


fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.8), gridspec_kw={"wspace": 0.3})
w = 0.8 / len(MODELS)
for ax, key, lab in ((axs[0], "u", "Relative $L_2$ error of u"), (axs[1], "s", r"Relative $L_2$ error of $\sigma$")):
    for j, m in enumerate(MODELS):
        for i, L in enumerate(Ls):
            v = best(L, m, key)
            if len(v) == 0:
                continue
            x = i + (j - (len(MODELS) - 1) / 2) * w
            ax.bar(x, np.exp(np.log(v).mean()), w * 0.9, color=COL[m], label=m if i == 0 else None)
            ax.scatter(x + np.linspace(-w / 4, w / 4, len(v)), v, s=6, color="#222222", zorder=3)
    if key == "u":
        fe = [np.mean([r["fe_du"] for r in rows if r["L"] == L]) for L in Ls]
        ax.plot(range(len(Ls)), fe, "k--", lw=1, marker="_", ms=14, label="FE: h vs 2h")
    ax.set_xticks(range(len(Ls))); ax.set_xticklabels([f"L = {L:g}" for L in Ls])
    ax.set_yscale("log"); ax.set_ylabel(lab)
axs[0].set_title("(a) Displacement, best scale per seed", loc="left")
axs[1].set_title("(b) Stress (same scale)", loc="left")
axs[0].legend(fontsize=7, frameon=False, ncol=2)
for ext in ("png", "pdf"):
    fig.savefig(f"figs/tsv_periodic.{ext}", bbox_inches="tight")
print("saved figs/tsv_periodic")
