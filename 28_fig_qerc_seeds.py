# -*- coding: utf-8 -*-
"""28 - 26 の 5 シード比較（量子リザバー vs スケールを振った古典ランダム特徴量）を図にする"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})
rows = json.load(open("results/qerc_ikeda_seeds.json"))
ORDER = [("cliffordT", "QERC\nClifford+T"), ("haar", "QERC\nHaar"), ("ising", "QERC\nIsing"),
         ("crf0.5", "RF ×0.5"), ("crf1.0", "RF ×1"), ("crf1.5", "RF ×1.5"), ("crf3.0", "RF ×3"), ("crf5.0", "RF ×5")]
COL = {"c": "#1F3A68", "r": "#B5651D"}

fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.8), gridspec_kw={"wspace": 0.3})
for ax, key, lab in ((axs[0], "u", "Relative $L_2$ error of u"), (axs[1], "kt", "$K_t$ (exact 3)")):
    for i, (m, _) in enumerate(ORDER):
        v = np.array([r[key] for r in rows if r["model"] == m])
        c = COL["r"] if m.startswith("crf") else COL["c"]
        center = np.exp(np.log(v).mean()) if key == "u" else v.mean()
        ax.bar(i, center, 0.7, color=c, alpha=0.9 if key == "u" else 0.0, edgecolor=c)
        ax.scatter(i + np.linspace(-0.15, 0.15, len(v)), v, s=10, color="#222222", zorder=3)
    ax.set_xticks(range(len(ORDER))); ax.set_xticklabels([l for _, l in ORDER], fontsize=7, rotation=40, ha="right")
    ax.set_ylabel(lab)
    if key == "u":
        ax.set_yscale("log"); ax.set_title("(a) Kirsch, PDE only, N=10, 5 seeds", loc="left")
    else:
        ax.axhline(3, color="#888888", lw=1, ls="--"); ax.set_ylim(2, 3.5)
        ax.set_title("(b) Stress concentration factor", loc="left")
for ext in ("png", "pdf"):
    fig.savefig(f"figs/qerc_seeds.{ext}", bbox_inches="tight")
