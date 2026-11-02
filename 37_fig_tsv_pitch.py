# -*- coding: utf-8 -*-
"""37 - TSV 周期配列：ピッチ L に対する u 誤差（34 + 36 の結果）"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})
r34 = [dict(r, exp="base", N=8, n_si=6000) for r in json.load(open("results/tsv_periodic.json"))]
r36 = json.load(open("results/tsv_periodic_scaling.json"))
rows = r34 + r36
COL = {"QERC-Haar": "#1F3A68", "MLP2-RF": "#7A9A3A", "Fourier": "#8E3B7A"}


def best(sel):
    v = []
    for sd in sorted({r["seed"] for r in sel}):
        rr = [r["u"] for r in sel if r["seed"] == sd]
        v.append(min(rr))
    return np.exp(np.mean(np.log(v))), np.min(v), np.max(v)


fig, ax = plt.subplots(figsize=(4.6, 3.0))
for m, c in COL.items():
    Ls, g, lo, hi = [], [], [], []
    for L in [3.0, 4.0, 6.0, 8.0, 10.0]:
        sel = [r for r in rows if r["model"] == m and r["L"] == L and r["N"] == 8 and r["exp"] in ("base", "A")]
        if sel:
            b = best(sel); Ls.append(L); g.append(b[0]); lo.append(b[0] - b[1]); hi.append(b[2] - b[0])
    ax.errorbar(Ls, g, yerr=[lo, hi], color=c, marker="o", ms=4, lw=1.5, capsize=2, label=f"{m} (256 features)")
    sel = [r for r in rows if r["model"] == m and r["exp"] == "C"]
    if sel:
        ax.scatter([6.0 + {"QERC-Haar": -0.15, "MLP2-RF": 0.0, "Fourier": 0.15}[m]], [best(sel)[0]],
                   marker="*", s=70, color=c, zorder=4)
ax.scatter([], [], marker="*", s=70, color="#555555", label="1024 features (L = 6)")
ax.set_yscale("log"); ax.set_xlabel("Pitch L (via radius = 1)"); ax.set_ylabel("Relative $L_2$ error of u")
ax.set_title("Periodic TSV array, best scale per seed, 3 seeds", loc="left")
ax.legend(fontsize=7, frameon=False)
for ext in ("png", "pdf"):
    fig.savefig(f"figs/tsv_pitch.{ext}", bbox_inches="tight")
print("saved")
