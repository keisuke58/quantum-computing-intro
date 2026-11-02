# -*- coding: utf-8 -*-
"""32 - 27 のショットノイズ結果を図にする（誤差・K_t vs 1 回路あたりのショット数）"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})
rows = json.load(open("results/qerc_shot_noise.json"))
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"wspace": 0.35})
for kind, col, lab in (("cliffordT", "#1F3A68", "Clifford+T"), ("haar", "#4A6FA5", "Haar")):
    for sd, mk in ((0, "o-"), (1, "^--")):
        rs = sorted([r for r in rows if r["kind"] == kind and r["seed"] == sd and np.isfinite(r["shots"])],
                    key=lambda r: r["shots"])
        inf = [r for r in rows if r["kind"] == kind and r["seed"] == sd and not np.isfinite(r["shots"])][0]
        S = [r["shots"] for r in rs]
        axs[0].loglog(S, [r["u"] for r in rs], mk, color=col, ms=4, label=f"{lab}, seed {sd}")
        axs[0].axhline(inf["u"], color=col, lw=0.6, ls=":")
        axs[1].semilogx(S, [r["kt"] for r in rs], mk, color=col, ms=4)
axs[0].axhspan(3e-5, 1e-4, color="#B5651D", alpha=0.2, lw=0, label="classical RF (tuned)")
axs[0].set_xlabel("Shots per circuit"); axs[0].set_ylabel("Relative $L_2$ error of u")
axs[0].set_title("(a) Error vs shots (dotted: exact)", loc="left"); axs[0].legend(frameon=False, fontsize=6.5)
axs[1].axhline(3, color="#888888", ls="--", lw=1); axs[1].set_ylim(0, 3.3)
axs[1].set_xlabel("Shots per circuit"); axs[1].set_ylabel("$K_t$ (exact 3)")
axs[1].set_title("(b) Stress concentration", loc="left")
for ext in ("png", "pdf"):
    fig.savefig(f"figs/qerc_shot_noise.{ext}", bbox_inches="tight")
