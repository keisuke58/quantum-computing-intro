# -*- coding: utf-8 -*-
"""41 - ショットノイズ対策（39）の結果を図にする"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})
rows = json.load(open("results/shot_mitigation.json"))
SH = [1e3, 1e4, 1e5, 1e6, 1e7]
METH = {"strong": ("Strong form (27)", "#B5651D", "-"), "strong-split": ("Strong form, split shots", "#B5651D", ":"),
        "ritz": ("Energy (Ritz), 1st deriv. only", "#1F3A68", "-"), "ritz-split": ("Energy, split shots", "#1F3A68", ":")}


def gm(N, S, m, k="u"):
    v = [r[k] for r in rows if r["N"] == N and r["shots"] == S and r["method"] == m]
    return np.exp(np.mean(np.log(v))) if k == "u" else np.mean(v)


fig, axs = plt.subplots(1, 3, figsize=(10.5, 2.9), gridspec_kw={"wspace": 0.35})
ax = axs[0]
for m, (lab, c, ls) in METH.items():
    ax.plot(SH, [gm(10, S, m) for S in SH], color=c, ls=ls, marker="o", ms=3, lw=1.5, label=lab)
for m in ("strong", "ritz"):
    ax.axhline(gm(10, np.inf, m), color=METH[m][1], lw=0.8, ls="--")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Shots per circuit"); ax.set_ylabel("Relative $L_2$ error of u")
ax.set_title("(a) Kirsch, N = 10", loc="left")
h_, l_ = ax.get_legend_handles_labels()
fig.legend(h_ + [plt.Line2D([], [], color="#555555", ls="--", lw=0.8)], l_ + ["No noise"], fontsize=7, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.07))
ax = axs[1]
for m in ("strong", "ritz"):
    ax.plot(SH, [gm(10, S, m, "kt") for S in SH], color=METH[m][1], marker="o", ms=3, lw=1.5, label=METH[m][0])
ax.axhline(3, color="#888888", lw=1, ls="--"); ax.set_xscale("log"); ax.set_ylim(0, 4)
ax.set_xlabel("Shots per circuit"); ax.set_ylabel("$K_t$ (exact 3)"); ax.set_title("(b) Stress concentration, N = 10", loc="left")
ax = axs[2]
for N, c in ((6, "#9DB4D3"), (8, "#5B8DB8"), (10, "#1F3A68")):
    ax.plot(SH, [gm(N, S, "ritz") for S in SH], color=c, marker="o", ms=3, lw=1.5, label=f"N = {N}")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Shots per circuit"); ax.set_ylabel("Relative $L_2$ error of u")
ax.set_title("(c) Energy form, qubit count", loc="left"); ax.legend(fontsize=7, frameon=False)
for ext in ("png", "pdf"):
    fig.savefig(f"figs/shot_mitigation.{ext}", bbox_inches="tight")
print("saved")
