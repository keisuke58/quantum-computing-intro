# -*- coding: utf-8 -*-
"""48 - 47 のパラメータスタディ（ピッチ，ビア径，ライナー）を図にする"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})
R = json.load(open("results/tsv_param.json"))
C = {"clamped": "#1F3A68", "free": "#B5651D"}
fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.0), gridspec_kw={"wspace": 0.35})
# (a) ピッチ（d = 5 µm）
ax = axs[0]
for bc in ("clamped", "free"):
    rr = sorted([r for r in R if r["kind"] == "pitch" and r["bc"] == bc], key=lambda r: r["p"])
    p = [r["p"] for r in rr]; k = [r["top_koz"] for r in rr]
    ax.plot(p, k, "o-", color=C[bc], lw=1.6, ms=4, label=bc)
    for pi, ki, r in zip(p, k, rr):
        if np.isnan(ki):
            ax.annotate("whole cell\n> 300 MPa", (pi, 2.6), ha="center", fontsize=7, color=C[bc])
ax.set_xlabel("pitch (µm), d = 5 µm"); ax.set_ylabel("KOZ width below surface (µm)")
ax.set_title("(a) Pitch", loc="left"); ax.set_ylim(0, 3.0); ax.legend(frameon=False, fontsize=7)
# (b) ビア径（p = 3d）
ax = axs[1]
for bc in ("clamped", "free"):
    rr = sorted([r for r in R if r["bc"] == bc and ((r["kind"] == "diam") or (r["kind"] == "pitch" and r["p"] == 15.0))],
                key=lambda r: r["d"])
    ax.plot([r["d"] for r in rr], [r["top_koz"] for r in rr], "o-", color=C[bc], lw=1.6, ms=4, label=bc)
ax.set_xlabel("via diameter d (µm), pitch = 3d"); ax.set_ylabel("KOZ width below surface (µm)")
ax.set_title("(b) Diameter", loc="left"); ax.set_ylim(0, 4.5)
# (c) ライナー（d = 5, p = 15）
ax = axs[2]
lab = ["no liner", "SiO2 liner 0.5 µm"]; x = np.arange(2); w = 0.36
for j, bc in enumerate(("clamped", "free")):
    base = [r for r in R if r["kind"] == "pitch" and r["p"] == 15.0 and r["bc"] == bc][0]
    lin = [r for r in R if r["kind"] == "liner" and r["bc"] == bc][0]
    ax.bar(x + (j - 0.5) * w, [base["top_vm_if"], lin["top_vm_if"]], w * 0.9, color=C[bc], label=bc)
ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_ylabel("von Mises in Si at interface (MPa)")
ax.set_title("(c) Liner, just below surface", loc="left")
fig.suptitle("Axisymmetric FE, height 50 µm, ΔT = −250 K. KOZ: distance from interface until von Mises < 300 MPa (0.25–0.5 µm below top)",
             fontsize=8, y=1.03)
for ext in ("png", "pdf"):
    fig.savefig(f"figs/tsv_param.{ext}", bbox_inches="tight")
print("saved")
