# -*- coding: utf-8 -*-
"""40 - CFRP 繊維単位セル（38）の結果を図にする"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})
D = json.load(open("results/cfrp_unitcell.json")); rows = D["rows"]
MODELS = ["QERC-Haar", "MLP2-RF", "tanh-RF", "Fourier"]
COL = {"QERC-Haar": "#1F3A68", "MLP2-RF": "#7A9A3A", "tanh-RF": "#B5651D", "Fourier": "#8E3B7A"}
VF = sorted({r["vf"] for r in rows})


def best(vf, kind, m):
    out = []
    for sd in sorted({r["seed"] for r in rows}):
        rr = [r for r in rows if r["vf"] == vf and r["kind"] == kind and r["model"] == m and r["seed"] == sd]
        out.append(min(rr, key=lambda r: r["field"]))
    return out


fig, axs = plt.subplots(1, 3, figsize=(10.5, 2.9), gridspec_kw={"wspace": 0.35})
w = 0.8 / len(MODELS)
for ax, kind, key, title in ((axs[0], "antiplane", "eff_err", "(a) Longitudinal shear $G_{12}$"),
                             (axs[1], "inplane", "eff_err", "(b) Transverse $C_{11}$")):
    for j, m in enumerate(MODELS):
        for i, vf in enumerate(VF):
            v = np.array([b[key] for b in best(vf, kind, m)])
            x = i + (j - 1.5) * w
            ax.bar(x, np.exp(np.log(v).mean()), w * 0.9, color=COL[m], label=m if i == 0 else None)
            ax.scatter(x + np.linspace(-w / 4, w / 4, len(v)), v, s=6, color="#222222", zorder=3)
    de = [np.mean([r["fe_de"] for r in rows if r["vf"] == vf and r["kind"] == kind]) for vf in VF]
    ax.plot(range(len(VF)), de, "k--", lw=1, label="FE: h vs 2h")
    ax.set_xticks(range(len(VF))); ax.set_xticklabels([f"$V_f$ = {v:g}" for v in VF])
    ax.set_yscale("log"); ax.set_ylabel("Rel. error of effective modulus"); ax.set_title(title, loc="left")
h_, l_ = axs[0].get_legend_handles_labels()
fig.legend(h_, l_, fontsize=7, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.36, 1.06))
# (c) 等価 G12: FE と Halpin-Tsai
fe = [f for f in D["fe"] if f["kind"] == "antiplane"]
ax = axs[2]
vv = np.linspace(0.05, 0.75, 100)
GF, GM = 15.0, 1.3; eta = (GF / GM - 1) / (GF / GM + 1)
ax.plot(vv, GM * (1 + eta * vv) / (1 - eta * vv), color="#888888", lw=1.5, label="Halpin-Tsai ($\\xi$ = 1)")
ax.plot([f["vf"] for f in fe], [f["fe_eff"] for f in fe], "o", color="#222222", ms=5, label="FE (square array)")
ax.set_xlabel("$V_f$"); ax.set_ylabel("$G_{12}$ (GPa)"); ax.set_title("(c) Reference values", loc="left")
ax.legend(fontsize=7, frameon=False)
for ext in ("png", "pdf"):
    fig.savefig(f"figs/cfrp_unitcell.{ext}", bbox_inches="tight")
print("saved")
