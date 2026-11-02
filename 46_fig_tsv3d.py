# -*- coding: utf-8 -*-
"""46 - Abaqus（fifa）の 3D 単位セルの結果を可視化する

左: 1/4 セルの切り出し図（面 x = 0, y = 0 はビアを縦に半分に切った断面，上は z = 25 µm の上面）に von Mises を塗る
右: ビアの中心から外向き（x = 0 の断面上，y 方向）の von Mises．中央面と表面のすぐ下で比べる
上段: 上下固定（MORE-Stress のシナリオ 1），下段: 上面自由（ウェハ表面）
"""
import gzip, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from scipy.interpolate import griddata

plt.rcParams.update({"font.size": 9, "savefig.dpi": 300})
_h = os.path.dirname(os.path.abspath(__file__))
P2, HH, A = 7.5, 25.0, 2.5


def load(bc):
    with gzip.open(os.path.join(_h, "results", "abaqus", f"tsv_3d_{bc}_stress.csv.gz"), "rt") as f:
        head = f.readline().strip().split(",")
        d = np.loadtxt(f, delimiter=",")
    return {k: d[:, i] for i, k in enumerate(head)}


def face(s, axis, val, tol=0.35, n=120):
    """面の近くの要素重心から，その面の 2 次元の格子に von Mises を補間"""
    c = np.stack([s["x"], s["y"], s["z"]], 1)
    near = np.abs(c[:, axis] - val) < tol
    other = [k for k in range(3) if k != axis]
    lim = [P2, P2, HH]
    g0 = np.linspace(0, lim[other[0]], n); g1 = np.linspace(0, lim[other[1]], n * (3 if lim[other[1]] == HH else 1))
    G0, G1 = np.meshgrid(g0, g1, indexing="ij")
    V = griddata(c[near][:, other], s["mises"][near], (G0, G1), method="linear")
    V2 = griddata(c[near][:, other], s["mises"][near], (G0, G1), method="nearest")
    V = np.where(np.isnan(V), V2, V)
    XYZ = [None, None, None]
    XYZ[axis] = np.full_like(G0, val); XYZ[other[0]] = G0; XYZ[other[1]] = G1
    return XYZ, V


def profile(s, z0, dz=0.6, nb=40):
    """x = 0 の断面（x < 0.35 µm），高さ z0 付近で y 方向に並べた von Mises（区間平均）"""
    m = (s["x"] < 0.35) & (np.abs(s["z"] - z0) < dz)
    edges = np.linspace(0, P2, nb + 1); yc = 0.5 * (edges[1:] + edges[:-1])
    v = [s["mises"][m & (s["y"] >= a) & (s["y"] < b)].mean() if (m & (s["y"] >= a) & (s["y"] < b)).any() else np.nan
         for a, b in zip(edges[:-1], edges[1:])]
    return yc, np.array(v)


fig = plt.figure(figsize=(9.5, 8.4))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.25], hspace=0.35, wspace=0.32, top=0.9, bottom=0.07, left=0.02, right=0.98)
for row, (bc, title) in enumerate((("clamped", "Top and bottom clamped (MORE-Stress scenario 1)"),
                                   ("free", "Top surface free (wafer surface)"))):
    s = load(bc)
    vmax = np.percentile(s["mises"], 99.5)        # 角の特異点で色が潰れないよう上位 0.5% で頭打ち
    norm = colors.Normalize(0, vmax); cmap = cm.viridis
    ax = fig.add_subplot(gs[row, 0], projection="3d")
    for axis, val in ((0, 0.0), (1, 0.0), (2, HH)):
        (X, Y, Z), V = face(s, axis, val)
        ax.plot_surface(X, Y, Z, facecolors=cmap(norm(V)), rstride=1, cstride=1, linewidth=0, antialiased=False,
                        shade=False)
    t = np.linspace(0, np.pi / 2, 50)
    ax.plot(A * np.cos(t), A * np.sin(t), HH, color="w", lw=1)       # 上面でのビアの縁
    ax.plot([0, 0], [A, A], [0, HH], color="w", lw=0.8); ax.plot([A, A], [0, 0], [0, HH], color="w", lw=0.8)
    ax.set_box_aspect((P2, P2, HH)); ax.view_init(elev=22, azim=-135)
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)"); ax.set_zlabel("z (µm)")
    fig.text(0.02, 0.93 - 0.47 * row, f"({'ab'[row]}) {title}", fontsize=10, weight="bold")
    mpl = cm.ScalarMappable(norm=norm, cmap=cmap); mpl.set_array([])
    fig.colorbar(mpl, ax=ax, shrink=0.55, pad=0.02, label="von Mises (MPa)")
    ax2 = fig.add_subplot(gs[row, 1])
    for z0, lab, c in ((0.3, "mid-plane (z ≈ 0)", "#1F3A68"), (HH - 0.5, "just below the top (z ≈ 24.5 µm)", "#B5651D")):
        yc, v = profile(s, z0)
        ax2.plot(yc, v, color=c, lw=1.8, label=lab)
    ax2.axvline(A, color="#888888", lw=1, ls="--")
    ax2.text(A - 0.1, ax2.get_ylim()[0] + 0.05 * np.ptp(ax2.get_ylim()), "Cu | Si", fontsize=8, color="#555555", ha="center")
    ax2.set_xlabel("distance from via centre (µm)"); ax2.set_ylabel("von Mises (MPa)")
    ax2.set_xlim(0, P2); ax2.legend(fontsize=7, frameon=False)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)
fig.suptitle("TSV unit cell (pitch 15 µm, Ø 5 µm, h 50 µm, ΔT = −250 K), Abaqus C3D10 — quarter cell, upper half", fontsize=10, y=0.985)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(_h, "figs", f"tsv3d_abaqus.{ext}"), bbox_inches="tight")
print("saved figs/tsv3d_abaqus")
