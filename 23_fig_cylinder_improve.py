# -*- coding: utf-8 -*-
"""
23 - 18_qpinn_cylinder_improve.py の結果を図にする

  (a) 変種ごとの相対 L2 誤差（u, v, p）：棒 = シード平均，点 = 各シード
  (b) 学習曲線（L-BFGS-B の損失）
  (c) 渦度 ω = ∂v/∂x − ∂u/∂y の比較（参照データ，量子 improved，回路なし，同規模 MLP）

使い方: python 23_fig_cylinder_improve.py --res a.json b.json c.json --params DIR --out figs/
"""
import argparse, importlib.util, json, os
import numpy as np
import jax, jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_s = importlib.util.spec_from_file_location(
    "c", os.path.join(os.path.dirname(os.path.abspath(__file__)), "18_qpinn_cylinder_improve.py"))
C = importlib.util.module_from_spec(_s); _s.loader.exec_module(C)

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans", "savefig.dpi": 300})

ORDER = ["V0-baseline", "V1-angle", "V3-readout", "C-identity", "C-mlp311"]
LABEL = {"V0-baseline": "Quantum improved\n(baseline)", "V1-angle": "V1: angle ×π",
         "V2-reupload": "V2: re-upload", "V3-readout": "V3: 12 observables",
         "C-identity": "Circuit removed\n(identity)", "C-mlp311": "Classical MLP\n(311 params)"}
SHORT = {"V0-baseline": "Q improved", "V1-angle": "V1 angle×π", "V2-reupload": "V2 re-upload",
         "V3-readout": "V3 12 obs.", "C-identity": "No circuit", "C-mlp311": "MLP (311)"}
# 量子系は青系，対照は灰・橙（CVD でも明度差で区別できる組み合わせ）
COLOR = {"V0-baseline": "#1F3A68", "V1-angle": "#4A6FA5", "V2-reupload": "#7A9CC6",
         "V3-readout": "#9DB6D8", "C-identity": "#8C8C8C", "C-mlp311": "#B5651D"}


def load_rows(paths):
    rows = []
    for p in paths:
        if os.path.exists(p):
            rows += json.load(open(p))
    return rows


def fig_errors(rows, ax_row):
    names = [n for n in ORDER if any(r["variant"] == n for r in rows)]
    for ax, key, title in zip(ax_row, ("u", "v", "p"), ("u", "v", "p")):
        for i, n in enumerate(names):
            vals = [r[key] for r in rows if r["variant"] == n]
            ax.bar(i, np.mean(vals), width=0.7, color=COLOR[n], edgecolor="white", linewidth=1)
            ax.scatter(np.full(len(vals), i) + np.linspace(-0.12, 0.12, len(vals)), vals,
                       s=12, color="#222222", zorder=3)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([SHORT[n] for n in names], rotation=40, ha="right", fontsize=7)
        ax.set_title(f"Relative $L_2$ error of {title}")
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.6); ax.set_axisbelow(True)
    return names


def fig_curves(rows, ax):
    for n in ORDER:
        rs = [r for r in rows if r["variant"] == n and r.get("loss_hist")]
        for j, r in enumerate(rs):
            ax.semilogy(r["loss_hist"], color=COLOR[n], linewidth=1.2, alpha=1 if j == 0 else 0.5,
                        label=LABEL[n].replace("\n", " ") if j == 0 else None)
    ax.set_xlabel("L-BFGS-B iteration"); ax.set_ylabel("Training loss")
    ax.legend(frameon=False, fontsize=7); ax.grid(color="#EEEEEE", linewidth=0.6)


def vorticity_field(name, pdir, data, ti):
    params, net = C.make_model(name, data, jax.random.PRNGKey(42))
    flat = np.load(os.path.join(pdir, f"{name}_s0.npy"))
    params = ravel_pytree(params)[1](jnp.asarray(flat, dtype=jnp.float32))
    tv = float(data["t"][ti, 0])

    def w(z):
        J = jax.jacfwd(lambda q: net(params, q))(z)   # (3, 3): d(u,v,p)/d(x,y,t)
        return J[1, 0] - J[0, 1]
    xyt = jnp.asarray(np.hstack([data["X"], np.full((data["N"], 1), tv)]), dtype=jnp.float32)
    return np.asarray(jax.jit(jax.vmap(w))(xyt))


def ref_vorticity(data, ti):
    nx, ny = 100, 50
    x = data["X"][:nx, 0]; y = data["X"][::nx, 1]
    u = data["U"][:, 0, ti].reshape(ny, nx); v = data["U"][:, 1, ti].reshape(ny, nx)
    return (np.gradient(v, x, axis=1) - np.gradient(u, y, axis=0)).reshape(-1)


def fig_vorticity(data, pdir, out, ti=100):
    nx, ny = 100, 50
    x = data["X"][:nx, 0]; y = data["X"][::nx, 1]
    fields = [("Reference (CFD)", ref_vorticity(data, ti))]
    for n in ("V0-baseline", "C-identity", "C-mlp311"):
        if os.path.exists(os.path.join(pdir, f"{n}_s0.npy")):
            fields.append((LABEL[n].replace("\n", " "), vorticity_field(n, pdir, data, ti)))
    lim = np.percentile(np.abs(fields[0][1]), 99)
    fig, axs = plt.subplots(len(fields), 1, figsize=(4.2, 1.65 * len(fields)), sharex=True,
                            gridspec_kw={"hspace": 0.45})
    for ax, (t, w) in zip(axs, fields):
        im = ax.pcolormesh(x, y, w.reshape(ny, nx), cmap="RdBu_r", vmin=-lim, vmax=lim,
                           shading="auto", rasterized=True)
        ax.set_aspect("equal"); ax.set_title(t, fontsize=8, loc="left"); ax.set_ylabel("y")
        ax.spines[["top", "right"]].set_visible(True)
    axs[-1].set_xlabel("x")
    fig.colorbar(im, ax=axs, shrink=0.6, label=r"$\omega$")
    fig.savefig(os.path.join(out, "cyl_vorticity.png"), bbox_inches="tight")
    fig.savefig(os.path.join(out, "cyl_vorticity.pdf"), bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", nargs="+", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--data", default="/home/user/geetrakala/qpinn/data/cylinder_wake.mat")
    ap.add_argument("--out", default="figs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = load_rows(a.res)

    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.9], hspace=0.9, wspace=0.35)
    fig_errors(rows, [fig.add_subplot(gs[0, i]) for i in range(3)])
    fig_curves(rows, fig.add_subplot(gs[1, :]))
    fig.savefig(os.path.join(a.out, "cyl_errors.png"), bbox_inches="tight")
    fig.savefig(os.path.join(a.out, "cyl_errors.pdf"), bbox_inches="tight")

    fig_vorticity(C.load_data(a.data), a.params, a.out)


if __name__ == "__main__":
    main()
