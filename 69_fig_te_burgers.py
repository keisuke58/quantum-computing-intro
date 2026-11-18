"""
Figure for the Burgers comparison run with the authors' TE-QPINN code (67_te_qpinn_authors.py, cluster job 23 / 26):
(a) final training loss and (b) relative L2 error against the spectral reference, per model and seed;
(c) the solutions at the final time t = 0.95 (seed 42).

  python 69_fig_te_burgers.py     -> figs/te_burgers.png / .pdf
"""
import glob, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.family": ["IPAGothic", "DejaVu Sans"], "font.size": 11, "mathtext.fontset": "cm",
                     "axes.spines.top": False, "axes.spines.right": False})
Q, C, INK, GREY = "#0072B2", "#D55E00", "#1E2433", "#5A6273"
MODELS = [("TOWER", "固定\n5 層\n(60)", Q), ("TOWER-match", "固定\n20 層\n(240)", Q),
          ("TE", "TE-\nQPINN\n(244)", Q), ("PINN-2x13", "PINN\n2×13\n(235)", C),
          ("PINN-3x10", "PINN\n3×10\n(261)", C), ("TE-twin", "双子\nモデル\n(266)", C)]


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(H, "results", "server", "te_authors_burgers_*.json"))):
        rows += json.load(open(f))
    get = lambda m, k: np.array([r[k] for r in rows if r["model"] == m])
    gm = lambda v: float(np.exp(np.mean(np.log(v))))
    fig = plt.figure(figsize=(16.5, 4.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.3)
    for j, (key, ttl, yl) in enumerate([("loss", "(a) 学習の最終ロス", "ロス"),
                                        ("rel_l2", "(b) 参照解との相対 $L_2$ 誤差（全時刻）", "相対 $L_2$ 誤差")]):
        ax = fig.add_subplot(gs[j])
        for i, (m, lab, col) in enumerate(MODELS):
            v = get(m, key)
            ax.scatter(i + np.linspace(-0.16, 0.16, len(v)), v, s=26, color=col, alpha=0.55, lw=0, zorder=3)
            ax.plot([i - 0.3, i + 0.3], [gm(v)] * 2, color=col, lw=2.4, zorder=4)
            ax.text(i, v.max() * (1.35 if key == "loss" else 1.09), f"{gm(v):.1e}" if key == "loss" else f"{gm(v):.2f}",
                    ha="center", va="bottom", fontsize=8.5, color=INK)
        ax.set_yscale("log"); ax.set_xticks(range(len(MODELS))); ax.set_xticklabels([l for _, l, _ in MODELS], fontsize=9)
        ax.set_ylabel(yl); ax.set_title(ttl, loc="left"); ax.grid(True, axis="y", color="#e8e8e8", lw=0.6); ax.set_axisbelow(True)
        ax.text(0.5, -0.24, "量子（青）", color=Q, fontsize=9, ha="right", transform=ax.transAxes)
        ax.text(0.52, -0.24, "古典（橙）　かっこ内はパラメータ数", color=C, fontsize=9, ha="left", transform=ax.transAxes)
        if key == "rel_l2":
            ax.set_ylim(0.05, 1.0)
        else:
            ax.set_ylim(5e-4, 3)
    ax = fig.add_subplot(gs[2])
    fl = lambda m: np.load(glob.glob(os.path.join(H, "results", "server", f"te_authors_burgers_*_{m}_fields.npz"))[0])
    z = fl("TE"); x = z["x"]
    ax.plot(x, z["ref"][-1], color=INK, lw=3, label="参照解")
    for m, lab, col, ls in [("TE", "TE-QPINN", Q, "-"), ("PINN-2x13", "古典 PINN 2 × 13", C, "-"), ("TOWER-match", "固定の埋め込み 20 層", Q, ":")]:
        ax.plot(x, fl(m)["u"][-1], color=col, lw=1.6, ls=ls, label=lab)
    ax.set_xlabel("位置 $x$"); ax.set_ylabel("$u(0.95, x)$"); ax.set_title("(c) 最終時刻の解（シード 42）", loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.savefig(os.path.join(H, "figs", "te_burgers.png"), dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(H, "figs", "te_burgers.pdf"), bbox_inches="tight")
    for m, _, _ in MODELS:
        print(f"{m:12s} loss {gm(get(m, 'loss')):.2e} relL2 {gm(get(m, 'rel_l2')):.3f} [{get(m, 'rel_l2').min():.3f}, {get(m, 'rel_l2').max():.3f}] "
              f"relL2(T) {gm(get(m, 'rel_l2_T')):.3f} MSE {gm(get(m, 'mse_ref')):.2e}")


if __name__ == "__main__":
    main()
