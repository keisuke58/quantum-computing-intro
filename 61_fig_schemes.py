"""
Schematic workflow diagrams, one per method (publication style, vector PDF + PNG):
  scheme_qrc   quantum reservoir PINN     fixed encoding + fixed reservoir U, outcome probabilities, least squares
  scheme_rf    classical random features  fixed random layer, least squares
  scheme_vq    variational QPINN          trainable circuit, <Z_i>, physics loss, Adam
  scheme_pinn  classical PINN             trainable MLP, physics loss, Adam

  python 61_fig_schemes.py   -> figs/scheme_*.pdf / .png
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch

H = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.family": ["IPAexGothic", "IPAGothic", "DejaVu Sans"], "font.size": 9,
                     "mathtext.fontset": "cm", "pdf.fonttype": 42})
INK, FIX, LS, LEARN = "#1E2433", "#5A6273", "#00805A", "#B4457F"
FIXF, LSF, LEARNF = "#EEF1F5", "#E3F3EC", "#F8E6F0"
WIRES = [6.0, 4.9, 3.8, 1.4]          # qubit wires / feature rows (last one = q_N)
YMID = 3.7


# ---------------------------------------------------------------- primitives
def box(ax, x, y, w, h, title, sub=None, ec=INK, ls="-", fc="none"):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=1.1, ls=ls, zorder=0))
    ax.text(x + w / 2, y + h + 0.42, title, ha="center", va="bottom", fontsize=10, color=INK)
    if sub:
        ax.text(x + w / 2, y + h + 0.12, sub, ha="center", va="bottom", fontsize=8, color=ec)


def arrow(ax, p0, p1, color=INK, lw=1.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9, color=color, lw=lw, ls=ls,
                                 shrinkA=0, shrinkB=0, zorder=2))


def node(ax, x, y, label, r=0.36, fs=9, ec=INK, fc="white"):
    ax.add_patch(Circle((x, y), r, fc=fc, ec=ec, lw=0.9, zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=fs, zorder=4)


def vdots(ax, x, y, color=INK):
    for d in (-0.2, 0, 0.2):
        ax.add_patch(Circle((x, y + d), 0.04, color=color, zorder=3))


def gate(ax, x, y, label, ec, fc, w=1.0, h=0.56, fs=7.5):
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fc=fc, ec=ec, lw=0.9, zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=fs, zorder=4)


def meter(ax, x, y):
    ax.add_patch(Rectangle((x - 0.28, y - 0.24), 0.56, 0.48, fc="white", ec=INK, lw=0.9, zorder=3))
    t = np.linspace(0.15 * np.pi, 0.85 * np.pi, 20)
    ax.plot(x + 0.19 * np.cos(t), y - 0.1 + 0.19 * np.sin(t), color=INK, lw=0.7, zorder=4)
    ax.plot([x, x + 0.14], [y - 0.1, y + 0.13], color=INK, lw=0.7, zorder=4)


def cnot(ax, x, yc, yt):
    ax.plot([x, x], [yc, yt], color=INK, lw=0.9, zorder=3)
    ax.add_patch(Circle((x, yc), 0.07, color=INK, zorder=4))
    ax.add_patch(Circle((x, yt), 0.15, fc="white", ec=INK, lw=0.9, zorder=4))
    ax.plot([x - 0.15, x + 0.15], [yt, yt], color=INK, lw=0.8, zorder=5)
    ax.plot([x, x], [yt - 0.15, yt + 0.15], color=INK, lw=0.8, zorder=5)


def inputs(ax, labels, note):
    ax.add_patch(Rectangle((0.3, 2.3), 1.5, 3.0, fc="none", ec=INK, ls=":", lw=0.9))
    for y, l in zip((4.55, 3.05), labels):
        node(ax, 1.05, y, l, r=0.5, fs=8)
    ax.text(1.05, 2.05, note, ha="center", va="top", fontsize=8)
    for y in (4.55, 3.05):
        arrow(ax, (1.55, y), (2.6, y if y > 4 else 3.05))


def wires(ax, x0, x1, labels=True):
    for i, y in enumerate(WIRES):
        ax.plot([x0, x1], [y, y], color=INK, lw=0.8, zorder=1)
        if labels:
            ax.text(x0 - 0.12, y, f"$q_{i+1}$" if i < 3 else "$q_N$", ha="right", va="center", fontsize=8)
    vdots(ax, (x0 + x1) / 2 - 2.5, 2.6)


def features(ax, x, labels, title, sub, ec):
    box(ax, x - 0.95, 0.3, 1.9, 6.45, title, sub, ec, "--" if ec == FIX else "-")
    for y, l in zip(WIRES, labels):
        node(ax, x, y, l, r=0.46, fs=8)
    vdots(ax, x, 2.6)


def readout(ax, xf, xu, color, formula):
    for y in WIRES:
        ax.plot([xf + 0.46, xu - 0.42], [y, YMID], color=color, lw=0.8, zorder=1)
    node(ax, xu, YMID, "$u$", r=0.42, fs=11)
    ax.text(xu + 0.25, 1.75, formula, ha="center", va="top", fontsize=8.5, color=color)


def solver_ls(ax, x, y):
    box(ax, x, y, 6.6, 5.0, "解き方：最小二乗", "学習なし", LS)
    ax.add_patch(FancyBboxPatch((x + 0.3, y + 3.0), 6.0, 1.5, boxstyle="round,pad=0.04", fc="white", ec=INK, lw=0.8))
    ax.text(x + 3.3, y + 3.75, "PDE の残差・初期条件・境界条件は\n係数 $\\boldsymbol{c}$ について線形", ha="center",
            va="center", fontsize=8)
    arrow(ax, (x + 3.3, y + 2.95), (x + 3.3, y + 2.45))
    ax.add_patch(FancyBboxPatch((x + 0.3, y + 0.95), 6.0, 1.45, boxstyle="round,pad=0.04", fc=LSF, ec=LS, lw=0.9))
    ax.text(x + 3.3, y + 1.67, "$\\boldsymbol{c}=\\arg\\min_{\\boldsymbol{c}}\\ \\|A\\boldsymbol{c}-\\boldsymbol{b}\\|_2^2$\n"
            "時間ステップ・Newton 反復ごとに 1 回", ha="center", va="center", fontsize=8)
    ax.text(x + 3.3, y + 0.45, "勾配法の反復なし", ha="center", va="center", fontsize=8.5, color=LS)


def solver_train(ax, x, y):
    box(ax, x, y, 6.6, 5.0, "解き方：物理の損失を学習", "勾配法（Adam）", LEARN)
    for i, t in enumerate(["$\\mathcal{L}_{\\rm PDE}$：選点での残差", "$\\mathcal{L}_{\\rm IC}$：初期条件"]):
        yy = y + 4.0 - i * 1.05
        ax.add_patch(FancyBboxPatch((x + 0.3, yy - 0.33), 6.0, 0.66, boxstyle="round,pad=0.04", fc="white", ec=INK, lw=0.8))
        ax.text(x + 3.3, yy, t, ha="center", va="center", fontsize=8.5)
    ax.add_patch(FancyBboxPatch((x + 0.3, y + 0.55), 6.0, 1.45, boxstyle="round,pad=0.04", fc=LEARNF, ec=LEARN, lw=0.9))
    ax.text(x + 3.3, y + 1.27, "$\\mathcal{L}=\\mathcal{L}_{\\rm PDE}+w\\,\\mathcal{L}_{\\rm IC}$ を最小化\n"
            "$\\boldsymbol{\\theta}\\leftarrow\\boldsymbol{\\theta}-\\eta\\,\\nabla_{\\boldsymbol{\\theta}}\\mathcal{L}$（$2\\times10^4$ 回）",
            ha="center", va="center", fontsize=8.5)


def feedback(ax, targets, x_from, label, color=LEARN):
    """targets: list of (x, y_tip)"""
    yb = -0.45
    ax.plot([x_from, x_from], [1.2, yb], color=color, lw=1.1)
    ax.plot([min(t[0] for t in targets), x_from], [yb, yb], color=color, lw=1.1)
    for xt, yt in targets:
        arrow(ax, (xt, yb), (xt, yt), color, lw=1.1)
    ax.text((min(t[0] for t in targets) + x_from) / 2, yb - 0.15, label, ha="center", va="top", fontsize=8.5,
            color=color)


def legend(ax, items):
    x = 0.3
    for txt, col, ls in items:
        ax.plot([x, x + 0.8], [-1.35, -1.35], color=col, lw=1.4, ls=ls)
        ax.text(x + 1.0, -1.35, txt, va="center", fontsize=8.5, color=col)
        x += 7.5


def finish(fig, ax, name):
    ax.set_xlim(0, 25.4); ax.set_ylim(-1.75, 7.9); ax.set_aspect("equal"); ax.axis("off")
    fig.savefig(os.path.join(H, "figs", f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(H, "figs", f"{name}.png"), dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)


ZIN = (["$\\cos\\pi x$", "$\\sin\\pi x$"], "入力 $\\boldsymbol{z}(x)$\n（周期境界を満たす）")
XT = (["$x$", "$t$"], "入力（時間と空間を\nまとめて扱う）")


# ---------------------------------------------------------------- the four methods
def quantum_reservoir():
    fig, ax = plt.subplots(figsize=(12.5, 4.6))
    inputs(ax, *ZIN)
    box(ax, 2.6, 0.3, 9.6, 6.45, "量子リザバー", "回路はすべて乱数で固定", FIX, "--")
    wires(ax, 3.4, 11.5)
    for y in WIRES:
        gate(ax, 4.45, y, "$R_Y(\\theta_l)$", FIX, FIXF, w=1.15)
        gate(ax, 5.85, y, "$P(\\theta_{N+l})$", FIX, FIXF, w=1.3)
        meter(ax, 10.95, y)
    ax.text(5.15, 0.45, "エンコーディング\n$\\boldsymbol{\\theta}=\\pi(\\tanh(W\\boldsymbol{z}+\\boldsymbol{b})+1)/2$",
            ha="center", va="bottom", fontsize=7.5, color=FIX)
    ax.add_patch(Rectangle((7.0, 1.0), 2.9, 5.35, fc=FIXF, ec=FIX, lw=1.0, zorder=3))
    ax.text(8.45, 3.72, "$U$\nHaar 乱数\nユニタリ", ha="center", va="center", fontsize=9, zorder=4)
    arrow(ax, (12.2, YMID), (13.05, YMID))
    ax.text(12.62, YMID + 0.2, "測定", ha="center", va="bottom", fontsize=7.5)
    features(ax, 14.0, ["$p_1$", "$p_2$", "$p_3$", "$p_{2^N}$"], "特徴量", "測定確率 $p_k$", FIX)
    readout(ax, 14.0, 16.6, LS, "$u=\\sum_k c_k\\,p_k$")
    arrow(ax, (17.05, YMID), (18.35, YMID))
    solver_ls(ax, 18.4, 1.2)
    feedback(ax, [(16.85, 0.85)], 21.7, "求めた係数 $c_k$ を出力に使う（回路は評価し直さない）", LS)
    legend(ax, [("破線：乱数で固定", FIX, "--"), ("緑：最小二乗で決める（係数 $c_k$ のみ）", LS, "-")])
    finish(fig, ax, "scheme_qrc")


def random_features():
    fig, ax = plt.subplots(figsize=(12.5, 4.6))
    inputs(ax, *ZIN)
    box(ax, 2.6, 0.3, 9.6, 6.45, "乱数の 1 層（古典）", "重み $\\boldsymbol{w}_k,b_k$ はすべて乱数で固定", FIX, "--")
    for i, y in enumerate(WIRES):
        k = f"{i+1}" if i < 3 else "m"
        ax.plot([4.0, 11.5], [y, y], color=INK, lw=0.8, zorder=1)
        ax.text(3.9, y, f"$k={k}$", ha="right", va="center", fontsize=8)
        gate(ax, 5.6, y, f"$\\boldsymbol{{w}}_{{{k}}}\\cdot\\boldsymbol{{z}}+b_{{{k}}}$", FIX, FIXF, w=1.9)
        gate(ax, 8.6, y, "$\\sigma(\\cdot)$", FIX, FIXF, w=1.1)
    vdots(ax, 5.6, 2.6); vdots(ax, 8.6, 2.6)
    ax.text(5.6, 0.45, "乱数の線形変換", ha="center", va="bottom", fontsize=7.5, color=FIX)
    ax.text(8.6, 0.45, "活性化 $\\sigma=\\tanh$ または $\\cos$", ha="center", va="bottom", fontsize=7.5, color=FIX)
    arrow(ax, (12.2, YMID), (13.05, YMID))
    features(ax, 14.0, ["$\\varphi_1$", "$\\varphi_2$", "$\\varphi_3$", "$\\varphi_m$"], "特徴量", "$m=2^N$ 個", FIX)
    readout(ax, 14.0, 16.6, LS, "$u=\\sum_k c_k\\,\\varphi_k$")
    arrow(ax, (17.05, YMID), (18.35, YMID))
    solver_ls(ax, 18.4, 1.2)
    feedback(ax, [(16.85, 0.85)], 21.7, "求めた係数 $c_k$ を出力に使う（特徴量は計算し直さない）", LS)
    legend(ax, [("破線：乱数で固定", FIX, "--"), ("緑：最小二乗で決める（係数 $c_k$ のみ）", LS, "-")])
    finish(fig, ax, "scheme_rf")


def variational_qpinn():
    fig, ax = plt.subplots(figsize=(12.5, 4.6))
    inputs(ax, *XT)
    box(ax, 2.6, 0.3, 9.6, 6.45, "変分量子回路", "回転角を学習（データ再アップロード，$L$ 層）", LEARN)
    wires(ax, 3.4, 11.5)
    for y in WIRES:
        gate(ax, 4.35, y, "$R_Y(\\boldsymbol{w}\\!\\cdot\\!\\boldsymbol{a}+b)$", LEARN, LEARNF, w=1.45, fs=7)
        gate(ax, 5.85, y, "$R_Z R_Y(\\theta)$", LEARN, LEARNF, w=1.2, fs=7)
        meter(ax, 10.95, y)
    for k, (yc, yt) in enumerate(((6.0, 4.9), (4.9, 3.8))):
        cnot(ax, 6.85 + 0.35 * k, yc, yt)
    ax.add_patch(Rectangle((3.5, 0.95), 4.1, 5.5, fc="none", ec=LEARN, ls=":", lw=0.9, zorder=2))
    ax.text(5.55, 0.45, "1 層（これを $L=6$ 回くり返す）", ha="center", va="bottom", fontsize=7.5, color=LEARN)
    ax.text(9.3, 2.6, "$\\times\\,L$", ha="center", va="center", fontsize=11, color=LEARN)
    ax.text(9.9, 0.45, "$\\boldsymbol{a}=(\\cos\\pi x,\\sin\\pi x,2t-1)$", ha="center", va="bottom", fontsize=7.5,
            color=LEARN)
    arrow(ax, (12.2, YMID), (13.05, YMID))
    ax.text(12.62, YMID + 0.2, "測定", ha="center", va="bottom", fontsize=7.5)
    features(ax, 14.0, ["$\\langle Z_1\\rangle$", "$\\langle Z_2\\rangle$", "$\\langle Z_3\\rangle$", "$\\langle Z_N\\rangle$"],
             "期待値", "パウリ $Z$ の測定", INK)
    readout(ax, 14.0, 16.6, LEARN, "$u=\\sum_i h_i\\langle Z_i\\rangle$")
    arrow(ax, (17.05, YMID), (18.35, YMID))
    solver_train(ax, 18.4, 1.2)
    feedback(ax, [(9.3, 0.3), (16.85, 0.85)], 21.7, "回転角 $\\boldsymbol{\\theta}$ と出力の重み $h$ を更新")
    legend(ax, [("赤紫：勾配法で学習する部分", LEARN, "-")])
    finish(fig, ax, "scheme_vq")


def classical_pinn():
    fig, ax = plt.subplots(figsize=(12.5, 4.6))
    inputs(ax, *XT)
    box(ax, 2.6, 0.3, 14.9, 6.45, "ニューラルネットワーク（古典）", "全ての重み $\\boldsymbol{\\theta}$ を学習（4 層 × 64，tanh）", LEARN)
    layers_x = [4.2, 6.9, 9.6, 12.3]
    hid = [6.0, 4.9, 3.8, 1.4]
    prev = [(2.6, 4.55), (2.6, 3.05)]
    for lx in layers_x:
        cur = [(lx, y) for y in hid]
        for (xa, ya) in prev:
            for (xb, yb) in cur:
                ax.plot([xa, xb], [ya, yb], color=LEARN, lw=0.35, alpha=0.8, zorder=1)
        prev = cur
    for (xa, ya) in prev:
        ax.plot([xa, 16.18], [ya, YMID], color=LEARN, lw=0.6, zorder=1)
    for lx in layers_x:
        for y in hid:
            node(ax, lx, y, "", r=0.3, ec=INK)
        vdots(ax, lx, 2.6)
    ax.text(8.25, 6.4, "隠れ層（各 64 ユニット，活性化 tanh）", ha="center", va="bottom", fontsize=7.5, color=LEARN)
    node(ax, 16.6, YMID, "$u$", r=0.42, fs=11)
    ax.text(16.6, YMID - 0.6, "$u(x,t)$", ha="center", va="top", fontsize=9)
    arrow(ax, (17.05, YMID), (18.35, YMID))
    solver_train(ax, 18.4, 1.2)
    feedback(ax, [(9.6, 0.3)], 21.7, "全ての重み $\\boldsymbol{\\theta}$ を更新")
    legend(ax, [("赤紫：勾配法で学習する部分", LEARN, "-")])
    finish(fig, ax, "scheme_pinn")


if __name__ == "__main__":
    quantum_reservoir(); random_features(); variational_qpinn(); classical_pinn()
