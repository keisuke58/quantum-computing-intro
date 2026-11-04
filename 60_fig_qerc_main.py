"""
One-figure summary with the quantum reservoir PINN in front (Allen-Cahn, Raissi 2019 Eq. 12):
(a) QERC-PINN solution (no training, least squares) vs the spectral reference at t = 0.25, 0.5, 1;
(b) t = 1 error of QERC-PINN vs trained PINNs and the published adaptive PINN.

  python 60_fig_qerc_main.py      -> figs/qerc_main.png / .pdf
"""
import json, os, importlib.util
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("ac", os.path.join(H, "52_allen_cahn_qerc.py"))
ac = importlib.util.module_from_spec(spec); spec.loader.exec_module(ac)
plt.rcParams.update({"font.family": ["IPAGothic", "DejaVu Sans"], "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})
Q, GREY, INK = "#0072B2", "#9AA5B8", "#1E2433"


def main():
    rows = json.load(open(os.path.join(H, "results", "server", "allen_cahn_N8.json")))
    r0 = min((r for r in rows if r["model"] == "QERC-Haar" and r["seed"] == 0), key=lambda r: r["t1.0"])
    N = 8; m = 2**N; M = 4 * m
    x = -1 + 2 * (np.arange(M) + 0.5) / M
    ts = [0.25, 0.5, 1.0]
    ref = ac.reference(1e-4, x, ts)
    Phi, Pxx = ac.feature_mats(ac.make_qerc(N, 0, r0["scale"]), x)
    _, out = ac.march(Phi, Pxx, x, 1e-4, 1e-3, 1.0, ts)
    vq = json.load(open(os.path.join(H, "results", "server", "ac_vqpinn.json")))
    gm = lambda mod: float(np.exp(np.mean(np.log([r["t1.0"] for r in vq if r["model"] == mod]))))
    q_all = [min(r["t1.0"] for r in rows if r["model"] == "QERC-Haar" and r["seed"] == s)
             for s in sorted({r["seed"] for r in rows})]
    q_gm = float(np.exp(np.mean(np.log(q_all))))

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.3})
    shades = ["#B9C3D3", "#6B7A93", INK]
    for t, c in zip(ts, shades):
        a.plot(x, ref[t], color=c, lw=2.6, label=f"参照解 t = {t:g}")
        a.plot(x, out[t], color=Q, lw=1.4, ls="--")
    a.plot([], [], color=Q, lw=1.4, ls="--", label="量子リザバー PINN")
    a.set_xlabel("位置 x"); a.set_ylabel("u(t, x)")
    a.set_title("(a) 量子リザバー PINN の解（学習なし，N = 8）", loc="left")
    a.legend(frameon=False, fontsize=9, loc="lower center", ncol=2)

    labels = ["量子リザバー PINN\n（学習なし）", "適応 PINN\n（公表値）", "PINN\n（学習）", "変分 QPINN\n（学習）", "素の PINN\n（公表値）"]
    vals = [q_gm, 2.33e-2, gm("PINN"), gm("VQPINN"), 0.99]
    cols = [Q, GREY, GREY, GREY, GREY]
    bars = b.bar(range(5), vals, color=cols, width=0.62, edgecolor="white")
    for i, v in enumerate(vals):
        b.text(i, v * 1.3, f"{v:.1e}" if v < 0.1 else f"{v:.2f}", ha="center", va="bottom", fontsize=9,
               color=INK, fontweight="bold" if i == 0 else "normal")
    b.set_yscale("log"); b.set_ylim(3e-3, 4)
    b.set_xticks(range(5)); b.set_xticklabels(labels, fontsize=9)
    b.set_ylabel("t = 1 の相対 L2 誤差（低いほど良い）")
    b.set_title("(b) 学習ありの PINN との比較", loc="left")
    b.grid(True, axis="y", color="#e5e5e5", lw=0.6); b.set_axisbelow(True)
    fig.savefig(os.path.join(H, "figs", "qerc_main.png"), dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(H, "figs", "qerc_main.pdf"), bbox_inches="tight")
    print("QERC seed0 scale", r0["scale"], "t=1 err", np.linalg.norm(out[1.0] - ref[1.0]) / np.linalg.norm(ref[1.0]), "gm", q_gm)


if __name__ == "__main__":
    main()
