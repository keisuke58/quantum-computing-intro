"""
Summary of all runs with the authors' TE-QPINN code (67_te_qpinn_authors.py, cluster jobs 21-33):
(a)-(c) relative L2 error against the reference for Burgers, Allen-Cahn and Cahn-Hilliard, per model, seed and
training set-up (authors' 300 outer steps, 1500 steps, and our staged causality "sched");
(d) training loss of Allen-Cahn TE-QPINN over 1500 outer steps: L-BFGS stops changing the parameters early;
(e) error per time slice for Allen-Cahn: the authors' set-up only fits the start (causality weights stay at 0).
Reads whatever results/server/te_authors_*.json exist, so it can be rerun while jobs are still running.

  python 70_fig_te_summary.py     -> figs/te_summary.png / .pdf, table on stdout
"""
import glob, json, os, re
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(H, "results", "server")
plt.rcParams.update({"font.family": ["IPAGothic", "DejaVu Sans"], "font.size": 11, "mathtext.fontset": "cm",
                     "axes.spines.top": False, "axes.spines.right": False})
INK, GREY = "#1E2433", "#5A6273"
# training set-up -> (label, colour, x offset)
VARS = {"500": ("500 ステップ（著者）", "#0072B2", 0.0), "300": ("300 ステップ（著者）", "#9AA3B5", -0.22),
        "1500": ("1500 ステップ", "#0072B2", 0.0), "sched": ("因果性を段階的に（倍精度）", "#009E73", 0.22)}
FIXED = {"burgers": ["TOWER", "NONE", "TOWER-match", "NONE-match"],
         "ac": ["CHEBYSHEV", "NONE", "CHEBYSHEV-match", "NONE-match"], "ch": ["CHEBYSHEV", "NONE", "CHEBYSHEV-match", "NONE-match"]}
QUANTUM = ["RAKALA", "RAKALA-imp", "RAKALA-match", "TE", "LQNN-TE"]
CLASSICAL = {"burgers": ["PINN-2x13", "PINN-3x10", "TE-twin"], "ac": ["PINN-match", "PINN-4x64", "TE-twin"],
             "ch": ["PINN-match", "PINN-4x64", "TE-twin"]}
LABEL = {"TOWER": "固定\nTower", "NONE": "固定\nなし", "CHEBYSHEV": "固定\nCheb.", "TOWER-match": "固定\nTower\n深い",
         "NONE-match": "固定\nなし\n深い", "CHEBYSHEV-match": "固定\nCheb.\n深い", "RAKALA": "Rakala", "RAKALA-imp": "Rakala\nimp.",
         "RAKALA-match": "Rakala\nimp.\n数合わせ", "TE": "TE-\nQPINN", "LQNN-TE": "LQNN-\nTE", "PINN-2x13": "PINN\n2×13",
         "PINN-3x10": "PINN\n3×10", "PINN-match": "PINN\n数合わせ", "PINN-4x64": "PINN\n4×64", "TE-twin": "双子\nモデル"}
TITLE = {"burgers": "(a) Burgers", "ac": "(b) Allen–Cahn", "ch": "(c) Cahn–Hilliard"}


def setup_of(fn):
    """training set-up from the file name: te_authors_<pde>_<TEMODEL>[_sched][_outer<N>].json"""
    pde = os.path.basename(fn).split("_")[2]
    if pde == "burgers":
        return "500"
    if "_sched" in fn:
        return "sched"
    m = re.search(r"_outer(\d+)\.json$", fn)
    return m.group(1) if m else "300"


def load():
    rows = []
    for fn in sorted(glob.glob(os.path.join(R, "te_authors_*.json"))):
        for r in json.load(open(fn)):
            r["setup"] = setup_of(fn)
            rows.append(r)
    return rows


def last_change(h, tol=1e-6):
    """last outer step at which the loss still changed by more than tol (relative)"""
    h = np.asarray(h, float)
    d = np.abs(np.diff(h)) > tol * np.abs(h[1:])
    return int(np.nonzero(d)[0][-1] + 1) if d.any() else 0


def main():
    rows = load()
    sel = lambda pde, m, s: [r for r in rows if r["pde"] == pde and r["model"] == m and r["setup"] == s]
    fig = plt.figure(figsize=(18, 10.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.85], hspace=0.42, wspace=0.22)
    table = []
    for j, pde in enumerate(["burgers", "ac", "ch"]):
        ax = fig.add_subplot(gs[0, j])
        cand = FIXED[pde] + QUANTUM + CLASSICAL[pde]
        models = [m for m in cand if any(r["pde"] == pde and r["model"] == m for r in rows)]
        nq = sum(m not in CLASSICAL[pde] for m in models)
        for i, m in enumerate(models):
            for s, (_, col, dx) in VARS.items():
                v = np.array([r["rel_l2"] for r in sel(pde, m, s)])
                if not len(v):
                    continue
                ax.scatter(i + dx + np.linspace(-0.07, 0.07, len(v)), v, s=14, color=col, alpha=0.55, lw=0, zorder=3)
                ax.plot([i + dx - 0.1, i + dx + 0.1], [v.mean()] * 2, color=col, lw=2.4, zorder=4)
                rr = sel(pde, m, s)
                table.append((pde, m, s, rr[0]["params"], len(v), v.mean(), v.min(), v.max(),
                              np.mean([r["loss"] for r in rr]), np.median([last_change(r["loss_hist"]) for r in rr])))
        ax.axvline(nq - 0.5, color=GREY, lw=0.8, ls="--")
        ax.text((nq - 1) / 2, 1.02, "量子", color=GREY, ha="center", transform=ax.get_xaxis_transform(), fontsize=10)
        ax.text(nq + (len(models) - nq - 1) / 2, 1.02, "古典", color=GREY, ha="center",
                transform=ax.get_xaxis_transform(), fontsize=10)
        ax.set_xticks(range(len(models))); ax.set_xticklabels([LABEL.get(m, m) for m in models], fontsize=8)
        ax.set_yscale("log"); ax.set_ylim(0.03, 10); ax.set_ylabel("相対 $L_2$ 誤差（全時刻）")
        ax.set_title(TITLE[pde], loc="left", pad=18); ax.grid(True, axis="y", color="#e8e8e8", lw=0.6); ax.set_axisbelow(True)
        ax.axhline(1, color=GREY, lw=0.6, ls=":")
        used = sorted({r["setup"] for r in rows if r["pde"] == pde}, key=list(VARS).index)
        ax.legend(handles=[plt.Line2D([], [], color=VARS[s][1], lw=2.4, label=VARS[s][0]) for s in used],
                  frameon=False, fontsize=8.5, loc="upper left")

    # (d) Allen-Cahn TE, 1500 steps: training loss per seed
    ax = fig.add_subplot(gs[1, 0])
    for r in sel("ac", "TE", "1500"):
        ax.plot(np.arange(1, len(r["loss_hist"]) + 1), r["loss_hist"], color="#0072B2", lw=1.2, alpha=0.8)
        k = last_change(r["loss_hist"])
        ax.plot(k, r["loss_hist"][k - 1], "o", color=INK, ms=4, zorder=4)
    ax.axvline(300, color=GREY, lw=0.8, ls="--"); ax.text(310, 0.6, "著者の上限\n300 ステップ", fontsize=8.5, color=GREY,
                                                          transform=ax.get_xaxis_transform())
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("L-BFGS の外側ステップ"); ax.set_ylabel("学習のロス")
    ax.set_title("(d) Allen–Cahn，TE-QPINN の学習（1500 ステップ，5 シード）", loc="left", fontsize=10.5)

    # (e) Allen-Cahn: error per time slice (seed 42)
    ax = fig.add_subplot(gs[1, 1])
    for pat, lab, col, ls in [("ac_TE_outer1500_TE", "TE-QPINN（1500 ステップ）", "#0072B2", "-"),
                              ("ac_classical_outer1500_PINN-4x64", "PINN 4×64（1500 ステップ）", "#D55E00", "-"),
                              ("ac_TE_sched_outer500_TE", "TE-QPINN（段階的）", "#009E73", "-"),
                              ("ac_classical_sched_outer500_PINN-4x64", "PINN 4×64（段階的）", "#009E73", "--"),
                              ("ac_classical_sched_outer500_PINN-match", "PINN 数合わせ（段階的）", "#009E73", ":")]:
        fn = os.path.join(R, f"te_authors_{pat}_fields.npz")
        if not os.path.exists(fn):
            continue
        z = np.load(fn)
        e = np.linalg.norm(z["u"] - z["ref"], axis=1) / np.linalg.norm(z["ref"], axis=1)
        ax.plot(z["t"], e, color=col, ls=ls, lw=1.6, label=lab)
    ax.set_yscale("log"); ax.set_xlabel("時刻 $t$"); ax.set_ylabel("その時刻の相対 $L_2$ 誤差")
    ax.set_title("(e) Allen–Cahn，時刻ごとの誤差（シード 42）", loc="left", fontsize=10.5)
    ax.grid(True, color="#e8e8e8", lw=0.6); ax.legend(frameon=False, fontsize=8.5, loc="lower right")

    # (f) caption-like notes
    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    ax.text(0, 1, "読み方\n"
            "・点は各シード，横線は平均．点線は誤差 1（ゼロの解と同じ）\n"
            "・Allen–Cahn と Cahn–Hilliard は，著者の設定（単精度，\n"
            "  因果性の強さ 200 の 1 段）では 300 でも 1500 でも\n"
            "  ほぼ同じ値：L-BFGS が 60〜220 ステップで止まる\n"
            "  （(d) の黒丸：ロスが変わった最後のステップ）\n"
            "・因果重み $w_i = \\exp(-200\\,\\Sigma_{k<i} L_k)$ が初めの\n"
            "  時刻のロスですぐ 0 になり，それより後の時刻は\n"
            "  学習されない（(e) で t ≈ 0.1 より後の誤差が増える）\n"
            "・33 番（緑）は倍精度で，強さを 0.01 → 100 と上げる",
            va="top", fontsize=9.5, color=INK, transform=ax.transAxes, linespacing=1.55)

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(H, "figs", f"te_summary.{ext}"), dpi=200, bbox_inches="tight", facecolor="white")
    print("| PDE | model | set-up | params | n | relL2 mean | min | max | loss mean | last change (median) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for t in table:
        print(f"| {t[0]} | {t[1]} | {t[2]} | {t[3]} | {t[4]} | {t[5]:.3f} | {t[6]:.3f} | {t[7]:.3f} | {t[8]:.2e} | {t[9]:.0f} |")


if __name__ == "__main__":
    main()
