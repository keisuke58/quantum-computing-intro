"""
Summary figure of the least-squares runs (cluster jobs 30, 32, 34-38): for each problem, the final-time relative L2 error
of the plain quantum reservoir, the best re-uploading reservoir and the best classical random features
(geometric mean over seeds, best scale for each model).

  python 71_fig_reservoir_summary.py     -> figs/reservoir_summary.png / .pdf
"""
import collections, glob, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.family": ["IPAGothic", "DejaVu Sans"], "font.size": 12, "mathtext.fontset": "cm",
                     "axes.spines.top": False, "axes.spines.right": False})
Q, C, INK, GREY = "#0072B2", "#D55E00", "#1E2433", "#5A6273"
PROBLEMS = [("AC 1 次元\n$N=8$", "allen_cahn_N8*.json", "t1.0"), ("AC 1 次元\n$N=10$", "allen_cahn_N10*.json", "t1.0"),
            ("CH 1 次元\n$N=8$", "cahn_hilliard_wz_N8*.json", "t1.0"), ("CH 1 次元\n$N=10$", "cahn_hilliard_wz_N10*.json", "t1.0"),
            ("AC 2 次元\n$N=10$", "ac2d_N10*.json", "t20.0"), ("CH 2 次元\n$N=10$", "ch2d_N10*.json", "t0.4")]


def best(pattern, key):
    acc = collections.defaultdict(list)
    for f in sorted(glob.glob(os.path.join(H, "results", "server", pattern))):
        if "data" in f or "_dt_" in f:
            continue
        for r in json.load(open(f)):
            if not r.get("n_data", 0):
                acc[(r["model"], r["scale"])].append(r[key])
    out = {}
    for (m, s), v in acc.items():
        v = np.array(v, float)
        e = float(np.exp(np.mean(np.log(v)))) if np.all(np.isfinite(v)) and np.all(v > 0) else np.inf
        out[m] = min(out.get(m, np.inf), e)
    return out


def main():
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    fmt = lambda v: f"{v:.1e}".replace("e-0", "e-")
    for i, (lab, pat, key) in enumerate(PROBLEMS):
        b = best(pat, key)
        plain = b["QERC-Haar"]
        mr, reup = min(((m, e) for m, e in b.items() if m.startswith("QERC-reup")), key=lambda t: t[1])
        mc, cls = min(((m, e) for m, e in b.items() if not m.startswith("QERC")), key=lambda t: t[1])
        print(f"{lab.replace(chr(10), ' '):22s} plain {plain:.2e}  reup {reup:.2e} ({mr})  classical {cls:.2e} ({mc})  reup/classical {reup / cls:.2f}")
        for dx, v, up, kw in [(-0.26, plain, True, dict(marker="o", facecolor="white", edgecolor=Q, lw=1.8)),
                              (0.0, reup, True, dict(marker="o", facecolor=Q, edgecolor=Q)),
                              (0.26, cls, False, dict(marker="s", facecolor=C, edgecolor=C))]:
            ax.scatter(i + dx, v, s=90, zorder=3, label=None, **kw)
            ax.text(i + dx, v * 1.5 if up else v / 1.55, fmt(v), ha="center", va="bottom" if up else "top", fontsize=9.5, color=INK)
    ax.scatter([], [], s=90, marker="o", facecolor="white", edgecolor=Q, lw=1.8, label="量子リザバー（入力 1 回）")
    ax.scatter([], [], s=90, marker="o", facecolor=Q, edgecolor=Q, label="量子リザバー（入れ直し，最良）")
    ax.scatter([], [], s=90, marker="s", facecolor=C, edgecolor=C, label="古典の乱数特徴量（最良）")
    ax.set_yscale("log"); ax.set_ylim(1e-7, 0.4)
    ax.set_xticks(range(len(PROBLEMS))); ax.set_xticklabels([p[0] for p in PROBLEMS])
    ax.set_ylabel("最終時刻の相対 $L_2$ 誤差"); ax.grid(True, axis="y", color="#e8e8e8", lw=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=10.5, loc="lower left", bbox_to_anchor=(1.0, 0.35))
    fig.savefig(os.path.join(H, "figs", "reservoir_summary.png"), dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(H, "figs", "reservoir_summary.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    main()
