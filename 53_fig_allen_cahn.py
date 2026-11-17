"""
Figure for the Allen-Cahn test (52_allen_cahn_qerc.py): reference u(t, x), snapshots, and error versus time
for the best scale of each model (seed 0). Reads the result files, re-runs only the best configurations.

  python 53_fig_allen_cahn.py --D 1e-4 --N 8 results/allen_cahn_D1e-4_N8.json [more.json ...]
"""
import argparse, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import importlib.util

_here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("ac", os.path.join(_here, "52_allen_cahn_qerc.py"))
ac = importlib.util.module_from_spec(spec); spec.loader.exec_module(ac)

COL = {"QERC-Haar": "#0072B2", "tanh-RF": "#E69F00", "Fourier": "#009E73"}
LAB = {"QERC-Haar": "Quantum reservoir (Haar)", "tanh-RF": "tanh random features", "Fourier": "Fourier random features"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--D", type=float, default=1e-4)
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--dt", type=float, default=1e-3)
    ap.add_argument("--out", default="figs/allen_cahn")
    a = ap.parse_args()
    rows = sum((json.load(open(f)) for f in a.files), [])
    rows = [r for r in rows if r["seed"] == 0 and r["N"] == a.N and np.isfinite(r["t1.0"])]
    best = {m: min((r for r in rows if r["model"] == m), key=lambda r: r["t1.0"]) for m in COL
            if any(r["model"] == m for r in rows)}
    m = 2**a.N; M = 4 * m
    x = -1 + 2 * (np.arange(M) + 0.5) / M
    ts = [round(t, 3) for t in np.arange(0.05, 1.0001, 0.05)]
    ref = ac.reference(a.D, x, ts)
    ctor = {"QERC-Haar": lambda s: ac.make_qerc(a.N, 0, s), "tanh-RF": lambda s: ac.make_tanh(m, 0, s),
            "Fourier": lambda s: ac.make_fourier(m, 0, s)}
    sol, err = {}, {}
    for name, r in best.items():
        Phi, Pxx = ac.feature_mats(ctor[name](r["scale"]), x)
        _, out = ac.march(Phi, Pxx, x, a.D, a.dt, 1.0, ts)
        sol[name] = out
        err[name] = [np.linalg.norm(out[t] - ref[t]) / np.linalg.norm(ref[t]) for t in ts]
        print(name, "x", r["scale"], " ".join(f"{e:.1e}" for e in err[name]))

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(11, 3.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.3, 1.0], wspace=0.32)
    # (a) reference
    ax = fig.add_subplot(gs[0])
    U = np.array([ref[t] for t in ts])
    im = ax.imshow(U.T, origin="lower", aspect="auto", extent=[0.025, 1.025, -1, 1], cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xlabel("t"); ax.set_ylabel("x"); ax.set_title("(a) reference u(t, x)", loc="left")
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    # (b) snapshot at t = 1 near an interface
    ax = fig.add_subplot(gs[1])
    ax.plot(x, ref[1.0], color="#222222", lw=2.4, label="reference")
    for name in sol:
        ax.plot(x, sol[name][1.0], color=COL[name], lw=1.4, ls="--", label=LAB[name])
    ax.set_xlabel("x"); ax.set_ylabel("u(1, x)"); ax.set_title("(b) solution at t = 1", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower center")
    # (c) error versus time
    ax = fig.add_subplot(gs[2])
    for name in err:
        ax.semilogy(ts, err[name], color=COL[name], lw=2, marker="o", ms=3, label=LAB[name])
    ax.set_xlabel("t"); ax.set_ylabel("relative L2 error of u"); ax.set_title("(c) error versus time", loc="left")
    ax.grid(True, which="major", color="#e5e5e5", lw=0.6)
    fig.savefig(a.out + ".png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(a.out + ".pdf", bbox_inches="tight")
    json.dump({"ts": ts, "err": err, "best": {k: v["scale"] for k, v in best.items()}},
              open(a.out + "_curves.json", "w"), indent=1)


if __name__ == "__main__":
    main()
