# -*- coding: utf-8 -*-
"""
24 - QERC の発見を図にする: 「リザバーの中身は線形読み出しで表せる関数を変えない」

  (a) 特徴量行列のランク vs qubit 数 N（2^N，1 つのリザバー，2 つのリザバーを連結）
  (b) PDE のみで解いたときの誤差 vs イジング時間 t（QERC は t に依存しない）
  (c) 教師ありフィットと PDE の誤差（QERC vs 同数の古典ランダム特徴量，N=8）
"""
import importlib.util, os
import numpy as np
import jax, jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)
_here = os.path.dirname(os.path.abspath(__file__))


def _load(name, fn):
    s = importlib.util.spec_from_file_location(name, os.path.join(_here, fn))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


M = _load("m", "22_qerc_pinn_kirsch.py")
Q = M.Q
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300})
QC, CC, GC = "#1F3A68", "#B5651D", "#8C8C8C"


def rank(F):
    sv = np.linalg.svd(np.asarray(F), compute_uv=False)
    return int((sv > sv[0] * 1e-10).sum())


def panel_rank(ax, Ns=(2, 4, 6, 8, 10)):
    g = np.random.default_rng(0); n = 4000
    r = np.sqrt(1 + g.random(n) * 15); th = g.random(n) * 2 * np.pi
    x, y = jnp.asarray(r * np.cos(th)), jnp.asarray(r * np.sin(th))
    one, two = [], []
    for N in Ns:
        F1 = jax.vmap(M.make_qerc(N, 0, 1.0)[0])(x, y)
        F2 = jax.vmap(M.make_qerc(N, 7, 2.3)[0])(x, y)
        one.append(rank(F1)); two.append(rank(np.c_[F1, F2]))
        print(f"N={N}: 2^N={2**N}, rank={one[-1]}, two reservoirs={two[-1]}", flush=True)
    ax.semilogy(Ns, [2**N for N in Ns], "--", color=GC, label="$2^N$ (nominal)")
    ax.semilogy(Ns, two, "s-", color=CC, ms=5, label="two reservoirs combined")
    ax.semilogy(Ns, one, "o-", color=QC, ms=5, label="one reservoir")
    ax.set_xlabel("Number of qubits N"); ax.set_ylabel("Rank of feature matrix")
    ax.set_title("(a) Independent features", loc="left")
    ax.legend(frameon=False, fontsize=7, loc="upper left")


def pde_err(feat, n_in=8000, n_bc=800, seed=100):
    fns = M.make_basis(feat)
    A, b = M.assemble(fns, *M.points(n_in, n_bc, seed), 1.0)
    return M.evaluate(fns, M.solve(A, b, 1e-10))[0]


def panel_time(ax, ts=(0.1, 0.3, 1, 3, 10)):
    for sd, ls in ((0, "o-"), (1, "^--")):
        e = [pde_err(M.make_qerc(8, sd, t)[0]) for t in ts]
        ax.semilogx(ts, e, ls, color=QC, ms=5, label=f"QERC N=8, seed {sd}")
        print("t sweep", sd, e, flush=True)
    ec = [pde_err(M.make_crf(256, sd)[0]) for sd in (0, 1)]
    ax.axhspan(min(ec), max(ec), color=CC, alpha=0.25, lw=0, label="classical RF (256)")
    ax.set_yscale("log"); ax.set_xlabel("Ising evolution time t")
    ax.set_ylabel("Relative $L_2$ error of u"); ax.set_title("(b) PDE solve vs t", loc="left")
    ax.legend(frameon=False, fontsize=7, loc="center right")


def panel_bars(ax):
    tr, te = Q.data(3000, 0), Q.data(2000, 1)
    sup_q, sup_c, pde_q, pde_c = [], [], [], []
    for sd in (0, 1):
        U = Q.ising_U(8, sd)
        sup_q.append(Q.fit_err(Q.qerc_feats(tr[2], tr[1], 8, U), Q.qerc_feats(te[2], te[1], 8, U), tr, te))
        sup_c.append(Q.fit_err(Q.crf_feats(tr[2], tr[1], 256, sd), Q.crf_feats(te[2], te[1], 256, sd), tr, te))
        pde_q.append(pde_err(M.make_qerc(8, sd)[0])); pde_c.append(pde_err(M.make_crf(256, sd)[0]))
    vals = [(sup_q, sup_c), (pde_q, pde_c)]
    for i, (q, c) in enumerate(vals):
        for j, (v, col, lab) in enumerate(((q, QC, "QERC (N=8)"), (c, CC, "classical RF (256)"))):
            ax.bar(i + (j - 0.5) * 0.36, np.mean(v), 0.34, color=col, label=lab if i == 0 else None)
            ax.scatter(np.full(2, i + (j - 0.5) * 0.36), v, s=10, color="#222222", zorder=3)
    ax.set_yscale("log"); ax.set_xticks([0, 1]); ax.set_xticklabels(["fit to exact\nsolution", "PDE only"])
    ax.set_ylabel("Relative $L_2$ error of u"); ax.set_title("(c) Supervised vs PDE", loc="left")
    ax.set_ylim(2e-5, 3); ax.legend(frameon=False, fontsize=7, loc="upper left")
    print("bars", vals, flush=True)


def main():
    os.makedirs(os.path.join(_here, "figs"), exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.5), gridspec_kw={"wspace": 0.45})
    panel_rank(axs[0]); panel_time(axs[1]); panel_bars(axs[2])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(_here, "figs", f"qerc_reservoir.{ext}"), bbox_inches="tight")


if __name__ == "__main__":
    main()
