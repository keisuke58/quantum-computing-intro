# -*- coding: utf-8 -*-
"""
26 - 25 の結果を 5 シードで確かめ，古典ランダム特徴量の重みスケールも調整する（公平な比較）

量子側は 25 と同じ設定のまま（調整なし）。古典側はスケール {0.5, 1, 1.5, 3, 5} を全部試し，
シードごとに「テスト誤差が最良のスケール」を採用する（古典側に有利な比較）。
"""
import argparse, importlib.util, json, os, time
import numpy as np
import jax, jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
_h = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("i", os.path.join(_h, "25_qerc_ikeda_pde.py"))
I = importlib.util.module_from_spec(_s); _s.loader.exec_module(I)
M = I.M


def make_crf(m, seed, scale):
    g = np.random.default_rng(seed)
    W = jnp.asarray(g.normal(size=(3, m)) * scale); b = jnp.asarray(g.random(m) * 2 * np.pi)

    def feat(x, y):
        _, th, a1 = M._pre(x, y)
        return jnp.tanh(jnp.stack([a1, jnp.cos(th), jnp.sin(th)]) @ W + b)
    return feat


def solve_eval(feat, pts):
    fns = M.make_basis(feat)
    A, b = M.assemble(fns, *pts, 1.0)
    return M.evaluate(fns, M.solve(A, b, 1e-10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--kinds", default="cliffordT,haar,ising")
    ap.add_argument("--scales", default="0.5,1,1.5,3,5")
    ap.add_argument("--out", default="results/qerc_ikeda_seeds.json")
    a = ap.parse_args()
    rows = []
    for sd in range(a.seeds):
        pts = M.points(8000, 800, 100 + sd)
        for kind in a.kinds.split(","):
            t0 = time.time()
            e = solve_eval(I.make_ikeda(a.N, sd, kind)[0], pts)
            rows.append({"model": kind, "seed": sd, "u": e[0], "s": e[1], "kt": e[2]})
            print(f"seed={sd} {kind:9s} u={e[0]:.2e} σ={e[1]:.2e} K_t={e[2]:.3f} {time.time()-t0:.0f}s", flush=True)
        for sc in map(float, a.scales.split(",")):
            e = solve_eval(make_crf(2**a.N, sd, sc), pts)
            rows.append({"model": f"crf{sc}", "seed": sd, "u": e[0], "s": e[1], "kt": e[2]})
            print(f"seed={sd} crf×{sc:<4} u={e[0]:.2e} σ={e[1]:.2e} K_t={e[2]:.3f}", flush=True)
        json.dump(rows, open(a.out, "w"), indent=1)

    print("\n=== 集計（u の幾何平均 [最小, 最大]）===")
    def stat(v):
        v = np.array(v); return f"{np.exp(np.log(v).mean()):.2e} [{v.min():.1e}, {v.max():.1e}]"
    for k in a.kinds.split(",") + [f"crf{float(s)}" for s in a.scales.split(",")]:
        print(f"{k:10s} {stat([r['u'] for r in rows if r['model'] == k])}")
    best = [min((r for r in rows if r["seed"] == s and r["model"].startswith("crf")), key=lambda r: r["u"])["u"]
            for s in range(a.seeds)]
    print(f"{'crf(best)':10s} {stat(best)}  ← シードごとに最良スケールを採用")


if __name__ == "__main__":
    main()
