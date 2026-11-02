# -*- coding: utf-8 -*-
"""36 - TSV 周期配列（34）で，ピッチ L が広いと Fourier が劣化する原因を調べる

  (A) L = 8, 10 に拡張（34 と同じ設定）
  (B) L = 6 で Si 側の選点を 2 倍（点密度の不足か）
  (C) L = 6 で特徴量を 2^10 = 1024 個に（表現力の不足か）
"""
import argparse, importlib.util, json, time
import numpy as np

spec = importlib.util.spec_from_file_location("P", "34_tsv_periodic.py")
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)


def grid(N, L):
    m = 2**N
    return {
        "QERC-Haar": ([1.0, 2.0, 4.0], P.scaled(lambda sd, s: P.T.make_qerc(N, sd, s), L)),
        "MLP2-RF": ([1.0, 2.0, 4.0], P.scaled(lambda sd, s: P.C.make_mlp2(N, sd, s), L)),
        "Fourier": ([4.0, 8.0, 16.0, 24.0], P.scaled(lambda sd, s: P.T.make_fourier(m, sd, s), L)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="A,B,C")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--h", type=float, default=0.005)
    ap.add_argument("--out", default="results/tsv_periodic_scaling.json")
    a = ap.parse_args()
    # (実験名, L, N, Si 選点数)
    plan = []
    if "A" in a.exp:
        plan += [("A", 8.0, 8, 6000), ("A", 10.0, 8, 6000)]
    if "B" in a.exp:
        plan += [("B", 6.0, 8, 12000)]
    if "C" in a.exp:
        plan += [("C", 6.0, 10, 6000)]
    rows, refs = [], {}
    for exp, L, N, n_si in plan:
        if L not in refs:
            t0 = time.time(); refs[L] = P.fe_solve(L, a.h)
            print(f"L={L}: FE 要素 {len(refs[L]['tri'])}  {time.time()-t0:.0f}s", flush=True)
        for sd in range(a.seeds):
            pts = P.collocation(L, 2000, n_si, 300, 200, 10 + sd)
            for name, (scales, ctor) in grid(N, L).items():
                for s in scales:
                    t0 = time.time()
                    eu, es = P.run(ctor(2 * sd, s), ctor(2 * sd + 1, s), pts, refs[L], N >= 10)
                    rows.append({"exp": exp, "L": L, "N": N, "n_si": n_si, "model": name, "seed": sd,
                                 "scale": s, "u": eu, "s": es})
                    print(f"{exp} L={L} N={N} n_si={n_si} seed={sd} {name:10s} ×{s:<4} u={eu:.2e} σ={es:.2e} "
                          f"{time.time()-t0:.0f}s", flush=True)
                    json.dump(rows, open(a.out, "w"), indent=1)
    print("\n=== シードごとに最良の倍率（u の幾何平均）===")
    for exp, L, N, n_si in plan:
        for name in ["QERC-Haar", "MLP2-RF", "Fourier"]:
            best = []
            for sd in range(a.seeds):
                rr = [r for r in rows if r["exp"] == exp and r["L"] == L and r["model"] == name and r["seed"] == sd]
                if rr:
                    best.append(min(rr, key=lambda r: r["u"]))
            if best:
                print(f"{exp} L={L} N={N} n_si={n_si} {name:10s} {np.exp(np.mean(np.log([b['u'] for b in best]))):.2e}"
                      f"  倍率 {[b['scale'] for b in best]}")


if __name__ == "__main__":
    main()
