# -*- coding: utf-8 -*-
"""
15 - QPINN の対照実験: パラメータ効率は本物か、量子回路は寄与しているか

14_qpinn_dem_thermo.py で「166 パラメータのハイブリッド QPINN が 2274 パラメータの
MLP と同じ精度」という結果が出た。査読者が真っ先に突くのは次の2点なので、それを潰す。

1. **同じパラメータ数の古典モデルとの比較**
   2274 パラメータの MLP と比べてもパラメータ効率の証明にはならない。
   ~160 パラメータの小さい MLP / Fourier feature モデルが同じ精度を出すなら、主張は消える。

2. **量子回路（特にもつれ）が寄与しているか**
   - CNOT を全部外す（積状態のみ）: 各 qubit が独立になり古典で効率よく計算できる。
     これで精度が落ちなければ「量子らしさ」は効いていない。
   - 回路を恒等写像に置き換える: 前後の古典層だけで出せる精度。

## 設定

- 問題: 2D 熱弾性の弾性段階（DEM）。温度は解析解を与える（熱モデルの誤差を切り離すため）
- シード: 3（既定）
- 予算: 古典 3000 step + L-BFGS 30、量子 1000 step + L-BFGS 20
  → **古典側に有利な設定**。それでも QPINN が勝つなら主張は強くなる（保守的な比較）
- Adam 終了時点と L-BFGS 後の両方を記録する（QPINN では L-BFGS が悪化させることがあるため）
"""

import argparse
import importlib.util
import json
import math
import os
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "dem_thermo", os.path.join(_HERE, "14_qpinn_dem_thermo.py"))
dem = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dem)

torch.set_default_dtype(torch.float64)


class SmallFourier(torch.nn.Module):
    """ランダム Fourier 特徴（固定）+ 小さい MLP。角度エンコード回路と同系統の関数クラス"""

    def __init__(self, d_in, d_out, n_freq=6, hidden=9, seed=0, scale=1.0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("B", torch.randn(d_in, n_freq, generator=g) * scale)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2 * n_freq + d_in, hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, d_out))

    def forward(self, x):
        z = x @ self.B
        return self.net(torch.cat([torch.sin(z), torch.cos(z), x], dim=1))


def model_specs(dim):
    """(名前, コンストラクタ(seed), 量子か)"""
    return [
        ("ハイブリッドQPINN",        lambda s: dem.QPINNHybrid(dim, dim, seed=s, n_qubits=4), True),
        ("CNOTなし(積状態)",          lambda s: dem.QPINNHybrid(dim, dim, seed=s, n_qubits=4,
                                                            entangle=False), True),
        ("回路→恒等写像",            lambda s: dem.QPINNHybrid(dim, dim, seed=s,
                                                            identity_width=4), False),
        ("小MLP",                    lambda s: dem.MLP(dim, dim, hidden=10, depth=2), False),
        ("小Fourier",                lambda s: SmallFourier(dim, dim, seed=s), False),
        ("大MLP(参考)",              lambda s: dem.MLP(dim, dim), False),
    ]


def run(dim=2, seeds=(0, 1, 2), c_steps=3000, c_lbfgs=30, q_steps=1000, q_lbfgs=20,
        only=None, out_json=None):
    print("=" * 78)
    print(f"  QPINN 対照実験（{dim}D 熱弾性・DEM・温度は解析解）")
    print("=" * 78)
    if not dem.verify_exact_solution(dim, verbose=False):
        raise SystemExit("解析解の検証に失敗した")

    results = {}
    for name, ctor, is_q in model_specs(dim):
        if only and name not in only:
            continue
        rows = []
        for s in seeds:
            torch.manual_seed(s)
            m = ctor(s)
            st, lb = (q_steps, q_lbfgs) if is_q else (c_steps, c_lbfgs)
            t0 = time.time()
            r = dem.solve_elastic(m, dim, None, form="dem", steps=st, lbfgs_steps=lb,
                                  use_exact_T=True, log_every=10**9)
            rows.append({"seed": s, "n_param": dem.n_params(m), "u_adam": r["err_adam"],
                         "u": r["err"], "s": r["s_err"], "time": time.time() - t0})
            print(f"  {name:<16} seed={s}  params={rows[-1]['n_param']:<5} "
                  f"u(Adam)={r['err_adam']:.3e}  u(+LBFGS)={r['err']:.3e}  "
                  f"σ={r['s_err']:.3e}  {rows[-1]['time']:.0f}s", flush=True)
        results[name] = rows
        if out_json:
            with open(out_json, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=1)

    print_table(results)
    return results


def _ms(v):
    return f"{np.mean(v):.2e} ± {np.std(v):.1e}"


def print_table(results):
    print("\n" + "=" * 78)
    print("  まとめ（シード平均 ± 標準偏差）")
    print("=" * 78)
    print(f"{'モデル':<18}{'params':>7}  {'u (Adam終了)':>20}  {'u (+L-BFGS)':>20}  {'最良':>9}")
    for name, rows in results.items():
        ua = [r["u_adam"] for r in rows]
        ul = [r["u"] for r in rows]
        best = [min(a, b) for a, b in zip(ua, ul)]
        print(f"{name:<18}{rows[0]['n_param']:>7}  {_ms(ua):>20}  {_ms(ul):>20}  "
              f"{np.mean(best):>9.2e}")
    print("\n「最良」= 各シードで Adam 終了時と L-BFGS 後の良い方（L-BFGS が悪化させる場合があるため）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=2, choices=[2, 3])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--c-steps", type=int, default=3000)
    ap.add_argument("--q-steps", type=int, default=1000)
    ap.add_argument("--c-lbfgs", type=int, default=30)
    ap.add_argument("--q-lbfgs", type=int, default=20)
    ap.add_argument("--only", default=None, help="モデル名をカンマ区切りで（部分実行）")
    ap.add_argument("--json", default=None, help="途中経過を書き出す JSON パス")
    args = ap.parse_args()
    run(dim=args.dim, seeds=tuple(range(args.seeds)), c_steps=args.c_steps,
        q_steps=args.q_steps, c_lbfgs=args.c_lbfgs, q_lbfgs=args.q_lbfgs, only=args.only.split(",") if args.only else None,
        out_json=args.json)
