# -*- coding: utf-8 -*-
"""
12 - QPINN アブレーション: 表現力 vs 最適化可能性

`11_qpinn_lame.py` の厚肉円筒（Lamé）ベンチを使い、回路構成を振って
QPINN の性能がどこで決まるかを切り分ける。

計画レビュー（docs/qpinn_plan_review.md §5）で優先度2に挙げた項目。
#3 星形グレインで詰まった時に原因を分離できるよう、#1 の段階で潰しておく。

## 測るもの

各構成について2つの誤差を出す:

1. **表現力の上限（supervised）**: 解析解 u(r) を直接フィットした時の相対 L2。
   PDE 損失を使わないので、純粋に「その回路がこの関数を表現できるか」だけを見る。
2. **PINN としての性能（PDE）**: PDE 残差 + 境界条件で学習した時の相対 L2。

この2つの差が「表現できるのに PDE 損失では学習できない」量、つまり
**最適化可能性の損失**。QPINN が固体力学で苦戦する理由がここにあるなら、
#3 の応力集中でも同じ切り分けが効く。

## 振るもの

- エンコード周波数の付け方（uniform / ladder）← 最重要
- qubit 数
- 変分層の深さ
- データ再アップロード回数
"""

import argparse
import importlib.util
import math
import os
import time

import numpy as np
import torch

# 11_qpinn_lame.py はモジュール名として import できないので importlib で読む
_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "qpinn_lame", os.path.join(_HERE, "11_qpinn_lame.py"))
lame = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lame)


# ==============================================================
# 1. 表現力の上限を測る（PDE を使わない教師ありフィット）
# ==============================================================
def supervised_fit(model, steps=1500, lr=0.05, n_points=64):
    """解析解を直接フィットして rel L2 を返す。回路の表現力の上限。

    PDE 損失と違って微分を取らないので、同じステップ数でも大幅に速い。
    """
    rs = torch.linspace(lame.A_IN, lame.B_OUT, n_points)
    target = torch.tensor(lame.u_exact(rs.numpy()))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.01)
    for _ in range(steps):
        opt.zero_grad()
        loss = ((model(rs) - target) ** 2).mean()
        loss.backward()
        opt.step()
        sched.step()
    return lame.rel_l2_u(model)


# ==============================================================
# 2. 1構成の評価
# ==============================================================
def eval_config(n_qubits, n_layers, n_encode, freq_mode, seeds=(0, 1),
                pde_steps=3000, sup_steps=1500, lr=0.02, verbose=True):
    sup_errs, pde_errs, sig_errs, times = [], [], [], []
    n_param = None

    for seed in seeds:
        torch.manual_seed(seed)
        m_sup = lame.QPINN(seed=seed, n_qubits=n_qubits, n_layers=n_layers,
                           n_encode=n_encode, freq_mode=freq_mode)
        sup_errs.append(supervised_fit(m_sup, steps=sup_steps))

        torch.manual_seed(seed)
        m_pde = lame.QPINN(seed=seed, n_qubits=n_qubits, n_layers=n_layers,
                           n_encode=n_encode, freq_mode=freq_mode)
        n_param = sum(p.numel() for p in m_pde.parameters())
        t0 = time.time()
        r = lame.train(f"q={n_qubits} l={n_layers} e={n_encode} {freq_mode} seed={seed}",
                       m_pde, steps=pde_steps, lr=lr, log_every=10 ** 9)
        times.append(time.time() - t0)
        pde_errs.append(r["u_err"])
        sig_errs.append(r["s_err"])

    out = {
        "n_qubits": n_qubits, "n_layers": n_layers, "n_encode": n_encode,
        "freq_mode": freq_mode, "n_param": n_param,
        "sup": float(np.mean(sup_errs)), "sup_sd": float(np.std(sup_errs)),
        "pde": float(np.mean(pde_errs)), "pde_sd": float(np.std(pde_errs)),
        "sig": float(np.mean(sig_errs)),
        "time": float(np.mean(times)),
    }
    if verbose:
        print(f"  ★ q={n_qubits} l={n_layers} e={n_encode} {freq_mode:<7} "
              f"params={n_param:<3} 表現力={out['sup']:.2e} PDE={out['pde']:.2e} "
              f"σ_r={out['sig']:.2e} ({out['time']:.0f}s/seed)", flush=True)
    return out


# ==============================================================
# 3. スイープの定義
# ==============================================================
def sweep_configs(quick=False):
    """(n_qubits, n_layers, n_encode, freq_mode) のリスト。

    基準構成は 11_qpinn_lame.py と同じ (4, 3, 2, uniform)。
    各軸を1本ずつ振る（全組み合わせはシミュレータのコスト的に無理）。
    """
    base = (4, 3, 2, "uniform")
    configs = [base]

    # A. エンコード周波数（最重要）
    configs.append((4, 3, 2, "ladder"))

    # B. qubit 数
    for q in ([2] if quick else [2, 6]):
        configs.append((q, 3, 2, "uniform"))

    # C. 変分層の深さ
    for l in ([1] if quick else [1, 2, 5]):
        configs.append((4, l, 2, "uniform"))

    # D. 再アップロード回数（フーリエ周波数の上限）
    for e in ([1] if quick else [1, 3]):
        configs.append((4, 3, e, "uniform"))

    # 重複除去（順序は保つ）
    seen, uniq = set(), []
    for c in configs:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


# ==============================================================
# 4. 出力
# ==============================================================
def print_table(results):
    print("\n" + "=" * 86)
    print("  アブレーション結果（2 シード平均）")
    print("=" * 86)
    print(f"{'qubit':>6}{'層':>4}{'再UP':>6}{'エンコード':>12}{'params':>8}"
          f"{'表現力':>12}{'PDE':>12}{'σ_r':>12}{'秒':>7}")
    for r in results:
        print(f"{r['n_qubits']:>6}{r['n_layers']:>4}{r['n_encode']:>6}"
              f"{r['freq_mode']:>12}{r['n_param']:>8}"
              f"{r['sup']:>12.2e}{r['pde']:>12.2e}{r['sig']:>12.2e}{r['time']:>7.0f}")

    print("\n「表現力」= 解析解を直接フィットした時の rel L2（PDE 損失を使わない上限）")
    print("「PDE」   = PDE 残差 + 境界条件で学習した時の rel L2")
    print("両者の比が大きいほど『表現できるのに PDE 損失では学習できない』")


def write_markdown(results, path):
    lines = [
        "# QPINN アブレーション結果（ベンチ #1: 厚肉円筒）",
        "",
        "生成元: `12_qpinn_ablation.py`（2 シード平均）。",
        "基準構成は 4 qubit / 3 層 / 再アップロード 2 回 / uniform エンコード。",
        "collocation 64 点、Adam + cosine annealing。PDE 学習は 2000 step（1200 step では未収束で、",
        "構成の差ではなく収束不足を測ってしまうことを実測で確認した）。",
        "",
        "- **表現力**: 解析解 u(r) を直接フィットした時の相対 L2。PDE 損失を使わないので、",
        "  その回路がこの関数を表現できるかだけを見る（＝達成可能な下限）。",
        "- **PDE**: PDE 残差 + 境界条件で学習した時の相対 L2。",
        "- **比 PDE/表現力**: 大きいほど「表現できるのに学習できない」。最適化可能性の損失。",
        "",
        "| qubit | 層 | 再UP | エンコード | params | 表現力 | PDE | 比 | σ_r | 秒/seed |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        ratio = r["pde"] / r["sup"] if r["sup"] > 0 else float("nan")
        lines.append(
            f"| {r['n_qubits']} | {r['n_layers']} | {r['n_encode']} | {r['freq_mode']} "
            f"| {r['n_param']} | {r['sup']:.2e} | {r['pde']:.2e} | {ratio:.0f}× "
            f"| {r['sig']:.2e} | {r['time']:.0f} |")
    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n→ {path} に書き出した")


# ==============================================================
# 5. 実行
# ==============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="構成を減らした短縮版")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--pde-steps", type=int, default=3000,
                    help="PDE 学習のステップ数。1200 程度だと未収束で、構成の差ではなく "
                         "収束不足を測ってしまう（実測で確認済み）")
    ap.add_argument("--colloc", type=int, default=64,
                    help="collocation 点数。既定 64 で 11_qpinn_lame.py と揃えてある。"
                         "（コストはバッチ化されていて点数にほぼ依存しない）")
    ap.add_argument("--sup-steps", type=int, default=1500)
    ap.add_argument("--out", default="docs/qpinn_ablation_results.md")
    args = ap.parse_args()

    lame.N_COLLOC = args.colloc
    configs = sweep_configs(quick=args.quick)
    print("=" * 86)
    print(f"  QPINN アブレーション: {len(configs)} 構成 × {args.seeds} シード")
    print(f"  PDE {args.pde_steps} step / 教師あり {args.sup_steps} step / "
          f"collocation {args.colloc} 点")
    print("=" * 86)
    print("表現力（教師ありフィット）と PDE 学習を別々に測り、差を見る")

    results = []
    t0 = time.time()
    for cfg in configs:
        results.append(eval_config(*cfg, seeds=tuple(range(args.seeds)),
                                   pde_steps=args.pde_steps, sup_steps=args.sup_steps))
    print(f"\n総時間: {time.time() - t0:.0f}s")

    print_table(results)
    write_markdown(results, os.path.join(_HERE, args.out))
    return results


if __name__ == "__main__":
    main()
