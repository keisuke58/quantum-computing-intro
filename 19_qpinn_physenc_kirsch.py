# -*- coding: utf-8 -*-
"""
19 - 物理を考慮したエンコーディングの QPINN（Kirsch，円孔付き板の応力集中）

13_qpinn_kirsch.py では，(x, y) をそのまま回転角に入れる素の QPINN は表現力が足りず
（8 qubit でも rel L2 ≈ 2e-1 で頭打ち），PDE 学習でも 7.2e-1 にとどまった。

量子回路の出力は入力角度のフーリエ級数（Schuld et al. 2021）。一方 Kirsch 解は極座標で

    u_r = r [ α₀ + α₁ρ² + (β₀ + β₁ρ² + β₂ρ⁴) cos 2θ ]
    u_θ = r (γ₀ + γ₁ρ² + γ₂ρ⁴) sin 2θ,            ρ = a/r

の形をしている（verify_polar_form で数値的に確かめる）。そこで回路はそのままに，座標の入れ方だけを変える:

  - θ を回転角 RZ に入れる       → cos 2θ, sin 2θ が低次のフーリエ成分になる
  - ρ を Chebyshev エンコード     → RY(arccos ξ) で出力が ξ の多項式になる
  - 出力に r を掛け，(u_r, u_θ) をデカルト成分に戻す

段階ごとの効果と，同じ座標を与えた古典 MLP（公平な対照）を比べる:

  Q-xy      : 素の QPINN（13 と同じ）
  Q-polar   : (θ, r) を角度に入れる
  Q-cheb    : (θ, ρ の Chebyshev)
  Q-phys    : Q-cheb + 出力に r を掛けて極座標成分として読む
  M-phys    : Q-phys と同じ前処理・後処理で，回路の代わりに小さな MLP（パラメータ数を揃える）
  M-xy      : (x, y) 入力の小さな MLP

使い方:
  python 19_qpinn_physenc_kirsch.py --mode sup          # 表現力（教師ありフィット）
  python 19_qpinn_physenc_kirsch.py --mode pde          # PDE 学習（物理のみ）
"""

import argparse
import importlib.util
import json
import math
import os
import time

import numpy as np
import torch
import pennylane as qml

_spec = importlib.util.spec_from_file_location(
    "kirsch", os.path.join(os.path.dirname(os.path.abspath(__file__)), "13_qpinn_kirsch.py"))
K = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(K)

torch.set_default_dtype(torch.float64)

A, R = K.A_HOLE, K.R_OUT
RHO_MIN = A / R
XI_MAX = 0.95          # arccos の端点（ξ=±1）で微分が発散するのを避ける


# ==============================================================
# 1. 解析解が仮定した極座標の形に入っていることの確認
# ==============================================================
def verify_polar_form(n=4000, seed=0):
    """u_r / r, u_θ / r を基底 {1, ρ², ρ⁴} × {1, cos2θ} / {sin2θ} で最小二乗フィットし残差を見る"""
    rng = np.random.default_rng(seed)
    r = np.sqrt(A**2 + rng.random(n) * (R**2 - A**2))
    th = rng.random(n) * 2 * np.pi
    u, v = K.uv_exact(r * np.cos(th), r * np.sin(th))
    ur = u * np.cos(th) + v * np.sin(th)
    ut = -u * np.sin(th) + v * np.cos(th)
    rho = A / r
    P = np.stack([np.ones_like(rho), rho**2, rho**4], 1)
    Br = np.concatenate([P, P * np.cos(2 * th)[:, None]], 1)
    Bt = P * np.sin(2 * th)[:, None]
    res = []
    for B, y in ((Br, ur / r), (Bt, ut / r)):
        c, *_ = np.linalg.lstsq(B, y, rcond=None)
        res.append(np.linalg.norm(B @ c - y) / np.linalg.norm(y))
    print(f"極座標の形の確認: u_r/r 残差 {res[0]:.1e}, u_θ/r 残差 {res[1]:.1e}")
    return max(res) < 1e-10


# ==============================================================
# 2. 座標の前処理
# ==============================================================
def polar(xy):
    r = torch.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2)
    th = torch.atan2(xy[:, 1], xy[:, 0])
    return r, th


def cheb_angle(r):
    """ρ = a/r ∈ [a/R, 1] → ξ ∈ [-XI_MAX, XI_MAX] → arccos ξ"""
    rho = A / r
    xi = XI_MAX * (2 * (rho - RHO_MIN) / (1 - RHO_MIN) - 1)
    return torch.arccos(xi)


def to_cart(ur, ut, th):
    c, s = torch.cos(th), torch.sin(th)
    return torch.stack([ur * c - ut * s, ur * s + ut * c], dim=1)


# ==============================================================
# 3. モデル
# ==============================================================
N_QUBITS, N_LAYERS, N_ENCODE = 4, 3, 2


class QPhys(K.QPINN):
    """13 の QPINN と同じ回路（RY に第 1 入力，RZ に第 2 入力）。入力と出力の扱いだけ変える"""

    def __init__(self, mode, seed=0):
        super().__init__(seed=seed, n_qubits=N_QUBITS, n_layers=N_LAYERS, n_encode=N_ENCODE)
        self.mode = mode

    def forward(self, xy):
        if self.mode == "xy":
            return super().forward(xy)
        r, th = polar(xy)
        if self.mode == "polar":
            a1 = math.pi * (r - A) / (R - A)
        else:
            a1 = cheb_angle(r)
        out = self.circuit(a1, th, self.theta)
        out = torch.stack([out[0].reshape(-1), out[1].reshape(-1)], dim=1)
        out = self.scale * out + self.shift
        if self.mode == "phys":
            return to_cart(r * out[:, 0], r * out[:, 1], th)
        return out


class MLPSmall(torch.nn.Module):
    """パラメータ数を QPINN（52）に揃えた小さな MLP。mode="phys" なら Q-phys と同じ前処理・後処理"""

    def __init__(self, mode, hidden=5, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.mode = mode
        d_in = 2 if mode == "xy" else 3          # phys: (arccos ξ, cos θ, sin θ)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, 2))

    def forward(self, xy):
        if self.mode == "xy":
            return self.net(xy / R)
        r, th = polar(xy)
        f = torch.stack([cheb_angle(r), torch.cos(th), torch.sin(th)], dim=1)
        out = self.net(f)
        return to_cart(r * out[:, 0], r * out[:, 1], th)


def build(name, seed):
    return {
        "Q-xy": lambda: QPhys("xy", seed),
        "Q-polar": lambda: QPhys("polar", seed),
        "Q-cheb": lambda: QPhys("cheb", seed),
        "Q-phys": lambda: QPhys("phys", seed),
        "M-phys": lambda: MLPSmall("phys", seed=seed),
        "M-xy": lambda: MLPSmall("xy", seed=seed),
    }[name]()


# ==============================================================
# 4. 実行
# ==============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sup", "pde"], default="sup")
    ap.add_argument("--models", default="Q-xy,Q-polar,Q-cheb,Q-phys,M-phys,M-xy")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lbfgs", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not (K.verify_exact_solution(verbose=False) and verify_polar_form()):
        raise SystemExit("解析解の検証に失敗した")

    rows = []
    for name in args.models.split(","):
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            m = build(name, seed)
            t0 = time.time()
            if args.mode == "sup":
                u_err, kt = K.supervised_fit(m, steps=args.steps)
                row = {"name": name, "seed": seed, "n_param": K.n_params(m),
                       "u_err": u_err, "kt": kt, "s_err": K.rel_l2_stress(m)}
            else:
                lr = 0.01 if name.startswith("M") else 0.05
                res = K.train(name, m, steps=args.steps, lr=lr, lbfgs_steps=args.lbfgs,
                              log_every=max(args.steps // 4, 1))
                row = {k: res[k] for k in ("name", "n_param", "u_err", "s_err", "kt")}
                row["seed"] = seed
            row["time"] = time.time() - t0
            rows.append(row)
            print(f"RESULT {name:8s} seed={seed} params={row['n_param']:<3} "
                  f"u={row['u_err']:.3e} σ={row['s_err']:.3e} K_t={row['kt']:.3f} "
                  f"{row['time']:.0f}s", flush=True)
            if args.out:
                with open(args.out, "w") as f:
                    json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
