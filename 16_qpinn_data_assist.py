# -*- coding: utf-8 -*-
"""
16 - データ同化した QPINN: 参照データを加えると QPINN の弱点はどこまで埋まるか

これまでのベンチ（11 Lamé, 13 Kirsch）は「方程式だけから解く」設定だった。
OIST の Rakala 実装（2D Navier-Stokes 円柱後流）は参照解のデータ 5,000 点を損失に加えており、
条件が大きく違う。同じ土俵で比べるため、固体の問題にも変位データを加えて比較する。

    損失 = 物理の損失（PDE 残差 + 境界条件） + w_data × mean |u_pred − u_data|²

データは変位（Kirsch では (u, v)）。DIC（画像相関法）で全視野の変位を測る状況を想定。
データ点数 0 は従来の「物理のみ」と同じ。

比較するモデル:
  - 大 MLP（参考）
  - ハイブリッド / 角度エンコード QPINN
  - 同じパラメータ数の小 MLP（15_qpinn_controls.py で QPINN と同等以上だった対照）
"""

import argparse
import importlib.util
import json
import os
import time

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

_HERE = os.path.dirname(os.path.abspath(__file__))


_CACHE = {}


def _load(name, fname):
    if name in _CACHE:
        return _CACHE[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    _CACHE[name] = m
    return m


W_DATA = 10.0   # データ損失の重み（境界条件の重みと同程度）
NOISE = 0.0     # データに加えるノイズ（データの RMS に対する相対値．0.01 = 1%）


def _add_noise(d, seed):
    """データの RMS に比例したガウスノイズを加える（評価は常にノイズなしの解析解と比較）"""
    if NOISE <= 0:
        return d
    g = torch.Generator().manual_seed(5000 + seed)
    rms = torch.sqrt((d ** 2).mean())
    return d + NOISE * rms * torch.randn(d.shape, generator=g)


# ==============================================================
# Lamé（1D 軸対称）
# ==============================================================
def run_lame(model, n_data, seed, steps, lr, lbfgs):
    L = _load("lame", "11_qpinn_lame.py")
    g = torch.Generator().manual_seed(1000 + seed)
    r_c = torch.linspace(L.A_IN, L.B_OUT, L.N_COLLOC)
    r_b = torch.tensor([L.A_IN, L.B_OUT])
    if n_data > 0:
        r_d = L.A_IN + (L.B_OUT - L.A_IN) * torch.rand(n_data, generator=g)
        u_d = _add_noise(torch.tensor(L.u_exact(r_d.numpy())), seed)

    def loss_fn():
        loss = L.pinn_loss(model, r_c, r_b)[0]
        if n_data > 0:
            loss = loss + W_DATA * ((model(r_d) - u_d) ** 2).mean()
        return loss

    t = _train(model, loss_fn, steps, lr, lbfgs)
    return {"u": L.rel_l2_u(model), "s": L.rel_l2_sigma(model), "kt": None, "time": t,
            "n_param": L.n_params(model)}


def lame_models(seed):
    L = _load("lame", "11_qpinn_lame.py")
    return {
        "大MLP": (lambda: L.MLPPINN(hidden=16, depth=2), False),     # 321
        "QPINN": (lambda: L.QPINN(seed=seed), True),                 # 50
        "小MLP": (lambda: L.MLPPINN(hidden=5, depth=2), False),      # 46
    }


# ==============================================================
# Kirsch（2D 応力集中）
# ==============================================================
def run_kirsch(model, n_data, seed, steps, lr, lbfgs):
    K = _load("kirsch", "13_qpinn_kirsch.py")
    xy, xy_in, xy_out, th_b = K.sample_points(512, 64)
    ue, ve = K.uv_exact(xy_out[:, 0].numpy(), xy_out[:, 1].numpy())
    uv_out = torch.tensor(np.stack([ue, ve], axis=1))
    if n_data > 0:
        g = torch.Generator().manual_seed(2000 + seed)
        t = torch.rand(n_data, generator=g)
        r = torch.sqrt(K.A_HOLE**2 + t * (K.R_OUT**2 - K.A_HOLE**2))
        th = torch.rand(n_data, generator=g) * 2 * np.pi
        xy_d = torch.stack([r * torch.cos(th), r * torch.sin(th)], dim=1)
        ud, vd = K.uv_exact(xy_d[:, 0].numpy(), xy_d[:, 1].numpy())
        uv_d = _add_noise(torch.tensor(np.stack([ud, vd], axis=1)), seed)

    def loss_fn():
        loss = K.pinn_loss(model, xy, xy_in, xy_out, uv_out, th_b)[0]
        if n_data > 0:
            loss = loss + W_DATA * ((model(xy_d) - uv_d) ** 2).mean()
        return loss

    t = _train(model, loss_fn, steps, lr, lbfgs)
    return {"u": K.rel_l2_uv(model), "s": K.rel_l2_stress(model), "kt": K.kt_predicted(model),
            "time": t, "n_param": K.n_params(model)}


def kirsch_models(seed):
    K = _load("kirsch", "13_qpinn_kirsch.py")
    return {
        "大MLP": (lambda: K.MLPPINN(), False),                                   # 2274
        "QPINN": (lambda: K.QPINNHybrid(seed=seed, n_qubits=4, hidden=8), True),  # 166
        "小MLP": (lambda: K.MLPPINN(hidden=10, depth=2), False),                 # 162
    }


# ==============================================================
# 学習ループ（Adam → 古典のみ L-BFGS）
# ==============================================================
def _train(model, loss_fn, steps, lr, lbfgs):
    t0 = time.time()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.01)
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
        sched.step()
    if lbfgs > 0:
        lopt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20, history_size=50,
                                 line_search_fn="strong_wolfe")

        def closure():
            lopt.zero_grad()
            l = loss_fn()
            l.backward()
            return l
        for _ in range(lbfgs):
            if not torch.isfinite(lopt.step(closure)):
                break
    return time.time() - t0


# ==============================================================
# 実行
# ==============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=["lame", "kirsch"], required=True)
    ap.add_argument("--ndata", default=None, help="データ点数をカンマ区切りで")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--models", default=None, help="大MLP,QPINN,小MLP から選択")
    ap.add_argument("--c-steps", type=int, default=3000)
    ap.add_argument("--q-steps", type=int, default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--noise", type=float, default=0.0, help="データのノイズ（RMS 比，0.05 = 5%%）")
    a = ap.parse_args()
    global NOISE
    NOISE = a.noise

    if a.problem == "lame":
        run, models_fn = run_lame, lame_models
        ndata = [int(s) for s in (a.ndata or "0,4,16").split(",")]
        q_steps, q_lr = a.q_steps or 2000, 0.02
    else:
        run, models_fn = run_kirsch, kirsch_models
        ndata = [int(s) for s in (a.ndata or "0,50,500").split(",")]
        q_steps, q_lr = a.q_steps or 1500, 0.01
    wanted = a.models.split(",") if a.models else None

    rows = []
    for name in (wanted or list(models_fn(0).keys())):
        for n in ndata:
            for seed in range(a.seeds):
                torch.manual_seed(seed)
                ctor, is_q = models_fn(seed)[name]
                m = ctor()
                # 古典は Adam + L-BFGS（L-BFGS で 3 倍改善した）、
                # QPINN は Adam のみ（L-BFGS は 1.1 倍しか効かず時間が 2 倍になった）
                r = run(m, n, seed, q_steps if is_q else a.c_steps,
                        q_lr if is_q else 0.01, 0 if is_q else 30)
                r.update({"problem": a.problem, "model": name, "ndata": n, "seed": seed, "noise": NOISE})
                rows.append(r)
                kt = f"  K_t={r['kt']:.3f}" if r["kt"] is not None else ""
                print(f"{a.problem} noise={NOISE:.0%} {name:<6} n_data={n:<4} seed={seed}  params={r['n_param']:<5}"
                      f" u={r['u']:.3e}  σ={r['s']:.3e}{kt}  {r['time']:.0f}s", flush=True)
                if a.json:
                    json.dump(rows, open(a.json, "w"), ensure_ascii=False, indent=1)

    print("\n=== まとめ（シード平均） ===")
    for name in dict.fromkeys(r["model"] for r in rows):
        for n in ndata:
            sel = [r for r in rows if r["model"] == name and r["ndata"] == n]
            if not sel:
                continue
            us = [r["u"] for r in sel]
            ss = [r["s"] for r in sel]
            kt = ""
            if sel[0]["kt"] is not None:
                kt = f"  K_t={np.mean([r['kt'] for r in sel]):.3f}"
            print(f"{name:<6} n_data={n:<4} params={sel[0]['n_param']:<5} "
                  f"u={np.mean(us):.3e}±{np.std(us):.1e}  σ={np.mean(ss):.3e}{kt}")


if __name__ == "__main__":
    main()
