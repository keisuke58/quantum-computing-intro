# -*- coding: utf-8 -*-
"""
43 - 42 の軸対称 FE の検証: 上下固定のとき，高さの中央面は平面ひずみに近いので，
同じ円柱セル（Cu の芯 r < a，Si の殻 a < r < R，外周 u_r = 0）の平面ひずみの解析解（Lamé 型）と比べる．

解析解（ε_zz = 0）:
  芯: u = A r，殻: u = B r + C / r（C = −B R²）
  r = a で u と σ_rr が連続．σ_rr = (λ+2μ) u' + λ u/r − βΔT，σ_θθ = λ u' + (λ+2μ) u/r − βΔT，σ_zz = λ (u' + u/r) − βΔT
"""
import importlib.util, os
import numpy as np

_h = os.path.dirname(os.path.abspath(__file__))
s = importlib.util.spec_from_file_location("ax", os.path.join(_h, "42_tsv_axisym.py"))
AX = importlib.util.module_from_spec(s); s.loader.exec_module(AX)
(l1, m1, b1), (l2, m2, b2) = AX.MAT[0], AX.MAT[1]
a, R, dT = AX.A_R, AX.R_CELL, AX.DT

# 未知数 A, B（C = −B R²）
#   u の連続:  A a = B a − B R² / a
#   σ_rr の連続: 2(l1+m1) A − b1 dT = 2(l2+m2) B + 2 m2 B R² / a² − b2 dT
M = np.array([[a, -(a - R**2 / a)], [2 * (l1 + m1), -(2 * (l2 + m2) + 2 * m2 * R**2 / a**2)]])
A, B = np.linalg.solve(M, [0.0, (b1 - b2) * dT]); Cc = -B * R**2


def exact(r):
    core = r < a
    u = np.where(core, A * r, B * r + Cc / r)
    up = np.where(core, A, B - Cc / r**2); uor = np.where(core, A, B + Cc / r**2)
    lam = np.where(core, l1, l2); mu = np.where(core, m1, m2); bet = np.where(core, b1, b2)
    srr = (lam + 2 * mu) * up + lam * uor - bet * dT
    stt = lam * up + (lam + 2 * mu) * uor - bet * dT
    szz = lam * (up + uor) - bet * dT
    return u, np.stack([srr, stt, szz], 1)


print(f"解析解: A = {A:.4e}, B = {B:.4e}, C = {Cc:.4e}")
for h in (0.1, 0.05, 0.025):
    ref = AX.fe_solve("clamped", h)
    # 中央面の節点（z = 0）での u_r
    P = ref["pts"]; on = (P[:, 1] < 1e-9) & (P[:, 0] > 0)
    ue, _ = exact(P[on, 0])
    e_u = np.linalg.norm(ref["u"][on, 0] - ue) / np.linalg.norm(ue)
    # 中央面に接する要素の重心での応力（界面をまたぐ要素はない）
    cz = ref["cen"][:, 1]; band = cz < h / 2
    _, Se = exact(ref["cen"][band, 0])
    Sf = ref["S"][band][:, :3]
    e_s = np.linalg.norm(Sf - Se) / np.linalg.norm(Se)
    vm = lambda S: np.sqrt(0.5 * ((S[:, 0] - S[:, 1])**2 + (S[:, 1] - S[:, 2])**2 + (S[:, 2] - S[:, 0])**2))
    mae = np.mean(np.abs(vm(Sf) - vm(Se))) / vm(Se).max()
    print(f"h = {h:<5}: 中央面 u_r の相対誤差 {e_u:.2e}，応力の相対誤差 {e_s:.2e}，von Mises の規格化 MAE {mae:.2e}",
          flush=True)
