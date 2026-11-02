# -*- coding: utf-8 -*-
"""
45 - fifa の Abaqus の結果（results/abaqus）を自作の参照解と突き合わせる

(a) 軸対称（clamped / free）: 42 の自作軸対称 FE と比較（変位と von Mises，中央面付近と表面付近）
(b) 3D 単位セル（clamped）  : 中央面 z ≈ 0 を 34 の平面ひずみ FE（L = 6）と比較．44 の --ref plane の近似が妥当かを見る
単位: Abaqus は µm, MPa．ここではビア半径 a = 2.5 µm を長さの単位にし，応力は MPa のまま比べる（自作側を 100 倍）
"""
import gzip, importlib.util, os
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

_h = os.path.dirname(os.path.abspath(__file__))


def _load(name, fn):
    s = importlib.util.spec_from_file_location(name, os.path.join(_h, fn))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


def csv(path):
    with gzip.open(path, "rt") as f:
        head = f.readline().strip().split(",")
        d = np.loadtxt(f, delimiter=",")
    return {h: d[:, i] for i, h in enumerate(head)}


def interp(P, V, Q):
    """線形補間（凸包の外は最近傍）"""
    out = LinearNDInterpolator(P, V)(Q)
    bad = np.isnan(out).any(axis=1) if out.ndim > 1 else np.isnan(out)
    if bad.any():
        out[bad] = NearestNDInterpolator(P, V)(Q[bad])
    return out


def nmae(a, b):
    return float(np.mean(np.abs(a - b)) / np.abs(b).max())


AX = _load("ax", "42_tsv_axisym.py")
HH = AX.H_HALF
print("(a) 軸対称: Abaqus（CAX8, 要素 84,500）vs 自作 FE（P1, h = 0.025）")
for bc in ("clamped", "free"):
    base = os.path.join(_h, "results", "abaqus", f"tsv_axisym_{bc}")
    s, u = csv(base + "_stress.csv.gz"), csv(base + "_disp.csv.gz")
    fe = AX.fe_solve(bc, 0.025)
    # 変位（節点）: Abaqus 節点で自作 FE を補間
    Qn = np.c_[u["x"], u["y"]] / 2.5
    Ua = np.c_[u["U1"], u["U2"]] / 2.5e-3
    Uf = interp(fe["pts"], fe["u"], Qn)
    e_u = np.linalg.norm(Uf - Ua) / np.linalg.norm(Ua)
    # von Mises（要素重心）: 相ごとに補間して界面をまたがないようにする
    Qc = np.c_[s["x"], s["y"]] / 2.5
    vm_a = s["mises"]
    vm_f = np.zeros(len(Qc))
    ph_a = (Qc[:, 0] > AX.A_R).astype(int)
    vm_fe = AX.von_mises(fe["S"]) * 100.0
    for ph in (0, 1):
        mk, mf = ph_a == ph, fe["phase"] == ph
        vm_f[mk] = interp(fe["cen"][mf], vm_fe[mf], Qc[mk])
    z = Qc[:, 1]
    line = f"  {bc:8s}: 変位 {e_u:.1e}"
    for key, band in (("中央面付近", z < 0.5), ("表面付近", (z > HH - 1.0) & (z < HH - 0.25))):
        line += f"，von Mises（{key}）{nmae(vm_f[band], vm_a[band]):.1e}"
    line += f"，最大 von Mises Abaqus {vm_a.max():.0f} MPa / 自作 {vm_fe.max():.0f} MPa"
    print(line, flush=True)

print("\n(b) 3D 単位セル（clamped）の中央面 vs 平面ひずみ FE（L = 6，h = 0.01）")
T3 = _load("t3", "44_tsv_3d.py")
ref2 = T3.ref_plane(0.01)
base = os.path.join(_h, "results", "abaqus", "tsv_3d_clamped")
s, u = csv(base + "_stress.csv.gz"), csv(base + "_disp.csv.gz")
z = s["z"] / 2.5
for zmax in (0.5, 2.0, 5.0):
    band = z < zmax
    Q = np.c_[s["x"][band], s["y"][band]] / 2.5
    vm2 = T3.von_mises(ref2["S"]) * 100.0
    ph = (np.hypot(Q[:, 0], Q[:, 1]) >= 1.0).astype(int)
    v = np.zeros(len(Q))
    ph2 = (np.hypot(ref2["cen"][:, 0], ref2["cen"][:, 1]) >= 1.0).astype(int)
    for p in (0, 1):
        v[ph == p] = interp(ref2["cen"][ph2 == p, :2], vm2[ph2 == p], Q[ph == p])
    print(f"  z < {zmax:<3}（{band.sum()} 要素）: von Mises の規格化 MAE {nmae(v, s['mises'][band]):.1e}", flush=True)
zn = u["z"] / 2.5; nb = zn < 0.05
Qn = np.c_[u["x"][nb], u["y"][nb]] / 2.5
Ua = np.c_[u["U1"][nb], u["U2"][nb]] / 2.5e-3
U2 = interp(ref2["nodes"][:, :2], ref2["U"][:, :2], Qn)
print(f"  中央面の節点（{nb.sum()} 個）: 面内変位の相対誤差 {np.linalg.norm(U2 - Ua) / np.linalg.norm(Ua):.1e}")
print(f"  最大 von Mises: 3D {s['mises'].max():.0f} MPa（z = {s['z'][s['mises'].argmax()]:.2f} µm）")
