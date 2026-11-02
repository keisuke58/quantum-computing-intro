# -*- coding: utf-8 -*-
"""
42 - TSV 1 本の熱応力を軸対称 (r, z) で解く: 量子リザバー特徴量 vs 古典特徴量

寸法は MORE-Stress（Zhu et al., DATE 2025, arXiv:2411.12690）に合わせる（ビア半径 a = 2.5 µm を長さの単位にする）:
  直径 5 µm → a = 1，高さ 50 µm → h = 20，ピッチ 15 µm の正方配列 → 同じ面積の円柱セル R = 15/(2.5·√π) ≈ 3.385
  ΔT = −250 K．ライナー（0.5 µm の誘電体）は省略．材料は 30 と同じ Cu / Si．

高さの中央 z = 0 で上下対称とし，z ∈ [0, h/2] を解く．境界条件:
  r = 0（軸）     : u_r = 0（Cu の u_r を r·φ と置いて自動的に満たす），∂w/∂r = 0
  r = R（セル外周）: u_r = 0，σ_rz = 0（周期配列の近似）
  z = 0（中央面）  : w = 0，σ_rz = 0
  z = h/2（表面）  : "clamped" = u_r = w = 0（MORE-Stress のシナリオ 1），"free" = σ_zz = σ_rz = 0（ウェハ表面）
  r = a（界面）    : 変位の連続，表面力（σ_rr, σ_rz）の連続．熱膨張差による跳びを右辺に入れる

参照解: 軸対称 P1 FE（界面 r = a に節点をそろえた構造格子，h と 2h で収束確認）．
評価: 変位の相対 L2 誤差と，von Mises 応力の規格化平均絶対誤差（MORE-Stress と同じ：最大値で割った MAE）
  "mid"  = 中央面付近（z < 0.5），"top" = 表面付近（h/2 − 1 < z < h/2 − 0.25．free のとき角の特異点は避ける）
"""
import argparse, importlib.util, json, os, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import jax, jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
_h = os.path.dirname(os.path.abspath(__file__))


def _load(name, fn):
    s = importlib.util.spec_from_file_location(name, os.path.join(_h, fn))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


T = _load("t30", "30_qerc_tsv_single.py")
C = _load("t31", "31_tsv_classical_controls.py")
DT = T.DT
MAT = {0: (T.LAM1, T.MU1, T.BET1), 1: (T.LAM2, T.MU2, T.BET2)}     # 0: Cu, 1: Si
A_R = 1.0
H_HALF = 10.0                          # h/2（h = 50 µm）
R_CELL = 15.0 / (2.5 * np.sqrt(np.pi))  # ピッチ 15 µm と同じ面積の円柱セル


# ==============================================================
# 1. 参照解: 軸対称 P1 FE
# ==============================================================
def mesh(h):
    nr1 = int(round(A_R / h)); nr2 = int(round((R_CELL - A_R) / h)); nz = int(round(H_HALF / h))
    r = np.concatenate([np.linspace(0, A_R, nr1 + 1), np.linspace(A_R, R_CELL, nr2 + 1)[1:]])
    z = np.linspace(0, H_HALF, nz + 1)
    Rg, Zg = np.meshgrid(r, z, indexing="ij")
    pts = np.stack([Rg.ravel(), Zg.ravel()], 1)
    nrp, nzp = len(r), len(z)
    idx = lambda i, j: i * nzp + j
    tri = []
    for i in range(nrp - 1):
        for j in range(nzp - 1):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            tri += [(a, b, c), (a, c, d)]
    tri = np.array(tri)
    cen = pts[tri].mean(1)
    phase = (cen[:, 0] > A_R).astype(int)
    return pts, tri, phase


def Dmat(lam, mu):
    return np.array([[lam + 2 * mu, lam, lam, 0], [lam, lam + 2 * mu, lam, 0],
                     [lam, lam, lam + 2 * mu, 0], [0, 0, 0, mu]])


def fe_solve(bc, h):
    pts, tri, phase = mesh(h)
    n = len(pts); rows, cols, vals = [], [], []; F = np.zeros(2 * n); Bs = []
    for e, t in enumerate(tri):
        (r1, z1), (r2, z2), (r3, z3) = pts[t]
        det = (r2 - r1) * (z3 - z1) - (r3 - r1) * (z2 - z1)
        b = np.array([z2 - z3, z3 - z1, z1 - z2]) / det
        c = np.array([r3 - r2, r1 - r3, r2 - r1]) / det
        rc = (r1 + r2 + r3) / 3; A = 0.5 * abs(det)
        B = np.zeros((4, 6))
        B[0, 0::2] = b; B[1, 0::2] = 1 / (3 * rc); B[2, 1::2] = c; B[3, 0::2] = c; B[3, 1::2] = b
        lam, mu, bet = MAT[phase[e]]
        w = 2 * np.pi * rc * A
        dofs = np.empty(6, int); dofs[0::2] = 2 * t; dofs[1::2] = 2 * t + 1
        rows.append(np.repeat(dofs, 6)); cols.append(np.tile(dofs, 6))
        vals.append((w * B.T @ Dmat(lam, mu) @ B).ravel())
        F[dofs] += w * B.T @ (bet * DT * np.array([1.0, 1.0, 1.0, 0.0]))
        Bs.append(B)
    K = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(2 * n, 2 * n))
    tol = 1e-9; fixed = np.zeros(2 * n, bool)
    fixed[2 * np.where(pts[:, 0] < tol)[0]] = True                       # 軸: u_r = 0
    fixed[2 * np.where(np.abs(pts[:, 0] - R_CELL) < tol)[0]] = True      # 外周: u_r = 0
    fixed[2 * np.where(pts[:, 1] < tol)[0] + 1] = True                   # 中央面: w = 0
    if bc == "clamped":
        top = np.where(np.abs(pts[:, 1] - H_HALF) < tol)[0]
        fixed[2 * top] = True; fixed[2 * top + 1] = True
    U = np.zeros(2 * n); free = ~fixed
    U[free] = spla.spsolve(K[free][:, free].tocsc(), F[free])
    S = np.zeros((len(tri), 4))
    for e, t in enumerate(tri):
        dofs = np.empty(6, int); dofs[0::2] = 2 * t; dofs[1::2] = 2 * t + 1
        lam, mu, bet = MAT[phase[e]]
        S[e] = Dmat(lam, mu) @ (Bs[e] @ U[dofs]) - bet * DT * np.array([1.0, 1.0, 1.0, 0.0])
    return {"pts": pts, "tri": tri, "phase": phase, "u": U.reshape(-1, 2), "S": S, "cen": pts[tri].mean(1)}


def von_mises(S):
    rr, tt, zz, rz = S[:, 0], S[:, 1], S[:, 2], S[:, 3]
    return np.sqrt(0.5 * ((rr - tt) ** 2 + (tt - zz) ** 2 + (zz - rr) ** 2) + 3 * rz ** 2)


# ==============================================================
# 2. 特徴量と最小二乗
# ==============================================================
def scaled(make):
    """30 / 31 の特徴量は入力を /4 して使うので，r と z を別々に [0, 4] へ伸縮する"""
    kr, kz = 4.0 / R_CELL, 4.0 / H_HALF

    def wrap(*args):
        f = make(*args)
        return lambda r, z: f(kr * r, kz * z)
    return wrap


def phase_derivs(feat, is_cu):
    """(u 用, w 用) の derivs．Cu の u_r は r·φ として軸で 0 にする"""
    dw = T.derivs(feat)
    du = T.derivs(lambda r, z: r * feat(r, z)) if is_cu else dw
    return du, dw


def ev(d, P):
    Q = jnp.asarray(P)
    return np.asarray(d[0](Q)), np.asarray(d[1](Q)), np.asarray(d[2](Q))


def stress_rows(du, dw, P, lam, mu):
    """[c_u | c_w] に対する σ_rr, σ_θθ, σ_zz, σ_rz の行（熱の項 −βΔT は含まない）"""
    U, JU, _ = ev(du, P); _, JW, _ = ev(dw, P)
    r = P[:, :1]
    Ur, Uz, Wr, Wz = JU[:, :, 0], JU[:, :, 1], JW[:, :, 0], JW[:, :, 1]
    Uor = U / r
    Z = np.zeros_like(U)
    srr = np.c_[(lam + 2 * mu) * Ur + lam * Uor, lam * Wz]
    stt = np.c_[lam * Ur + (lam + 2 * mu) * Uor, lam * Wz]
    szz = np.c_[lam * (Ur + Uor), (lam + 2 * mu) * Wz]
    srz = np.c_[mu * Uz, mu * Wr]
    return srr, stt, szz, srz, U, Z


def collocation(n_cu, n_si, n_if, n_edge, seed):
    g = np.random.default_rng(seed)
    p_cu = np.stack([A_R * np.sqrt(0.001 + 0.999 * g.random(n_cu)), H_HALF * g.random(n_cu)], 1)
    p_si = np.stack([A_R + (R_CELL - A_R) * g.random(n_si), H_HALF * g.random(n_si)], 1)
    zi = np.linspace(0, H_HALF, n_if)
    p_if = np.stack([np.full(n_if, A_R), zi], 1)
    s_r = np.linspace(1e-3, R_CELL, n_edge); s_z = np.linspace(0, H_HALF, n_edge)
    edges = {"mid": np.stack([s_r, 0 * s_r], 1), "top": np.stack([s_r, 0 * s_r + H_HALF], 1),
             "outer": np.stack([0 * s_z + R_CELL, s_z], 1), "axis": np.stack([0 * s_z, s_z], 1)}
    return p_cu, p_si, p_if, edges


def assemble(d, bc, pts):
    """d = {0: (du, dw), 1: (du, dw)}．未知数 = [Cu u | Cu w | Si u | Si w]"""
    p_cu, p_si, p_if, edges = pts
    m = {ph: np.asarray(d[ph][1][0](jnp.asarray(p_si[:1]))).shape[1] for ph in (0, 1)}
    off = {0: 0, 1: 2 * m[0]}; ntot = 2 * m[0] + 2 * m[1]
    rows, rhs = [], []

    def add(ph, blk, b, w):
        R = np.zeros((len(b), ntot)); R[:, off[ph]:off[ph] + blk.shape[1]] = blk
        rows.append(w * R); rhs.append(w * np.asarray(b, float))

    def add2(blk1, blk2, b, w):
        R = np.zeros((len(b), ntot)); R[:, 0:2 * m[0]] = blk1; R[:, off[1]:off[1] + 2 * m[1]] = blk2
        rows.append(w * R); rhs.append(w * np.asarray(b, float))

    # 釣り合い式
    for ph, P in ((0, p_cu), (1, p_si)):
        lam, mu, _ = MAT[ph]; du, dw = d[ph]
        U, JU, HU = ev(du, P); _, JW, HW = ev(dw, P); r = P[:, :1]
        Urr, Uzz, Urz, Ur, Uz = HU[:, :, 0, 0], HU[:, :, 1, 1], HU[:, :, 0, 1], JU[:, :, 0], JU[:, :, 1]
        Wrr, Wzz, Wrz, Wr = HW[:, :, 0, 0], HW[:, :, 1, 1], HW[:, :, 0, 1], JW[:, :, 0]
        er = np.c_[(lam + 2 * mu) * (Urr + Ur / r - U / r ** 2) + mu * Uzz, (lam + mu) * Wrz]
        ez = np.c_[(lam + mu) * (Urz + Uz / r), mu * (Wrr + Wr / r) + (lam + 2 * mu) * Wzz]
        w = 1 / np.sqrt(len(P))
        add(ph, er, np.zeros(len(P)), w); add(ph, ez, np.zeros(len(P)), w)
    # 界面 r = a
    n = len(p_if); w = 1 / np.sqrt(n)
    s1 = stress_rows(*d[0], p_if, *MAT[0][:2]); s2 = stress_rows(*d[1], p_if, *MAT[1][:2])
    U1 = ev(d[0][0], p_if)[0]; W1 = ev(d[0][1], p_if)[0]; U2 = ev(d[1][0], p_if)[0]; W2 = ev(d[1][1], p_if)[0]
    Z1, Z2 = np.zeros_like(U1), np.zeros_like(U2)
    add2(np.c_[U1, Z1], -np.c_[U2, Z2], np.zeros(n), w)
    add2(np.c_[Z1, W1], -np.c_[Z2, W2], np.zeros(n), w)
    add2(s1[0], -s2[0], np.full(n, (MAT[0][2] - MAT[1][2]) * DT), w)     # σ_rr の連続（熱の跳び）
    add2(s1[3], -s2[3], np.zeros(n), w)                                    # σ_rz の連続
    # 境界
    for name, P in edges.items():
        for ph in (0, 1):
            mask = (P[:, 0] <= A_R) if ph == 0 else (P[:, 0] >= A_R)
            if name == "axis" and ph == 1 or name == "outer" and ph == 0 or not mask.any():
                continue
            Q = P[mask]; w = 1 / np.sqrt(len(P)); lam, mu, bet = MAT[ph]; du, dw = d[ph]
            U = ev(du, Q)[0]; Wv, JW, _ = ev(dw, Q); Z = np.zeros_like(U)
            if name == "axis":
                add(ph, np.c_[Z, JW[:, :, 0]], np.zeros(len(Q)), w)                 # ∂w/∂r = 0
                continue
            srr, stt, szz, srz, _, _ = stress_rows(du, dw, Q, lam, mu)
            if name == "mid":
                add(ph, np.c_[Z, Wv], np.zeros(len(Q)), w); add(ph, srz, np.zeros(len(Q)), w)
            elif name == "outer":
                add(ph, np.c_[U, Z], np.zeros(len(Q)), w); add(ph, srz, np.zeros(len(Q)), w)
            elif name == "top" and bc == "clamped":
                add(ph, np.c_[U, Z], np.zeros(len(Q)), w); add(ph, np.c_[Z, Wv], np.zeros(len(Q)), w)
            elif name == "top":
                add(ph, szz, np.full(len(Q), bet * DT), w); add(ph, srz, np.zeros(len(Q)), w)
    return np.concatenate(rows), np.concatenate(rhs), m


def predict(d, c, m, P, phase):
    U = np.zeros((len(P), 2)); S = np.zeros((len(P), 4))
    off = {0: 0, 1: 2 * m[0]}
    for ph in (0, 1):
        mask = phase == ph
        if not mask.any():
            continue
        Q = P[mask]; lam, mu, bet = MAT[ph]; du, dw = d[ph]
        cu, cw = c[off[ph]:off[ph] + m[ph]], c[off[ph] + m[ph]:off[ph] + 2 * m[ph]]
        srr, stt, szz, srz, Ub, _ = stress_rows(du, dw, Q, lam, mu)
        cc = np.r_[cu, cw]
        U[mask] = np.stack([Ub @ cu, ev(dw, Q)[0] @ cw], 1)
        S[mask] = np.stack([srr @ cc - bet * DT, stt @ cc - bet * DT, szz @ cc - bet * DT, srz @ cc], 1)
    return U, S


def run(feat_cu, feat_si, bc, pts, ref, ridge=1e-10):
    d = {0: phase_derivs(feat_cu, True), 1: phase_derivs(feat_si, False)}
    A, b, m = assemble(d, bc, pts)
    c = T.solve(A, b, ridge)
    g = np.random.default_rng(0)
    iu = g.choice(len(ref["pts"]), min(20000, len(ref["pts"])), replace=False)
    P = np.maximum(ref["pts"][iu], [1e-6, 0]); nph = (P[:, 0] > A_R + 1e-9).astype(int)
    U, _ = predict(d, c, m, P, nph)
    e_u = np.linalg.norm(U - ref["u"][iu]) / np.linalg.norm(ref["u"][iu])
    out = {"u": e_u}
    zc = ref["cen"][:, 1]
    for key, band in (("mid", zc < 0.5), ("top", (zc > H_HALF - 1.0) & (zc < H_HALF - 0.25))):
        _, S = predict(d, c, m, ref["cen"][band], ref["phase"][band])
        vm, vm_ref = von_mises(S), von_mises(ref["S"][band])
        out["vm_" + key] = float(np.mean(np.abs(vm - vm_ref)) / vm_ref.max())
    return out


# ==============================================================
# 3. 実行
# ==============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc", default="clamped,free")
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--h", type=float, default=0.02)
    ap.add_argument("--out", default="results/tsv_axisym.json")
    a = ap.parse_args()
    m = 2 ** a.N; rows = []
    grid = {
        "QERC-Haar": ([1.0, 2.0, 4.0], scaled(lambda sd, s: T.make_qerc(a.N, sd, s))),
        "MLP2-RF": ([1.0, 2.0, 4.0], scaled(lambda sd, s: C.make_mlp2(a.N, sd, s))),
        "tanh-RF": ([2.0, 4.0, 8.0], scaled(lambda sd, s: T.make_tanh(m, sd, s))),
        "Fourier": ([4.0, 8.0, 16.0, 24.0], scaled(lambda sd, s: T.make_fourier(m, sd, s))),
    }
    for bc in a.bc.split(","):
        t0 = time.time(); ref = fe_solve(bc, a.h); ref2 = fe_solve(bc, 2 * a.h)
        from scipy.interpolate import LinearNDInterpolator
        du = np.nanmean(np.linalg.norm(LinearNDInterpolator(ref2["pts"], ref2["u"])(ref["pts"]) - ref["u"], axis=1)) \
            / np.mean(np.linalg.norm(ref["u"], axis=1))
        print(f"bc={bc}: FE 要素 {len(ref['tri'])}, h と 2h の差（u）≈ {du:.1e}  {time.time()-t0:.0f}s", flush=True)
        for sd in range(a.seeds):
            pts = collocation(3000, 8000, 300, 300, 10 + sd)
            for name, (scales, ctor) in grid.items():
                for s in scales:
                    t0 = time.time()
                    try:
                        res = run(ctor(2 * sd, s), ctor(2 * sd + 1, s), bc, pts, ref)
                    except np.linalg.LinAlgError:
                        continue
                    rows.append({"bc": bc, "model": name, "seed": sd, "scale": s, "fe_du": du, **res})
                    print(f"bc={bc} seed={sd} {name:10s} ×{s:<4} u={res['u']:.2e} "
                          f"vm_mid={res['vm_mid']:.2e} vm_top={res['vm_top']:.2e} {time.time()-t0:.0f}s", flush=True)
                    json.dump(rows, open(a.out, "w"), indent=1)
    print("\n=== シードごとに u 最良の倍率（幾何平均）===")
    for bc in a.bc.split(","):
        for name in grid:
            best = []
            for sd in range(a.seeds):
                rr = [r for r in rows if r["bc"] == bc and r["model"] == name and r["seed"] == sd]
                if rr:
                    best.append(min(rr, key=lambda r: r["u"]))
            if best:
                gm = lambda k: np.exp(np.mean(np.log([b[k] for b in best])))
                print(f"bc={bc:8s} {name:10s} u {gm('u'):.2e}  vm_mid {gm('vm_mid'):.2e}  vm_top {gm('vm_top'):.2e}")


if __name__ == "__main__":
    main()
