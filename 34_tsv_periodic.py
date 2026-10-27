# -*- coding: utf-8 -*-
"""
34 - TSV 周期配列の熱応力（単位セル）: 量子リザバー特徴量 vs 古典特徴量

正方格子（ピッチ L）に並んだ Cu ビア（半径 a = 1）を一様に冷却（ΔT = −250 K，平面ひずみ）．
巨視的なひずみを 0（周囲に拘束されたウェハ）とすると，周期性と単位セルの対称性から，
1/4 セル [0, L/2]² の 4 辺すべてで

    法線方向変位 = 0，接線方向の表面力 = 0（ローラー条件）

となる．周期境界条件をこの対称条件に置き換えて解く．

参照解: 自前の FE（P1 三角形，円弧に沿った適合メッシュ）．メッシュを 2 段階で細かくして収束を確認する．
学習側: 30 と同じく Cu と Si で別々の特徴量を使い，界面で変位と表面力の連続（熱膨張差による跳び）を課し，
最小二乗 1 回で解く．比較する特徴量:
  QERC（Ikeda 型エンコード + Haar），テンソル積の古典ランダム射影（31），2 層の古典ランダムネット（31），
  1 層 tanh / Fourier ランダム特徴量．
"""
import argparse, importlib.util, json, os, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import Delaunay
import jax, jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
_h = os.path.dirname(os.path.abspath(__file__))


def _load(name, fn):
    s = importlib.util.spec_from_file_location(name, os.path.join(_h, fn))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


T = _load("t30", "30_qerc_tsv_single.py")      # 材料定数，Navier・表面力の行，解法
C = _load("t31", "31_tsv_classical_controls.py")
A_R, DT = T.A_R, T.DT
MAT = {0: (T.LAM1, T.MU1, T.BET1), 1: (T.LAM2, T.MU2, T.BET2)}     # 0: Cu, 1: Si


# ==============================================================
# 1. 参照解: 1/4 セルの FE（P1，適合メッシュ）
# ==============================================================
def mesh_quarter(L, h):
    """[0, L/2]² の 1/4 セル．円弧 r = a 上に節点を置き，Delaunay で三角形分割"""
    H = L / 2
    pts = []
    n_arc = max(int(np.ceil((np.pi / 2) * A_R / h)), 8)
    # 円弧の内外に同心円の層を置く（界面を正確に表す）
    for r in np.concatenate([np.arange(h, A_R, h), [A_R], A_R + np.arange(1, 6) * h]):
        n = max(int(np.ceil((np.pi / 2) * r / h)), 2)
        t = np.linspace(0, np.pi / 2, n + 1)
        P = np.stack([r * np.cos(t), r * np.sin(t)], 1)
        pts.append(P[(P[:, 0] <= H + 1e-12) & (P[:, 1] <= H + 1e-12)])
    pts.append(np.zeros((1, 2)))
    # それより外側は正方格子
    g = np.linspace(0, H, int(np.ceil(H / h)) + 1)
    X, Y = np.meshgrid(g, g)
    P = np.stack([X.ravel(), Y.ravel()], 1)
    pts.append(P[np.hypot(P[:, 0], P[:, 1]) > A_R + 5.5 * h])
    pts = np.concatenate(pts)
    pts = np.unique(np.round(pts, 12), axis=0)
    tri = Delaunay(pts).simplices
    cen = pts[tri].mean(1)
    phase = (np.hypot(cen[:, 0], cen[:, 1]) > A_R).astype(int)      # 0: Cu, 1: Si
    area = 0.5 * np.abs(np.cross(pts[tri[:, 1]] - pts[tri[:, 0]], pts[tri[:, 2]] - pts[tri[:, 0]]))
    keep = area > 1e-14
    return pts, tri[keep], phase[keep]


def fe_solve(L, h):
    pts, tri, phase = mesh_quarter(L, h)
    H = L / 2
    n = len(pts); rows, cols, vals = [], [], []; F = np.zeros(2 * n)
    Bs = []
    for e, t in enumerate(tri):
        (x1, y1), (x2, y2), (x3, y3) = pts[t]
        det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        b = np.array([y2 - y3, y3 - y1, y1 - y2]) / det
        c = np.array([x3 - x2, x1 - x3, x2 - x1]) / det
        B = np.zeros((3, 6)); B[0, 0::2] = b; B[1, 1::2] = c; B[2, 0::2] = c; B[2, 1::2] = b
        lam, mu, bet = MAT[phase[e]]
        D = np.array([[lam + 2 * mu, lam, 0], [lam, lam + 2 * mu, 0], [0, 0, mu]])
        A = 0.5 * abs(det)
        dofs = np.empty(6, int); dofs[0::2] = 2 * t; dofs[1::2] = 2 * t + 1
        Ke = A * B.T @ D @ B
        fe = A * B.T @ (bet * DT * np.array([1.0, 1.0, 0.0]))       # σ = Dε − βΔT[1,1,0]
        rows.append(np.repeat(dofs, 6)); cols.append(np.tile(dofs, 6)); vals.append(Ke.ravel())
        F[dofs] += fe
        Bs.append(B)
    K = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(2 * n, 2 * n))
    fixed = np.zeros(2 * n, bool)
    tol = 1e-9
    fixed[2 * np.where(np.abs(pts[:, 0]) < tol)[0]] = True            # x = 0: u = 0
    fixed[2 * np.where(np.abs(pts[:, 0] - H) < tol)[0]] = True        # x = L/2: u = 0
    fixed[2 * np.where(np.abs(pts[:, 1]) < tol)[0] + 1] = True        # y = 0: v = 0
    fixed[2 * np.where(np.abs(pts[:, 1] - H) < tol)[0] + 1] = True    # y = L/2: v = 0
    free = ~fixed
    u = np.zeros(2 * n)
    u[free] = spla.spsolve(K[free][:, free].tocsc(), F[free])
    # 要素の応力
    S = np.zeros((len(tri), 3))
    for e, t in enumerate(tri):
        dofs = np.empty(6, int); dofs[0::2] = 2 * t; dofs[1::2] = 2 * t + 1
        lam, mu, bet = MAT[phase[e]]
        D = np.array([[lam + 2 * mu, lam, 0], [lam, lam + 2 * mu, 0], [0, 0, mu]])
        S[e] = D @ (Bs[e] @ u[dofs]) - bet * DT * np.array([1.0, 1.0, 0.0])
    return {"pts": pts, "tri": tri, "phase": phase, "u": u.reshape(-1, 2), "S": S,
            "cen": pts[tri].mean(1)}


# ==============================================================
# 2. 最小二乗（PIELM 形式）
# ==============================================================
def collocation(L, n_cu, n_si, n_if, n_edge, seed):
    g = np.random.default_rng(seed); H = L / 2
    r = A_R * np.sqrt(g.random(n_cu)); t = g.random(n_cu) * np.pi / 2
    p_cu = np.stack([r * np.cos(t), r * np.sin(t)], 1)
    q = g.random((4 * n_si, 2)) * H
    p_si = q[np.hypot(q[:, 0], q[:, 1]) > A_R][:n_si]
    ti = np.linspace(0, np.pi / 2, n_if)
    p_if = np.stack([A_R * np.cos(ti), A_R * np.sin(ti)], 1)
    s = np.linspace(0, H, n_edge)
    edges = {   # (点, 法線方向の成分 0=x/1=y, その辺が Cu を通るか)
        "x0": np.stack([0 * s, s], 1), "xH": np.stack([0 * s + H, s], 1),
        "y0": np.stack([s, 0 * s], 1), "yH": np.stack([s, 0 * s + H], 1)}
    return p_cu, p_si, p_if, ti, edges


def assemble(d_cu, d_si, pts, w_bc=1.0):
    p_cu, p_si, p_if, ti, edges = pts
    B1, J1, H1 = d_cu; B2, J2, H2 = d_si
    m1 = np.asarray(B1(jnp.asarray(p_cu[:1]))).shape[1]; m2 = np.asarray(B2(jnp.asarray(p_si[:1]))).shape[1]
    rows, rhs = [], []

    def add(blk1, blk2, b, w):
        n = len(b)
        blk1 = np.zeros((n, 2 * m1)) if blk1 is None else blk1
        blk2 = np.zeros((n, 2 * m2)) if blk2 is None else blk2
        rows.append(w * np.concatenate([blk1, blk2], 1)); rhs.append(w * b)

    for e in T.navier_rows(np.asarray(H1(jnp.asarray(p_cu))), T.LAM1, T.MU1):
        add(e, None, np.zeros(len(p_cu)), 1 / np.sqrt(len(p_cu)))
    for e in T.navier_rows(np.asarray(H2(jnp.asarray(p_si))), T.LAM2, T.MU2):
        add(None, e, np.zeros(len(p_si)), 1 / np.sqrt(len(p_si)))
    # 界面
    n = len(p_if); w = w_bc / np.sqrt(n)
    F1, F2 = np.asarray(B1(jnp.asarray(p_if))), np.asarray(B2(jnp.asarray(p_if)))
    o1, o2 = np.zeros_like(F1), np.zeros_like(F2)
    add(np.c_[F1, o1], -np.c_[F2, o2], np.zeros(n), w)
    add(np.c_[o1, F1], -np.c_[o2, F2], np.zeros(n), w)
    nx, ny = np.cos(ti)[:, None], np.sin(ti)[:, None]
    t1 = T.traction_rows(np.asarray(J1(jnp.asarray(p_if))), T.LAM1, T.MU1, nx, ny)
    t2 = T.traction_rows(np.asarray(J2(jnp.asarray(p_if))), T.LAM2, T.MU2, nx, ny)
    jump = (T.BET1 - T.BET2) * DT
    add(t1[0], -t2[0], jump * np.cos(ti), w)
    add(t1[1], -t2[1], jump * np.sin(ti), w)
    # 4 辺のローラー条件: 法線変位 0，接線表面力 0（σ_xy = 0）．辺上の点は相に応じて振り分け
    for name, P in edges.items():
        k = 0 if name[0] == "x" else 1                # 法線方向の変位成分
        inside = np.hypot(P[:, 0], P[:, 1]) < A_R
        for ph, mask in ((0, inside), (1, ~inside)):
            if mask.sum() == 0:
                continue
            Q = jnp.asarray(P[mask]); w = w_bc / np.sqrt(len(P))
            Bf, Jf = (B1, J1) if ph == 0 else (B2, J2)
            lam, mu, _ = MAT[ph]
            F = np.asarray(Bf(Q)); Z = np.zeros_like(F)
            ud = np.c_[F, Z] if k == 0 else np.c_[Z, F]
            Jm = np.asarray(Jf(Q))
            sxy = np.concatenate([mu * Jm[:, :, 1], mu * Jm[:, :, 0]], 1)
            for blk in (ud, sxy):
                add(blk if ph == 0 else None, blk if ph == 1 else None, np.zeros(mask.sum()), w)
    return np.concatenate(rows), np.concatenate(rhs), m1


def predict(d_cu, d_si, w, m1, P, phase):
    """点 P（相 phase）での変位と応力"""
    U = np.zeros((len(P), 2)); S = np.zeros((len(P), 3))
    for ph, (B, J, _) in ((0, d_cu), (1, d_si)):
        mask = phase == ph
        if not mask.any():
            continue
        wp = w[:2 * m1] if ph == 0 else w[2 * m1:]
        Q = jnp.asarray(P[mask]); F = np.asarray(B(Q)); Jm = np.asarray(J(Q)); m = F.shape[1]
        wu, wv = wp[:m], wp[m:]
        U[mask] = np.stack([F @ wu, F @ wv], 1)
        ux, uy, vx, vy = Jm[:, :, 0] @ wu, Jm[:, :, 1] @ wu, Jm[:, :, 0] @ wv, Jm[:, :, 1] @ wv
        lam, mu, bet = MAT[ph]
        S[mask] = np.stack([(lam + 2 * mu) * ux + lam * vy - bet * DT,
                            lam * ux + (lam + 2 * mu) * vy - bet * DT, mu * (uy + vx)], 1)
    return U, S


def run(feat_cu, feat_si, pts, ref, chunked=False):
    der = C.chunked_derivs if chunked else T.derivs
    d1, d2 = der(feat_cu), der(feat_si)
    A, b, m1 = assemble(d1, d2, pts)
    w = T.solve(A, b, 1e-10)
    # 評価点は FE 節点・要素重心から 20000 点ずつ抽出（メモリ節約）
    g = np.random.default_rng(0)
    iu = g.choice(len(ref["pts"]), min(20000, len(ref["pts"])), replace=False)
    ie = g.choice(len(ref["cen"]), min(20000, len(ref["cen"])), replace=False)
    P_u = ref["pts"][iu]
    node_phase = (np.hypot(P_u[:, 0], P_u[:, 1]) > A_R + 1e-9).astype(int)
    U, _ = predict(d1, d2, w, m1, P_u, node_phase)
    _, S = predict(d1, d2, w, m1, ref["cen"][ie], ref["phase"][ie])
    e_u = np.linalg.norm(U - ref["u"][iu]) / np.linalg.norm(ref["u"][iu])
    e_s = np.linalg.norm(S - ref["S"][ie]) / np.linalg.norm(ref["S"][ie])
    return e_u, e_s


# ==============================================================
# 3. 特徴量（座標を L/2 で規格化）
# ==============================================================
def scaled(make, L):
    """30 / 31 の特徴量は (x, y)/4 を入力にするので，座標を 4/(L/2) 倍して [0, 1] に揃える"""
    k = 4.0 / (L / 2)

    def wrap(*args):
        f = make(*args)
        if getattr(f, "extra", None) is not None:
            g = lambda x, y, R=f.extra: f(k * x, k * y, R)
            g.extra = f.extra
            return g
        return lambda x, y: f(k * x, k * y)
    return wrap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", default="3,4,6")
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--h", type=float, default=0.005)
    ap.add_argument("--with_tensor", action="store_true")
    ap.add_argument("--out", default="results/tsv_periodic.json")
    a = ap.parse_args()
    m = 2**a.N
    rows = []
    for L in map(float, a.L.split(",")):
        t0 = time.time()
        ref = fe_solve(L, a.h)
        ref2 = fe_solve(L, 2 * a.h)
        # 粗いメッシュとの差（参照解の精度の目安）
        from scipy.interpolate import LinearNDInterpolator
        ip = LinearNDInterpolator(ref2["pts"], ref2["u"])
        du = np.nanmean(np.linalg.norm(ip(ref["pts"]) - ref["u"], axis=1)) / np.mean(np.linalg.norm(ref["u"], axis=1))
        print(f"L={L}: FE 節点 {len(ref['pts'])}, 要素 {len(ref['tri'])}, h と 2h の差（u）≈ {du:.1e}  "
              f"{time.time()-t0:.0f}s", flush=True)
        grid = {
            "QERC-Haar": ([1.0, 2.0, 4.0], scaled(lambda sd, s: T.make_qerc(a.N, sd, s), L), False),
            "MLP2-RF": ([1.0, 2.0, 4.0], scaled(lambda sd, s: C.make_mlp2(a.N, sd, s), L), False),
            "tanh-RF": ([2.0, 4.0, 8.0, 16.0], scaled(lambda sd, s: T.make_tanh(m, sd, s), L), False),
            "Fourier": ([2.0, 4.0, 8.0, 16.0], scaled(lambda sd, s: T.make_fourier(m, sd, s), L), False),
        }
        if a.with_tensor:
            grid["tensor-RP"] = ([2.0], scaled(lambda sd, s: C.make_tensor_rp(a.N, sd, s), L), True)
        for sd in range(a.seeds):
            pts = collocation(L, 2000, 6000, 300, 200, 10 + sd)
            for name, (scales, ctor, chunk) in grid.items():
                for s in scales:
                    t0 = time.time()
                    eu, es = run(ctor(2 * sd, s), ctor(2 * sd + 1, s), pts, ref, chunk)
                    rows.append({"L": L, "model": name, "seed": sd, "scale": s, "u": eu, "s": es, "fe_du": du})
                    print(f"L={L} seed={sd} {name:10s} ×{s:<4} u={eu:.2e} σ={es:.2e} {time.time()-t0:.0f}s",
                          flush=True)
                    json.dump(rows, open(a.out, "w"), indent=1)
    print("\n=== シードごとに最良の倍率（u の幾何平均）===")
    for L in sorted({r["L"] for r in rows}):
        for name in dict.fromkeys(r["model"] for r in rows):
            v = [min(r["u"] for r in rows if r["L"] == L and r["model"] == name and r["seed"] == sd)
                 for sd in range(a.seeds) if any(r["L"] == L and r["model"] == name and r["seed"] == sd for r in rows)]
            if v:
                print(f"L={L} {name:10s} {np.exp(np.mean(np.log(v))):.2e}  {['%.1e' % x for x in v]}")


if __name__ == "__main__":
    main()
