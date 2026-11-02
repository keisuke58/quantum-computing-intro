# -*- coding: utf-8 -*-
"""
38 - CFRP 繊維単位セルの均質化: 量子リザバー特徴量 vs 古典特徴量

正方配列の炭素繊維（半径 a = 1）/ エポキシ．繊維体積率 Vf から 1/4 セル [0, H]²（H = sqrt(π/(4Vf))）を決め，
繊維は原点（セルの角）に置く．2 つの問題を解き，等価弾性率を求める．

(a) 繊維方向のせん断（面外せん断）: w(x, y) は ∇·(G∇w) = 0．巨視的せん断 Γ = 1 を与えると，対称性から
      x = 0: w = 0，x = H: w = H，y = 0, H: ∂w/∂y = 0
    等価 G12 = <G ∂w/∂x>（セル平均）．解析式: Halpin-Tsai（ξ = 1，Hashin の複合円筒と一致）
(b) 繊維に垂直な面内（平面ひずみ）: 巨視的ひずみ ε̄xx = 1（ε̄yy = 0）を与えると
      x = 0: u = 0，x = H: u = H，y = 0, H: v = 0，4 辺で σxy = 0
    等価 C11 = <σxx>．

参照解は 34 と同じ界面適合メッシュの P1 FE（h と 2h で収束確認）．
学習側は繊維とマトリクスで別々の特徴量を使い，界面で連続条件を課して最小二乗 1 回で解く．
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


P34 = _load("p34", "34_tsv_periodic.py")
T, C = P34.T, P34.C
A_R = 1.0

# 材料（GPa）．繊維: T300 級の横方向特性を等方で近似，マトリクス: エポキシ
G_F, G_M = 15.0, 1.3                                   # 面外せん断 G12
E_F, NU_F, E_M, NU_M = 15.0, 0.2, 3.5, 0.35            # 面内（横方向）


def lame(E, nu):
    return E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))


MAT_IN = {0: lame(E_F, NU_F), 1: lame(E_M, NU_M)}      # 0: 繊維，1: マトリクス
MAT_AP = {0: G_F, 1: G_M}


def halpin_tsai_g12(vf):
    eta = (G_F / G_M - 1) / (G_F / G_M + 1)
    return G_M * (1 + eta * vf) / (1 - eta * vf)


def cell_size(vf):
    return np.sqrt(np.pi / (4 * vf))


# ==============================================================
# 1. 参照解（P1 FE）
# ==============================================================
def _geom(pts, t):
    (x1, y1), (x2, y2), (x3, y3) = pts[t]
    det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    b = np.array([y2 - y3, y3 - y1, y1 - y2]) / det
    c = np.array([x3 - x2, x1 - x3, x2 - x1]) / det
    return b, c, 0.5 * abs(det)


def fe_antiplane(H, h):
    pts, tri, phase = P34.mesh_quarter(2 * H, h)
    n = len(pts); rows, cols, vals = [], [], []; G = np.zeros((len(tri), 3, 2)); area = np.zeros(len(tri))
    for e, t in enumerate(tri):
        b, c, A = _geom(pts, t)
        Bm = np.stack([b, c], 1)                       # 3×2（節点ごとの勾配）
        Ke = A * MAT_AP[phase[e]] * Bm @ Bm.T
        rows.append(np.repeat(t, 3)); cols.append(np.tile(t, 3)); vals.append(Ke.ravel())
        G[e] = Bm; area[e] = A
    K = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    w = np.zeros(n); tol = 1e-9
    fx0 = np.abs(pts[:, 0]) < tol; fxH = np.abs(pts[:, 0] - H) < tol
    w[fxH] = H
    fixed = fx0 | fxH; free = ~fixed
    rhs = -K[free][:, fixed] @ w[fixed]
    w[free] = spla.spsolve(K[free][:, free].tocsc(), rhs)
    grad = np.einsum("eij,ei->ej", G, w[tri])
    mu = np.array([MAT_AP[p] for p in phase])
    g_eff = np.sum(area * mu * grad[:, 0]) / H**2
    return {"pts": pts, "tri": tri, "phase": phase, "w": w, "area": area, "eff": g_eff}


def fe_inplane(H, h):
    pts, tri, phase = P34.mesh_quarter(2 * H, h)
    n = len(pts); rows, cols, vals = [], [], []; Bs = []; area = np.zeros(len(tri))
    for e, t in enumerate(tri):
        b, c, A = _geom(pts, t)
        B = np.zeros((3, 6)); B[0, 0::2] = b; B[1, 1::2] = c; B[2, 0::2] = c; B[2, 1::2] = b
        lam, mu = MAT_IN[phase[e]]
        D = np.array([[lam + 2 * mu, lam, 0], [lam, lam + 2 * mu, 0], [0, 0, mu]])
        dofs = np.empty(6, int); dofs[0::2] = 2 * t; dofs[1::2] = 2 * t + 1
        Ke = A * B.T @ D @ B
        rows.append(np.repeat(dofs, 6)); cols.append(np.tile(dofs, 6)); vals.append(Ke.ravel())
        Bs.append(B); area[e] = A
    K = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(2 * n, 2 * n))
    U = np.zeros(2 * n); fixed = np.zeros(2 * n, bool); tol = 1e-9
    ix0 = np.where(np.abs(pts[:, 0]) < tol)[0]; ixH = np.where(np.abs(pts[:, 0] - H) < tol)[0]
    iy0 = np.where(np.abs(pts[:, 1]) < tol)[0]; iyH = np.where(np.abs(pts[:, 1] - H) < tol)[0]
    fixed[2 * ix0] = True; fixed[2 * ixH] = True; U[2 * ixH] = H
    fixed[2 * iy0 + 1] = True; fixed[2 * iyH + 1] = True
    free = ~fixed
    U[free] = spla.spsolve(K[free][:, free].tocsc(), -K[free][:, fixed] @ U[fixed])
    S = np.zeros((len(tri), 3))
    for e, t in enumerate(tri):
        dofs = np.empty(6, int); dofs[0::2] = 2 * t; dofs[1::2] = 2 * t + 1
        lam, mu = MAT_IN[phase[e]]
        D = np.array([[lam + 2 * mu, lam, 0], [lam, lam + 2 * mu, 0], [0, 0, mu]])
        S[e] = D @ (Bs[e] @ U[dofs])
    c11 = np.sum(area * S[:, 0]) / H**2
    return {"pts": pts, "tri": tri, "phase": phase, "u": U.reshape(-1, 2), "S": S, "area": area, "eff": c11}


# ==============================================================
# 2. 最小二乗（PIELM 形式）
# ==============================================================
def collocation(H, n_f, n_m, n_if, n_edge, seed):
    g = np.random.default_rng(seed)
    r = A_R * np.sqrt(g.random(n_f)); t = g.random(n_f) * np.pi / 2
    p_f = np.stack([r * np.cos(t), r * np.sin(t)], 1)
    q = g.random((6 * n_m, 2)) * H
    p_m = q[np.hypot(q[:, 0], q[:, 1]) > A_R][:n_m]
    ti = np.linspace(0, np.pi / 2, n_if)
    p_if = np.stack([A_R * np.cos(ti), A_R * np.sin(ti)], 1)
    s = np.linspace(0, H, n_edge)
    edges = {"x0": np.stack([0 * s, s], 1), "xH": np.stack([0 * s + H, s], 1),
             "y0": np.stack([s, 0 * s], 1), "yH": np.stack([s, 0 * s + H], 1)}
    return p_f, p_m, p_if, ti, edges


class Sys:
    """[繊維の係数 | マトリクスの係数] に対する行を溜める"""
    def __init__(self, n1, n2):
        self.n1, self.n2, self.rows, self.rhs = n1, n2, [], []

    def add(self, b1, b2, b, w):
        n = len(b)
        b1 = np.zeros((n, self.n1)) if b1 is None else b1
        b2 = np.zeros((n, self.n2)) if b2 is None else b2
        self.rows.append(w * np.concatenate([b1, b2], 1)); self.rhs.append(w * np.asarray(b, float))

    def get(self):
        return np.concatenate(self.rows), np.concatenate(self.rhs)


def _split_edge(P):
    inside = np.hypot(P[:, 0], P[:, 1]) < A_R
    return ((0, inside), (1, ~inside))


def assemble_antiplane(d, H, pts):
    p_f, p_m, p_if, ti, edges = pts
    m = [np.asarray(d[k][0](jnp.asarray(p_f[:1]))).shape[1] for k in (0, 1)]
    S = Sys(*m)
    for ph, P in ((0, p_f), (1, p_m)):
        Hh = np.asarray(d[ph][2](jnp.asarray(P)))
        lap = Hh[:, :, 0, 0] + Hh[:, :, 1, 1]
        S.add(lap if ph == 0 else None, lap if ph == 1 else None, np.zeros(len(P)), 1 / np.sqrt(len(P)))
    n = len(p_if); w = 1 / np.sqrt(n); Q = jnp.asarray(p_if)
    F1, F2 = np.asarray(d[0][0](Q)), np.asarray(d[1][0](Q))
    J1, J2 = np.asarray(d[0][1](Q)), np.asarray(d[1][1](Q))
    nx, ny = np.cos(ti)[:, None], np.sin(ti)[:, None]
    S.add(F1, -F2, np.zeros(n), w)
    S.add(G_F * (J1[:, :, 0] * nx + J1[:, :, 1] * ny), -G_M * (J2[:, :, 0] * nx + J2[:, :, 1] * ny), np.zeros(n), w)
    for name, P in edges.items():
        for ph, mask in _split_edge(P):
            if not mask.any():
                continue
            Q = jnp.asarray(P[mask]); w = 1 / np.sqrt(len(P))
            if name[0] == "x":
                blk = np.asarray(d[ph][0](Q)); b = np.full(mask.sum(), 0.0 if name == "x0" else H)
            else:
                blk = np.asarray(d[ph][1](Q))[:, :, 1]; b = np.zeros(mask.sum())
            S.add(blk if ph == 0 else None, blk if ph == 1 else None, b, w)
    A, b = S.get()
    return A, b, m[0]


def assemble_inplane(d, H, pts):
    p_f, p_m, p_if, ti, edges = pts
    m = [np.asarray(d[k][0](jnp.asarray(p_f[:1]))).shape[1] for k in (0, 1)]
    S = Sys(2 * m[0], 2 * m[1])
    for ph, P in ((0, p_f), (1, p_m)):
        lam, mu = MAT_IN[ph]
        for e in T.navier_rows(np.asarray(d[ph][2](jnp.asarray(P))), lam, mu):
            S.add(e if ph == 0 else None, e if ph == 1 else None, np.zeros(len(P)), 1 / np.sqrt(len(P)))
    n = len(p_if); w = 1 / np.sqrt(n); Q = jnp.asarray(p_if)
    F1, F2 = np.asarray(d[0][0](Q)), np.asarray(d[1][0](Q))
    o1, o2 = np.zeros_like(F1), np.zeros_like(F2)
    S.add(np.c_[F1, o1], -np.c_[F2, o2], np.zeros(n), w)
    S.add(np.c_[o1, F1], -np.c_[o2, F2], np.zeros(n), w)
    nx, ny = np.cos(ti)[:, None], np.sin(ti)[:, None]
    t1 = T.traction_rows(np.asarray(d[0][1](Q)), *MAT_IN[0], nx, ny)
    t2 = T.traction_rows(np.asarray(d[1][1](Q)), *MAT_IN[1], nx, ny)
    S.add(t1[0], -t2[0], np.zeros(n), w); S.add(t1[1], -t2[1], np.zeros(n), w)
    for name, P in edges.items():
        k = 0 if name[0] == "x" else 1
        val = H if name == "xH" else 0.0
        for ph, mask in _split_edge(P):
            if not mask.any():
                continue
            Q = jnp.asarray(P[mask]); w = 1 / np.sqrt(len(P)); lam, mu = MAT_IN[ph]
            F = np.asarray(d[ph][0](Q)); Z = np.zeros_like(F); Jm = np.asarray(d[ph][1](Q))
            ud = np.c_[F, Z] if k == 0 else np.c_[Z, F]
            sxy = np.concatenate([mu * Jm[:, :, 1], mu * Jm[:, :, 0]], 1)
            S.add(ud if ph == 0 else None, ud if ph == 1 else None, np.full(mask.sum(), val), w)
            S.add(sxy if ph == 0 else None, sxy if ph == 1 else None, np.zeros(mask.sum()), w)
    A, b = S.get()
    return A, b, 2 * m[0]


def _eval_feats(d, ph, P, chunk=4096):
    F, J = [], []
    for i in range(0, len(P), chunk):
        Q = jnp.asarray(P[i:i + chunk]); F.append(np.asarray(d[ph][0](Q))); J.append(np.asarray(d[ph][1](Q)))
    return np.concatenate(F), np.concatenate(J)


def evaluate(kind, d, w, m1, ref, H, n_eval=20000):
    """FE 節点での場の誤差と，FE 要素重心での中点則による等価弾性率"""
    g = np.random.default_rng(0)
    iu = g.choice(len(ref["pts"]), min(n_eval, len(ref["pts"])), replace=False)
    P = ref["pts"][iu]; nph = (np.hypot(P[:, 0], P[:, 1]) > A_R + 1e-9).astype(int)
    cen = ref["pts"][ref["tri"]].mean(1); eph = ref["phase"]
    wsplit = (w[:m1], w[m1:])
    if kind == "antiplane":
        U = np.zeros(len(P)); flux = 0.0
        for ph in (0, 1):
            mk = nph == ph
            if mk.any():
                U[mk] = _eval_feats(d, ph, P[mk])[0] @ wsplit[ph]
            me = eph == ph
            Jm = _eval_feats(d, ph, cen[me])[1]
            flux += np.sum(ref["area"][me] * MAT_AP[ph] * (Jm[:, :, 0] @ wsplit[ph]))
        e_f = np.linalg.norm(U - ref["w"][iu]) / np.linalg.norm(ref["w"][iu])
    else:
        U = np.zeros((len(P), 2)); flux = 0.0
        for ph in (0, 1):
            wp = wsplit[ph]; mm = len(wp) // 2; wu, wv = wp[:mm], wp[mm:]
            mk = nph == ph
            if mk.any():
                F = _eval_feats(d, ph, P[mk])[0]; U[mk] = np.stack([F @ wu, F @ wv], 1)
            me = eph == ph
            Jm = _eval_feats(d, ph, cen[me])[1]
            lam, mu = MAT_IN[ph]
            sxx = (lam + 2 * mu) * (Jm[:, :, 0] @ wu) + lam * (Jm[:, :, 1] @ wv)
            flux += np.sum(ref["area"][me] * sxx)
        e_f = np.linalg.norm(U - ref["u"][iu]) / np.linalg.norm(ref["u"][iu])
    eff = flux / H**2
    return e_f, abs(eff - ref["eff"]) / ref["eff"], eff


def run(kind, feat_f, feat_m, H, pts, ref, chunked=False):
    der = C.chunked_derivs if chunked else T.derivs
    d = (der(feat_f), der(feat_m))
    A, b, m1 = (assemble_antiplane if kind == "antiplane" else assemble_inplane)(d, H, pts)
    w = T.solve(A, b, 1e-10)
    return evaluate(kind, d, w, m1, ref, H)


# ==============================================================
# 3. 実行
# ==============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vf", default="0.3,0.5,0.6")
    ap.add_argument("--kind", default="antiplane,inplane")
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--h", type=float, default=0.005)
    ap.add_argument("--out", default="results/cfrp_unitcell.json")
    a = ap.parse_args()
    m = 2**a.N; rows = []; fe_rows = []
    for vf in map(float, a.vf.split(",")):
        H = cell_size(vf)
        for kind in a.kind.split(","):
            fe = fe_antiplane if kind == "antiplane" else fe_inplane
            t0 = time.time(); ref = fe(H, a.h); ref2 = fe(H, 2 * a.h)
            de = abs(ref["eff"] - ref2["eff"]) / ref["eff"]
            ht = halpin_tsai_g12(vf) if kind == "antiplane" else None
            fe_rows.append({"vf": vf, "kind": kind, "fe_eff": ref["eff"], "fe_eff_2h": ref2["eff"], "halpin_tsai": ht})
            print(f"Vf={vf} {kind}: H={H:.3f} FE 等価係数 {ref['eff']:.4f} GPa（2h との差 {de:.1e}）"
                  + (f"，Halpin-Tsai {ht:.4f}" if ht else "") + f"  {time.time()-t0:.0f}s", flush=True)
            L = 2 * H
            grid = {
                "QERC-Haar": ([1.0, 2.0, 4.0], P34.scaled(lambda sd, s: T.make_qerc(a.N, sd, s), L)),
                "MLP2-RF": ([1.0, 2.0, 4.0], P34.scaled(lambda sd, s: C.make_mlp2(a.N, sd, s), L)),
                "tanh-RF": ([2.0, 4.0, 8.0], P34.scaled(lambda sd, s: T.make_tanh(m, sd, s), L)),
                "Fourier": ([4.0, 8.0, 16.0, 24.0], P34.scaled(lambda sd, s: T.make_fourier(m, sd, s), L)),
            }
            for sd in range(a.seeds):
                pts = collocation(H, 2000, 4000, 300, 200, 10 + sd)
                for name, (scales, ctor) in grid.items():
                    for s in scales:
                        t0 = time.time()
                        ef, ee, eff = run(kind, ctor(2 * sd, s), ctor(2 * sd + 1, s), H, pts, ref)
                        rows.append({"vf": vf, "kind": kind, "model": name, "seed": sd, "scale": s,
                                     "field": ef, "eff_err": ee, "eff": eff, "fe_eff": ref["eff"], "fe_de": de})
                        print(f"Vf={vf} {kind:9s} seed={sd} {name:10s} ×{s:<4} 場 {ef:.2e} 等価 {ee:.2e} "
                              f"{time.time()-t0:.0f}s", flush=True)
                        json.dump({"fe": fe_rows, "rows": rows}, open(a.out, "w"), indent=1)
    print("\n=== シードごとに場の誤差が最小の倍率（幾何平均）===")
    for vf in sorted({r["vf"] for r in rows}):
        for kind in dict.fromkeys(r["kind"] for r in rows):
            for name in dict.fromkeys(r["model"] for r in rows):
                best = []
                for sd in range(a.seeds):
                    rr = [r for r in rows if r["vf"] == vf and r["kind"] == kind and r["model"] == name and r["seed"] == sd]
                    if rr:
                        best.append(min(rr, key=lambda r: r["field"]))
                if best:
                    gm = lambda k: np.exp(np.mean(np.log([b[k] for b in best])))
                    print(f"Vf={vf} {kind:9s} {name:10s} 場 {gm('field'):.2e}  等価 {gm('eff_err'):.2e}")


if __name__ == "__main__":
    main()
