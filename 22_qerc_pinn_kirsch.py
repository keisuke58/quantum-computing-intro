# -*- coding: utf-8 -*-
"""
22 - QERC-PINN: 量子リザバーの特徴量で Kirsch 問題を「物理のみ」で解く（学習 = 最小二乗 1 回）

21_qerc_kirsch.py では，横磁場イジング時間発展の全基底確率 2^N 個を特徴量にすると，
解析解への教師ありフィットで変分 QPINN より 1〜2 桁良いことが分かった。ここでは解析解を使わず，
PDE と境界条件だけで解く（physics-informed extreme learning machine の量子版）。

  u_r = r Σ_k w_k F_k(x, y),   u_θ = r Σ_k c_k F_k(x, y)      （F_k: リザバーの特徴量）

と置くと，変位 (u, v) は重み (w, c) について線形。平面応力の平衡方程式，孔縁の表面力ゼロ，
外周の変位条件もすべて (w, c) について線形なので，

  min ‖ A [w; c] − b ‖²   （A: 特徴量の 1・2 階微分から作る行列）

を 1 回解けば終わる。反復最適化がないので，変分 QPINN の「最適化の壁」は原理的に存在しない。
特徴量の微分は JAX の自動微分で厳密に計算する（実機なら parameter-shift / 有限差分が必要）。

比較: 同数の古典ランダム特徴量（tanh, PIELM），同じ前処理・同じ解き方。
"""
import argparse, importlib.util, json, os, time
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

_s = importlib.util.spec_from_file_location(
    "k", os.path.join(os.path.dirname(os.path.abspath(__file__)), "13_qpinn_kirsch.py"))
K = importlib.util.module_from_spec(_s); _s.loader.exec_module(K)
_s = importlib.util.spec_from_file_location(
    "q", os.path.join(os.path.dirname(os.path.abspath(__file__)), "21_qerc_kirsch.py"))
Q = importlib.util.module_from_spec(_s); _s.loader.exec_module(Q)

A, R, S = K.A_HOLE, K.R_OUT, K.S_FAR
C, NU, MU = K.C_PS, K.NU, K.MU_G


# ==============================================================
# 1. 特徴量（入力 1 点 → 特徴量ベクトル）
# ==============================================================
def _pre(x, y):
    r = jnp.sqrt(x**2 + y**2); th = jnp.arctan2(y, x)
    xi = 0.95 * (2 * (A / r - A / R) / (1 - A / R) - 1)
    return r, th, jnp.arccos(xi)


def make_qerc(N, seed, t=1.0, scale=None, k_theta=None):
    """scale, k_theta を与えると低周波版: 全 qubit の RY に scale·a1，RZ(θ) は先頭 k_theta 個だけ。
    θ の最高周波数は k_theta，ρ 方向は scale で抑えられる（2 階微分での増幅を防ぐ）"""
    U = jnp.asarray(Q.ising_U(N, seed, t=t))

    def feat(x, y):
        _, th, a1 = _pre(x, y)
        psi = jnp.ones((1,), complex)
        for i in range(N):
            if scale is None:
                ay, az = (a1, th) if i % 2 == 0 else (th, a1)
            else:
                ay, az = scale * a1, (th if i < k_theta else 0.0 * th)
            q = jnp.stack([jnp.cos(ay / 2) * jnp.exp(-0.5j * az),
                           jnp.sin(ay / 2) * jnp.exp(0.5j * az)])
            psi = jnp.kron(psi, q)
        return jnp.abs(U @ psi) ** 2
    return feat, 2**N


def make_crf(m, seed):
    g = np.random.default_rng(seed)
    W = jnp.asarray(g.normal(size=(3, m)) * 1.5); b = jnp.asarray(g.random(m) * 2 * np.pi)

    def feat(x, y):
        _, th, a1 = _pre(x, y)
        return jnp.tanh(jnp.stack([a1, jnp.cos(th), jnp.sin(th)]) @ W + b)
    return feat, m


def make_basis(feat):
    """1 点 → (Φ_u, Φ_v)：u = Φ_u·[w; c], v = Φ_v·[w; c]"""
    def basis(p):
        x, y = p[0], p[1]
        r, th, _ = _pre(x, y)
        F = feat(x, y)
        cs, sn = jnp.cos(th), jnp.sin(th)
        bu = jnp.concatenate([r * cs * F, -r * sn * F])
        bv = jnp.concatenate([r * sn * F, r * cs * F])
        return jnp.stack([bu, bv])                 # (2, 2M)
    d1 = jax.vmap(jax.jacfwd(basis))               # (n, 2, 2M, 2)
    d2 = jax.vmap(jax.jacfwd(jax.jacfwd(basis)))   # (n, 2, 2M, 2, 2)
    return jax.jit(jax.vmap(basis)), jax.jit(d1), jax.jit(d2)


# ==============================================================
# 2. 線形系の組み立て
# ==============================================================
def points(n_in, n_bc, seed):
    g = np.random.default_rng(seed)
    r = np.sqrt(A**2 + g.random(n_in) * (R**2 - A**2)); th = g.random(n_in) * 2 * np.pi
    tb = np.linspace(0, 2 * np.pi, n_bc, endpoint=False)
    to = lambda rr, tt: np.stack([rr * np.cos(tt), rr * np.sin(tt)], 1)
    return to(r, th), to(np.full(n_bc, A), tb), to(np.full(n_bc, R), tb), tb


def stress_rows(J):
    """J: (n, 2, 2M, 2) = d(u,v)/d(x,y) の基底 → σxx, σyy, σxy の基底"""
    ux, uy, vx, vy = J[:, 0, :, 0], J[:, 0, :, 1], J[:, 1, :, 0], J[:, 1, :, 1]
    return C * (ux + NU * vy), C * (vy + NU * ux), MU * (uy + vx)


def assemble(fns, xy_in, xy_hole, xy_out, tb, w_bc):
    B, D1, D2 = fns
    H = D2(jnp.asarray(xy_in))                     # (n, 2, 2M, 2, 2)
    uxx, uyy, uxy = H[:, 0, :, 0, 0], H[:, 0, :, 1, 1], H[:, 0, :, 0, 1]
    vxx, vyy, vxy = H[:, 1, :, 0, 0], H[:, 1, :, 1, 1], H[:, 1, :, 0, 1]
    eq1 = C * (uxx + NU * vxy) + MU * (uyy + vxy)
    eq2 = MU * (uxy + vxx) + C * (vyy + NU * uxy)
    sxx, syy, sxy = stress_rows(D1(jnp.asarray(xy_hole)))
    nx, ny = np.cos(tb)[:, None], np.sin(tb)[:, None]
    t1, t2 = sxx * nx + sxy * ny, sxy * nx + syy * ny
    Bo = B(jnp.asarray(xy_out))
    ue, ve = K.uv_exact(xy_out[:, 0], xy_out[:, 1])
    s_in = 1 / np.sqrt(len(xy_in)); s_b = w_bc / np.sqrt(len(tb))
    Amat = np.concatenate([s_in * eq1, s_in * eq2, s_b * t1, s_b * t2,
                           s_b * Bo[:, 0], s_b * Bo[:, 1]])
    bvec = np.concatenate([np.zeros(2 * len(xy_in) + 2 * len(tb)), s_b * ue, s_b * ve])
    return np.asarray(Amat), bvec


def solve(Amat, b, ridge):
    # 列スケーリング + リッジ付きの正規方程式（2M ≤ 2048 なので直接解ける）
    sc = np.linalg.norm(Amat, axis=0) + 1e-30
    As = Amat / sc
    G = As.T @ As
    G[np.diag_indices_from(G)] += ridge
    return np.linalg.solve(G, As.T @ b) / sc


# ==============================================================
# 3. 評価
# ==============================================================
def evaluate(fns, W):
    B, D1, _ = fns
    g = np.random.default_rng(123)
    n = 3000
    r = np.sqrt(A**2 + g.random(n) * (R**2 - A**2)); th = g.random(n) * 2 * np.pi
    xy = np.stack([r * np.cos(th), r * np.sin(th)], 1)
    P = np.asarray(B(jnp.asarray(xy))) @ W           # (n, 2)
    ue, ve = K.uv_exact(xy[:, 0], xy[:, 1])
    e_u = np.sqrt(((P[:, 0] - ue)**2 + (P[:, 1] - ve)**2).sum() / (ue**2 + ve**2).sum())
    sp = [np.asarray(s) @ W for s in stress_rows(D1(jnp.asarray(xy)))]
    se = K.stress_exact(xy[:, 0], xy[:, 1])
    e_s = np.sqrt(sum(((a - b)**2).sum() for a, b in zip(sp, se)) / sum((b**2).sum() for b in se))
    # K_t: 孔縁の周方向応力の最大値 / S
    tt = np.linspace(0, 2 * np.pi, 721)
    xh = np.stack([A * np.cos(tt), A * np.sin(tt)], 1)
    sxx, syy, sxy = [np.asarray(s) @ W for s in stress_rows(D1(jnp.asarray(xh)))]
    stt = sxx * np.sin(tt)**2 + syy * np.cos(tt)**2 - 2 * sxy * np.sin(tt) * np.cos(tt)
    return e_u, e_s, stt.max() / S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", default="6,8,10")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--n_in", type=int, default=3000)
    ap.add_argument("--n_bc", type=int, default=400)
    ap.add_argument("--w_bc", type=float, default=10.0)
    ap.add_argument("--ridge", type=float, default=1e-10)
    ap.add_argument("--t", type=float, default=1.0, help="イジング時間発展の時間")
    ap.add_argument("--scale", type=float, default=None, help="低周波版: ρ 角の倍率")
    ap.add_argument("--k_theta", type=int, default=2, help="低周波版: θ を入れる qubit 数")
    ap.add_argument("--only_q", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = []
    for N in map(int, a.N.split(",")):
        for sd in range(a.seeds):
            pts = points(a.n_in, a.n_bc, 100 + sd)
            for kind in (("QERC",) if a.only_q else ("QERC", "古典RF")):
                feat, m = (make_qerc(N, sd, a.t, a.scale, a.k_theta) if kind == "QERC"
                           else make_crf(2**N, sd))
                t0 = time.time()
                fns = make_basis(feat)
                Amat, b = assemble(fns, *pts, a.w_bc)
                W = solve(Amat, b, a.ridge)
                t = time.time() - t0
                e_u, e_s, kt = evaluate(fns, W)
                rows.append({"kind": kind, "N": N, "n_feat": m, "seed": sd,
                             "u_err": e_u, "s_err": e_s, "kt": kt, "time": t})
                print(f"{kind:6s} N={N:<2} 特徴量={m:<5} seed={sd}  u={e_u:.2e}  σ={e_s:.2e}  "
                      f"K_t={kt:.3f}  {t:.0f}s", flush=True)
                if a.out:
                    json.dump(rows, open(a.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
