# -*- coding: utf-8 -*-
"""
27 - QERC-PINN（Ikeda 型エンコーダ）に測定ショットノイズを入れる

実機では確率 p_k(x) を S 回の測定から推定する。PDE には特徴量の空間微分も要るので，
エンコード角 a_l(x)（θ_l, φ_l）についての parameter-shift 則で微分を測ると想定する。
p_k は各角について周波数 1 の三角関数なので，shift 則は厳密:

  ∂_l p     = [p(a_l + π/2) − p(a_l − π/2)] / 2                    （回路 2 本）
  ∂_l² p    = [p(a_l + π) − p(a_l)] / 2                            （回路 2 本）
  ∂_l∂_m p  = [p(++) − p(+−) − p(−+) + p(−−)] / 4                   （回路 4 本）

各回路の推定値は分散 p(1−p)/S（多項分布）。シフトした回路の p を元の p で近似すると，
  Var[p] = p(1−p)/S,  Var[∂_l p] = p(1−p)/(2S),  Var[∂_l² p] = p(1−p)/(2S),  Var[∂_l∂_m p] = p(1−p)/(4S)
空間微分は連鎖律 dp/dx_i = Σ_l (∂a_l/∂x_i) ∂_l p などで組み立て，各項のノイズを独立なガウスとして足す
（回路の共有による相関は無視した近似モデル）。

ノイズは学習（線形系の組み立て）にだけ入れ，評価は厳密な特徴量で行う。
基底 Φ = g(x)·F(x) の微分については，ノイズの主要項 g·δF', g·δF'' だけを足す（g', g'' 項の小さなノイズは無視）。
"""
import argparse, importlib.util, json, os, time
import numpy as np
import jax, jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
_h = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("i", os.path.join(_h, "25_qerc_ikeda_pde.py"))
I = importlib.util.module_from_spec(_s); _s.loader.exec_module(I)
M = I.M
C, NU, MU = M.C, M.NU, M.MU


def build(N, seed, kind):
    """25 の make_ikeda と同じ乱数で，角度関数と角度→確率の関数を分けて作る"""
    g = np.random.default_rng(1000 + seed)
    W = jnp.asarray(g.normal(size=(3, 2 * N))); b = jnp.asarray(g.normal(size=2 * N))
    U = jnp.asarray(I.reservoir(kind, N, seed))

    def ang(p):
        _, th, a1 = M._pre(p[0], p[1])
        z = jnp.stack([a1 - 1.57, jnp.cos(th), jnp.sin(th)]) @ W + b
        return jnp.pi * (jnp.tanh(z) + 1) / 2

    def probs(a):
        psi = jnp.ones((1,), complex)
        for l in range(N):
            psi = jnp.kron(psi, jnp.stack([jnp.cos(a[l] / 2), jnp.exp(1j * a[N + l]) * jnp.sin(a[l] / 2)]))
        return jnp.abs(U @ psi) ** 2

    def feat(x, y):
        return probs(ang(jnp.stack([x, y])))
    return feat, jax.jit(jax.vmap(ang)), jax.jit(jax.vmap(jax.jacfwd(ang))), \
        jax.jit(jax.vmap(jax.jacfwd(jax.jacfwd(ang)))), jax.jit(jax.vmap(probs))


def noise_terms(xy, parts, shots, rng):
    """各点の δF (M), δF' (M,2), δF'' (M,2,2) をサンプルする"""
    _, A, dA, d2A, P = parts
    a = A(xy); J = np.asarray(dA(xy)); Hh = np.asarray(d2A(xy))     # (n,2N),(n,2N,2),(n,2N,2,2)
    p = np.asarray(P(a)); v = p * (1 - p) / shots                   # (n, M)
    n, m = p.shape
    dF = np.zeros((n, m, 2)); d2F = np.zeros((n, m, 2, 2))
    for i in range(2):
        s1 = (J[:, :, i] ** 2).sum(1)[:, None]                      # Σ_l (∂a_l/∂x_i)²
        dF[:, :, i] = rng.normal(size=(n, m)) * np.sqrt(v * s1 / 2)
    for i, j in ((0, 0), (1, 1), (0, 1)):
        Ji, Jj = J[:, :, i], J[:, :, j]
        diag = ((Ji * Jj) ** 2).sum(1)                              # l = m の項
        off = (Ji ** 2).sum(1) * (Jj ** 2).sum(1) - diag            # l ≠ m の項
        hh = (Hh[:, :, i, j] ** 2).sum(1)                           # ∂²a_l/∂x_i∂x_j の項
        var = v * (diag / 2 + off / 4 + hh / 2)[:, None]
        d2F[:, :, i, j] = rng.normal(size=(n, m)) * np.sqrt(var)
        d2F[:, :, j, i] = d2F[:, :, i, j]
    dF0 = rng.normal(size=(n, m)) * np.sqrt(v)
    return dF0, dF, d2F


def g_factors(xy):
    r = np.sqrt((xy ** 2).sum(1)); th = np.arctan2(xy[:, 1], xy[:, 0])
    c, s = np.cos(th), np.sin(th)
    # Φ_u = [r c F, −r s F],  Φ_v = [r s F, r c F]
    return np.stack([np.stack([r * c, -r * s], 1), np.stack([r * s, r * c], 1)], 1)   # (n, 2out, 2blk)


def lift(gf, x):
    """g(x)·x を基底の形 (n, 2, 2M, ...) に並べる"""
    out = [np.concatenate([gf[:, o, 0][:, None] * x, gf[:, o, 1][:, None] * x], axis=1)
           if x.ndim == 2 else
           np.concatenate([gf[:, o, 0].reshape(-1, *[1] * (x.ndim - 1)) * x,
                           gf[:, o, 1].reshape(-1, *[1] * (x.ndim - 1)) * x], axis=1)
           for o in range(2)]
    return np.stack(out, 1)


def assemble_noisy(fns, parts, xy_in, xy_hole, xy_out, tb, shots, rng):
    B, D1, D2 = fns
    Hm = np.asarray(D2(jnp.asarray(xy_in)))
    D1h = np.asarray(D1(jnp.asarray(xy_hole)))
    Bo = np.asarray(B(jnp.asarray(xy_out)))
    if np.isfinite(shots):
        _, _, d2 = noise_terms(jnp.asarray(xy_in), parts, shots, rng)
        Hm = Hm + lift(g_factors(xy_in), d2)
        _, d1, _ = noise_terms(jnp.asarray(xy_hole), parts, shots, rng)
        D1h = D1h + lift(g_factors(xy_hole), d1)
        d0, _, _ = noise_terms(jnp.asarray(xy_out), parts, shots, rng)
        Bo = Bo + lift(g_factors(xy_out), d0)
    uxx, uyy, uxy = Hm[:, 0, :, 0, 0], Hm[:, 0, :, 1, 1], Hm[:, 0, :, 0, 1]
    vxx, vyy, vxy = Hm[:, 1, :, 0, 0], Hm[:, 1, :, 1, 1], Hm[:, 1, :, 0, 1]
    eq1 = C * (uxx + NU * vxy) + MU * (uyy + vxy)
    eq2 = MU * (uxy + vxx) + C * (vyy + NU * uxy)
    sxx, syy, sxy = M.stress_rows(D1h)
    nx, ny = np.cos(tb)[:, None], np.sin(tb)[:, None]
    t1, t2 = sxx * nx + sxy * ny, sxy * nx + syy * ny
    ue, ve = M.K.uv_exact(xy_out[:, 0], xy_out[:, 1])
    s_in = 1 / np.sqrt(len(xy_in)); s_b = 1 / np.sqrt(len(tb))
    A = np.concatenate([s_in * eq1, s_in * eq2, s_b * t1, s_b * t2, s_b * Bo[:, 0], s_b * Bo[:, 1]])
    b = np.concatenate([np.zeros(2 * len(xy_in) + 2 * len(tb)), s_b * ue, s_b * ve])
    return A, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=10)
    ap.add_argument("--kinds", default="cliffordT,haar")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--shots", default="1e3,1e4,1e5,1e6,1e7,inf")
    ap.add_argument("--ridges", default="1e-10,1e-6,1e-3")
    ap.add_argument("--n_in", type=int, default=8000)
    ap.add_argument("--out", default="results/qerc_shot_noise.json")
    a = ap.parse_args()
    rows = []
    for sd in range(a.seeds):
        pts = M.points(a.n_in, 800, 100 + sd)
        for kind in a.kinds.split(","):
            parts = build(a.N, sd, kind)
            fns = M.make_basis(parts[0])
            for S in map(float, a.shots.split(",")):
                rng = np.random.default_rng(7 + sd)
                A, b = assemble_noisy(fns, parts, *pts, S, rng)
                # ノイズがあると正則化が効くので，リッジを数通り試し最良を記録（全条件で同じ候補）
                best = None
                for rg in map(float, a.ridges.split(",")):
                    e = M.evaluate(fns, M.solve(A.copy(), b, rg))
                    if best is None or e[0] < best[0][0]:
                        best = (e, rg)
                (eu, es, kt), rg = best
                rows.append({"kind": kind, "seed": sd, "shots": S, "u": eu, "s": es, "kt": kt, "ridge": rg})
                print(f"seed={sd} {kind:9s} shots={S:.0e}  u={eu:.2e}  σ={es:.2e}  K_t={kt:.3f}  ridge={rg:.0e}",
                      flush=True)
                json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
