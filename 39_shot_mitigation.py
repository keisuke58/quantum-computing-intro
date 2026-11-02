# -*- coding: utf-8 -*-
"""
39 - ショットノイズ対策: 強形式（2 階微分）→ エネルギー形式（1 階微分のみ）と，ショット分割による偏りの除去

問題と雑音モデルは 27 と同じ（Kirsch，Ikeda 型エンコード，parameter-shift 則で微分を測る近似モデル）．
1 点・1 回路あたりの総ショット数 S を揃えて，次の 4 通りを比べる．

  strong       : 27 と同じ強形式最小二乗．正規方程式 AᵀA は雑音の分散だけ偏る（errors-in-variables）
  strong-split : ショットを S/2 ずつ 2 回に分け，独立な推定 A1, A2 から (A1ᵀA2 + A2ᵀA1)/2 を作る．
                 E[A1ᵀA2] = AᵀA なので偏りがない
  ritz         : エネルギー最小化（Ritz / DEM）．剛性 K = ∫ ε(Φ)ᵀ D ε(Φ) dA は 1 階微分だけで組める．
                 孔の表面力ゼロは自然境界条件，外周の変位はペナルティで課す
  ritz-split   : ritz の K と外周ペナルティ項を，独立な 2 回の推定の積で作る

積分は極座標の Gauss-Legendre（r）× 台形則（θ）．評価は 27 と同じく厳密な特徴量で行う．
リッジ（と ritz の外周ペナルティ β）は候補から誤差最小のものを選ぶ（全手法で同じ手順．評価誤差で選ぶので楽観的な上限）．
量子ビット数 N も振る: 確率 p ≈ 2^-N なので，1 特徴量あたりの相対雑音は √(2^N / S) で増える．
"""
import argparse, importlib.util, json, os, time
import numpy as np
import jax, jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
_h = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("s27", os.path.join(_h, "27_qerc_shot_noise.py"))
S27 = importlib.util.module_from_spec(_s); _s.loader.exec_module(S27)
M = S27.M
A_H, R_O = M.A, M.R
C, NU, MU = M.C, M.NU, M.MU


def quad_points(n_r, n_t):
    x, w = np.polynomial.legendre.leggauss(n_r)
    r = A_H + (x + 1) / 2 * (R_O - A_H); wr = w / 2 * (R_O - A_H)
    t = np.linspace(0, 2 * np.pi, n_t, endpoint=False); wt = 2 * np.pi / n_t
    rr, tt = np.meshgrid(r, t, indexing="ij")
    W = (wr[:, None] * rr * wt).ravel()
    return np.stack([(rr * np.cos(tt)).ravel(), (rr * np.sin(tt)).ravel()], 1), W


def noisy_D1(fns, parts, xy, shots, rng):
    J = np.asarray(fns[1](jnp.asarray(xy)))
    if np.isfinite(shots):
        _, d1, _ = S27.noise_terms(jnp.asarray(xy), parts, shots, rng)
        J = J + S27.lift(S27.g_factors(xy), d1)
    return J


def noisy_B(fns, parts, xy, shots, rng):
    Bm = np.asarray(fns[0](jnp.asarray(xy)))
    if np.isfinite(shots):
        d0, _, _ = S27.noise_terms(jnp.asarray(xy), parts, shots, rng)
        Bm = Bm + S27.lift(S27.g_factors(xy), d0)
    return Bm


def strain_rows(J):
    ux, uy, vx, vy = J[:, 0, :, 0], J[:, 0, :, 1], J[:, 1, :, 0], J[:, 1, :, 1]
    return ux, vy, uy + vx


def stiffness(J1, J2, W):
    """Σ_q w_q ε(J1)ᵀ σ(J2)，対称化"""
    e1 = strain_rows(J1); s2 = M.stress_rows(J2)
    K = sum(a.T @ (W[:, None] * b) for a, b in zip(e1, s2))
    return (K + K.T) / 2


def solve_sym(G, r, ridge):
    s = np.sqrt(np.abs(np.diag(G))) + 1e-30
    Gs = G / np.outer(s, s)
    Gs[np.diag_indices_from(Gs)] += ridge
    return np.linalg.solve(Gs, r / s) / s


def best_solve(fns, G, r, ridges):
    best = None
    for rg in ridges:
        try:
            W = solve_sym(G.copy(), r, rg)
        except np.linalg.LinAlgError:
            continue
        e = M.evaluate(fns, W)
        if np.all(np.isfinite(e)) and (best is None or e[0] < best[0][0]):
            best = (e, rg)
    return best


def systems(method, fns, parts, pts, quad, n_bc, S, rng):
    """(G0, P, r0, r1): 正規行列 G = G0 + β·P，右辺 = r0 + β·r1（強形式は P = 0）"""
    xy_in, xy_hole, xy_out, tb = pts
    if method.startswith("strong"):
        if method == "strong":
            A, b = S27.assemble_noisy(fns, parts, xy_in, xy_hole, xy_out, tb, S, rng)
            G = A.T @ A; return G, None, A.T @ b, None
        A1, b = S27.assemble_noisy(fns, parts, xy_in, xy_hole, xy_out, tb, S / 2, rng)
        A2, _ = S27.assemble_noisy(fns, parts, xy_in, xy_hole, xy_out, tb, S / 2, rng)
        G = (A1.T @ A2 + A2.T @ A1) / 2
        return G, None, ((A1 + A2) / 2).T @ b, None
    xq, wq = quad
    ue, ve = M.K.uv_exact(xy_out[:, 0], xy_out[:, 1])
    wb = 2 * np.pi * R_O / n_bc
    if method == "ritz":
        J = noisy_D1(fns, parts, xq, S, rng); Bo = noisy_B(fns, parts, xy_out, S, rng)
        Kq = stiffness(J, J, wq)
        P = Bo[:, 0].T @ Bo[:, 0] + Bo[:, 1].T @ Bo[:, 1]
        r = Bo[:, 0].T @ ue + Bo[:, 1].T @ ve
    else:
        J1 = noisy_D1(fns, parts, xq, S / 2, rng); J2 = noisy_D1(fns, parts, xq, S / 2, rng)
        B1 = noisy_B(fns, parts, xy_out, S / 2, rng); B2 = noisy_B(fns, parts, xy_out, S / 2, rng)
        Kq = stiffness(J1, J2, wq)
        P = (B1[:, 0].T @ B2[:, 0] + B1[:, 1].T @ B2[:, 1]); P = (P + P.T) / 2
        Bm = (B1 + B2) / 2
        r = Bm[:, 0].T @ ue + Bm[:, 1].T @ ve
    return Kq, wb * P, np.zeros_like(r), wb * r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ns", default="6,8,10")
    ap.add_argument("--kind", default="haar")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--methods", default="strong,strong-split,ritz,ritz-split")
    ap.add_argument("--shots", default="1e3,1e4,1e5,1e6,1e7,inf")
    ap.add_argument("--ridges", default="1e-10,1e-8,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1")
    ap.add_argument("--betas", default="1,10,100,1000")
    ap.add_argument("--n_in", type=int, default=8000)
    ap.add_argument("--n_r", type=int, default=40)
    ap.add_argument("--n_t", type=int, default=200)
    ap.add_argument("--out", default="results/shot_mitigation.json")
    a = ap.parse_args()
    ridges = list(map(float, a.ridges.split(",")))
    quad = quad_points(a.n_r, a.n_t)
    betas = list(map(float, a.betas.split(",")))
    rows = []
    for N in map(int, a.Ns.split(",")):
        for sd in range(a.seeds):
            pts = M.points(a.n_in, 800, 100 + sd)
            parts = S27.build(N, sd, a.kind)
            fns = M.make_basis(parts[0])
            for S in map(float, a.shots.split(",")):
                for meth in a.methods.split(","):
                    if not np.isfinite(S) and meth.endswith("split"):
                        continue
                    t0 = time.time()
                    rng = np.random.default_rng(7 + sd)
                    G0, P, r0, r1 = systems(meth, fns, parts, pts, quad, 800, S, rng)
                    best = None
                    for bt in ([None] if P is None else betas):
                        G, r = (G0, r0) if P is None else (G0 + bt * P, r0 + bt * r1)
                        res = best_solve(fns, G, r, ridges)
                        if res and (best is None or res[0][0] < best[0][0]):
                            best = (res[0], res[1], bt)
                    (eu, es, kt), rg, bt = best if best else ((np.nan,) * 3, np.nan, None)
                    rows.append({"method": meth, "kind": a.kind, "N": N, "seed": sd, "shots": S,
                                 "u": eu, "s": es, "kt": kt, "ridge": rg, "beta": bt})
                    print(f"N={N} seed={sd} S={S:.0e} {meth:12s} u={eu:.2e} σ={es:.2e} K_t={kt:.3f} "
                          f"ridge={rg:.0e} β={bt} {time.time()-t0:.0f}s", flush=True)
                    json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
