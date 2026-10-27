# -*- coding: utf-8 -*-
"""
30 - TSV（シリコン貫通電極）1 本の熱応力を，ランダム特徴量 + 最小二乗（PIELM）で解く
     量子リザバー特徴量（QERC）と古典特徴量（tanh, Fourier）を比較する

問題（2D 平面ひずみ，一様な冷却 ΔT）:
  Cu ビア（半径 a）を Si（外半径 b）が囲む．各相の中は熱荷重なしの Navier 方程式
  （ΔT が一様なので体積力は出ない）．熱膨張の差は界面の表面力の跳びとして入る:
      [u] = 0,   σ_Cu n − σ_Si n = 0,   σ = λ tr(ε) I + 2μ ε − β ΔT I,  β = (3λ + 2μ) α
  外周 r = b では解析解の変位を与える．

解析解（軸対称）: Cu で u_r = A r，Si で u_r = C r + D / r．
  界面の変位・応力の連続と，外周の表面力ゼロから A, C, D を決める．

相ごとに別の特徴量 F_Cu(x), F_Si(x) を使い（領域分割），u, v をその線形結合で表す．
未知数 [w_u^Cu, w_v^Cu, w_u^Si, w_v^Si] について全条件が線形なので，最小二乗 1 回で解ける．

無次元化: 長さは a，弾性定数は 100 GPa，変位は 1e-3·a（熱ひずみ α ΔT を 1e-3 単位で表す）．
材料: Cu E=110 GPa, ν=0.35, α=17e-6/K．Si E=130 GPa, ν=0.28, α=2.6e-6/K．ΔT = −250 K．
"""
import argparse, json, time
import numpy as np
import jax, jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

A_R, B_R, DT = 1.0, 4.0, -250.0


def lame(E, nu):
    return E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))


LAM1, MU1 = lame(1.10, 0.35); ALPHA1 = 17e-6 / 1e-3      # Cu
LAM2, MU2 = lame(1.30, 0.28); ALPHA2 = 2.6e-6 / 1e-3     # Si
BET1, BET2 = (3 * LAM1 + 2 * MU1) * ALPHA1, (3 * LAM2 + 2 * MU2) * ALPHA2


# ==============================================================
# 1. 解析解
# ==============================================================
def coeffs():
    # 未知 [A, C, D]
    M = np.array([
        [A_R, -A_R, -1 / A_R],                                                   # u_r 連続
        [2 * (LAM1 + MU1), -2 * (LAM2 + MU2), 2 * MU2 / A_R**2],                 # σ_rr 連続
        [0, 2 * (LAM2 + MU2), -2 * MU2 / B_R**2],                                # 外周 σ_rr = 0
    ])
    rhs = np.array([0, (BET1 - BET2) * DT, BET2 * DT])
    return np.linalg.solve(M, rhs)


CA, CC, CD = coeffs()


def exact_u(x, y):
    r2 = x**2 + y**2
    f = np.where(r2 < A_R**2, CA, CC + CD / np.maximum(r2, 1e-30))
    return f * x, f * y


def exact_stress(x, y):
    """(σxx, σyy, σxy) を解析的に"""
    r2 = x**2 + y**2
    inside = r2 < A_R**2
    th = np.arctan2(y, x)
    srr = np.where(inside, 2 * (LAM1 + MU1) * CA - BET1 * DT,
                   2 * (LAM2 + MU2) * CC - 2 * MU2 * CD / np.maximum(r2, 1e-30) - BET2 * DT)
    stt = np.where(inside, srr,
                   2 * (LAM2 + MU2) * CC + 2 * MU2 * CD / np.maximum(r2, 1e-30) - BET2 * DT)
    c, s = np.cos(th), np.sin(th)
    return srr * c**2 + stt * s**2, srr * s**2 + stt * c**2, (srr - stt) * s * c


# ==============================================================
# 2. 特徴量（相ごと）
# ==============================================================
def _z(x, y):
    return jnp.stack([x / B_R, y / B_R])


def make_qerc(N, seed, scale):
    g = np.random.default_rng(1000 + seed)
    W = jnp.asarray(g.normal(size=(2, 2 * N)) * scale); b = jnp.asarray(g.normal(size=2 * N))
    z = (g.normal(size=(2**N,) * 2) + 1j * g.normal(size=(2**N,) * 2)) / np.sqrt(2)
    q, r = np.linalg.qr(z); U = jnp.asarray(q * (np.diag(r) / np.abs(np.diag(r))))   # Haar

    def feat(x, y):
        ang = jnp.pi * (jnp.tanh(_z(x, y) @ W + b) + 1) / 2
        psi = jnp.ones((1,), complex)
        for l in range(N):
            psi = jnp.kron(psi, jnp.stack([jnp.cos(ang[l] / 2), jnp.exp(1j * ang[N + l]) * jnp.sin(ang[l] / 2)]))
        return jnp.abs(U @ psi) ** 2
    return feat


def make_tanh(m, seed, scale):
    g = np.random.default_rng(seed)
    W = jnp.asarray(g.normal(size=(2, m)) * scale); b = jnp.asarray(g.random(m) * 2 * np.pi)
    return lambda x, y: jnp.tanh(_z(x, y) @ W + b)


def make_fourier(m, seed, scale):
    g = np.random.default_rng(seed)
    W = jnp.asarray(g.normal(size=(2, m)) * scale); b = jnp.asarray(g.random(m) * 2 * np.pi)
    return lambda x, y: jnp.cos(_z(x, y) @ W + b)


def make_exact_basis():
    """検証用: 解析解が厳密に入る基底（Cu: x, y / Si: x, y, x/r², y/r²）"""
    return (lambda x, y: jnp.stack([x, y, 1 + 0 * x]),
            lambda x, y: jnp.stack([x, y, x / (x**2 + y**2), y / (x**2 + y**2), 1 + 0 * x]))


def derivs(feat):
    f = lambda p: feat(p[0], p[1])
    return (jax.jit(jax.vmap(f)), jax.jit(jax.vmap(jax.jacfwd(f))),
            jax.jit(jax.vmap(jax.jacfwd(jax.jacfwd(f)))))


# ==============================================================
# 3. 線形系
# ==============================================================
def points(n_cu, n_si, n_if, n_out, seed):
    g = np.random.default_rng(seed)
    r1 = A_R * np.sqrt(g.random(n_cu)); t1 = g.random(n_cu) * 2 * np.pi
    r2 = np.sqrt(A_R**2 + g.random(n_si) * (B_R**2 - A_R**2)); t2 = g.random(n_si) * 2 * np.pi
    ti = np.linspace(0, 2 * np.pi, n_if, endpoint=False); to = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    P = lambda r, t: np.stack([r * np.cos(t), r * np.sin(t)], 1)
    return P(r1, t1), P(r2, t2), P(np.full(n_if, A_R), ti), ti, P(np.full(n_out, B_R), to)


def navier_rows(H, lam, mu):
    Fxx, Fyy, Fxy = H[:, :, 0, 0], H[:, :, 1, 1], H[:, :, 0, 1]
    ex = np.concatenate([(lam + 2 * mu) * Fxx + mu * Fyy, (lam + mu) * Fxy], 1)
    ey = np.concatenate([(lam + mu) * Fxy, (lam + 2 * mu) * Fyy + mu * Fxx], 1)
    return ex, ey


def traction_rows(J, lam, mu, nx, ny):
    Fx, Fy = J[:, :, 0], J[:, :, 1]
    sxx = np.concatenate([(lam + 2 * mu) * Fx, lam * Fy], 1)
    syy = np.concatenate([lam * Fx, (lam + 2 * mu) * Fy], 1)
    sxy = np.concatenate([mu * Fy, mu * Fx], 1)
    return sxx * nx + sxy * ny, sxy * nx + syy * ny


def assemble(d_cu, d_si, pts, w_if=1.0, w_out=1.0):
    p_cu, p_si, p_if, ti, p_out = pts
    B1, J1, H1 = d_cu; B2, J2, H2 = d_si
    m1 = np.asarray(B1(jnp.asarray(p_cu[:1]))).shape[1]; m2 = np.asarray(B2(jnp.asarray(p_si[:1]))).shape[1]
    Z1 = lambda n: np.zeros((n, 2 * m1)); Z2 = lambda n: np.zeros((n, 2 * m2))
    rows, rhs = [], []

    def add(blk1, blk2, b, w):
        rows.append(w * np.concatenate([blk1, blk2], 1)); rhs.append(w * b)

    s = 1 / np.sqrt(len(p_cu))
    for e in navier_rows(np.asarray(H1(jnp.asarray(p_cu))), LAM1, MU1):
        add(e, Z2(len(p_cu)), np.zeros(len(p_cu)), s)
    s = 1 / np.sqrt(len(p_si))
    for e in navier_rows(np.asarray(H2(jnp.asarray(p_si))), LAM2, MU2):
        add(Z1(len(p_si)), e, np.zeros(len(p_si)), s)
    # 界面
    n = len(p_if); s = w_if / np.sqrt(n)
    F1 = np.asarray(B1(jnp.asarray(p_if))); F2 = np.asarray(B2(jnp.asarray(p_if)))
    o1, o2 = np.zeros_like(F1), np.zeros_like(F2)
    add(np.c_[F1, o1], -np.c_[F2, o2], np.zeros(n), s)                 # u 連続
    add(np.c_[o1, F1], -np.c_[o2, F2], np.zeros(n), s)                 # v 連続
    nx, ny = np.cos(ti)[:, None], np.sin(ti)[:, None]
    t1 = traction_rows(np.asarray(J1(jnp.asarray(p_if))), LAM1, MU1, nx, ny)
    t2 = traction_rows(np.asarray(J2(jnp.asarray(p_if))), LAM2, MU2, nx, ny)
    jump = (BET1 - BET2) * DT
    add(t1[0], -t2[0], jump * np.cos(ti), s)
    add(t1[1], -t2[1], jump * np.sin(ti), s)
    # 外周（解析解の変位）
    n = len(p_out); s = w_out / np.sqrt(n)
    F2 = np.asarray(B2(jnp.asarray(p_out))); o2 = np.zeros_like(F2)
    ue, ve = exact_u(p_out[:, 0], p_out[:, 1])
    add(Z1(n), np.c_[F2, o2], ue, s)
    add(Z1(n), np.c_[o2, F2], ve, s)
    return np.concatenate(rows), np.concatenate(rhs), m1


def solve(A, b, ridge):
    sc = np.linalg.norm(A, axis=0) + 1e-30
    As = A / sc; G = As.T @ As; G[np.diag_indices_from(G)] += ridge
    return np.linalg.solve(G, As.T @ b) / sc


def evaluate(d_cu, d_si, w, m1, n=4000, seed=123):
    g = np.random.default_rng(seed)
    r = B_R * np.sqrt(g.random(n)); t = g.random(n) * 2 * np.pi
    x, y = r * np.cos(t), r * np.sin(t)
    inside = x**2 + y**2 < A_R**2
    wc, ws = w[:2 * m1], w[2 * m1:]
    U = np.zeros((n, 2)); S = np.zeros((n, 3))
    for mask, (B, J, _), wp, lam, mu, bet in ((inside, d_cu, wc, LAM1, MU1, BET1),
                                              (~inside, d_si, ws, LAM2, MU2, BET2)):
        p = jnp.asarray(np.stack([x[mask], y[mask]], 1))
        F = np.asarray(B(p)); Jm = np.asarray(J(p)); m = F.shape[1]
        wu, wv = wp[:m], wp[m:]
        U[mask] = np.stack([F @ wu, F @ wv], 1)
        ux, uy = Jm[:, :, 0] @ wu, Jm[:, :, 1] @ wu
        vx, vy = Jm[:, :, 0] @ wv, Jm[:, :, 1] @ wv
        S[mask] = np.stack([(lam + 2 * mu) * ux + lam * vy - bet * DT,
                            lam * ux + (lam + 2 * mu) * vy - bet * DT, mu * (uy + vx)], 1)
    ue, ve = exact_u(x, y); se = np.stack(exact_stress(x, y), 1)
    e_u = np.sqrt(((U[:, 0] - ue)**2 + (U[:, 1] - ve)**2).sum() / (ue**2 + ve**2).sum())
    e_s = np.linalg.norm(S - se) / np.linalg.norm(se)
    # 界面のすぐ外（Si 側）の周方向応力: KOZ の指標
    return e_u, e_s


def run(feat_cu, feat_si, pts, ridge=1e-10):
    d1, d2 = derivs(feat_cu), derivs(feat_si)
    A, b, m1 = assemble(d1, d2, pts)
    return evaluate(d1, d2, solve(A, b, ridge), m1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    print(f"解析解: A={CA:.4f}, C={CC:.4f}, D={CD:.4f}")
    pts = points(2000, 6000, 400, 400, 0)
    eu, es = run(*make_exact_basis(), pts)
    print(f"検証（解析解が入る基底）: u={eu:.1e}, σ={es:.1e}")
    m = 2**a.N
    grid = {
        "QERC-Haar": ([1.0, 2.0, 4.0, 8.0], lambda sd, s: make_qerc(a.N, sd, s)),
        "tanh-RF": ([1.0, 2.0, 4.0, 8.0, 16.0], lambda sd, s: make_tanh(m, sd, s)),
        "Fourier": ([1.0, 2.0, 4.0, 8.0, 16.0], lambda sd, s: make_fourier(m, sd, s)),
    }
    rows = []
    for sd in range(a.seeds):
        pts = points(2000, 6000, 400, 400, 10 + sd)
        for name, (scales, ctor) in grid.items():
            for s in scales:
                t0 = time.time()
                eu, es = run(ctor(2 * sd, s), ctor(2 * sd + 1, s), pts)
                rows.append({"model": name, "seed": sd, "scale": s, "u": eu, "s": es})
                print(f"seed={sd} {name:10s} ×{s:<4} u={eu:.2e} σ={es:.2e} {time.time()-t0:.0f}s", flush=True)
        if a.out:
            json.dump(rows, open(a.out, "w"), indent=1)
    print("\n=== シードごとに最良の倍率 ===")
    for name in grid:
        v = [min(r["u"] for r in rows if r["model"] == name and r["seed"] == sd) for sd in range(a.seeds)]
        print(f"{name:10s} u: 幾何平均 {np.exp(np.mean(np.log(v))):.2e}  {['%.1e' % x for x in v]}")


if __name__ == "__main__":
    main()
