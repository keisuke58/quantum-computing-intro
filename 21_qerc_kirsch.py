# -*- coding: utf-8 -*-
"""
21 - QERC（量子エクストリームリザバー計算）型の特徴量で Kirsch 解を表せるか（教師ありフィット）

QERC（Sakurai et al., PR Applied 2022 系）の構成:
  1. 入力を各 qubit の 1 量子ビット回転で符号化（積状態）
  2. 固定したランダム結合の横磁場イジング H = Σ J_ij X_i X_j + h Σ Z_i で時間発展 U = exp(-iHt)
  3. 計算基底の全確率 p_k = |<k|U|ψ(x)>|²（2^N 個）を特徴量とし，線形の読み出しだけを最小二乗で解く

N qubit で 2^N 個の特徴量が得られるのが要点。20_qelm_probe_kirsch.py（期待値 O(N²) 個）では
古典ランダム特徴量に負けたので，特徴量の数を指数的に増やす設計で比べ直す。

前処理は 19 と同じ（θ と ρ=a/r の Chebyshev 角，出力に r を掛けて極座標成分として読む）。
比較: 同数の古典ランダム特徴量（tanh），および「同じ N」の古典リザバー（N 個の tanh）。
"""
import argparse, importlib.util, os, time
import numpy as np
from scipy.linalg import expm

_s = importlib.util.spec_from_file_location(
    "k", os.path.join(os.path.dirname(os.path.abspath(__file__)), "13_qpinn_kirsch.py"))
K = importlib.util.module_from_spec(_s); _s.loader.exec_module(K)
A, R = K.A_HOLE, K.R_OUT


def data(n, seed):
    g = np.random.default_rng(seed)
    r = np.sqrt(A**2 + g.random(n) * (R**2 - A**2)); th = g.random(n) * 2 * np.pi
    u, v = K.uv_exact(r * np.cos(th), r * np.sin(th))
    xi = 0.95 * (2 * (A / r - A / R) / (1 - A / R) - 1)
    return r, th, np.arccos(xi), u, v


def ising_U(N, seed, t=1.0, h=1.0):
    g = np.random.default_rng(seed)
    X = np.array([[0, 1], [1, 0]]); Z = np.diag([1., -1.]); I = np.eye(2)
    def op(single, i):
        m = np.array([[1.]])
        for k in range(N):
            m = np.kron(m, single if k == i else I)
        return m
    Xs = [op(X, i) for i in range(N)]
    H = sum(h * op(Z, i) for i in range(N))
    for i in range(N):
        for j in range(i + 1, N):
            H = H + g.uniform(-1, 1) * Xs[i] @ Xs[j]
    return expm(-1j * H * t)


def qerc_feats(a1, th, N, U):
    """qubit i に RY(a1)·RZ(θ) を入れた積状態（偶数番は a1 主体，奇数番は θ 主体に混ぜる）"""
    n = len(a1); psi = np.ones((n, 1), complex)
    for i in range(N):
        ay, az = (a1, th) if i % 2 == 0 else (th, a1)
        q = np.stack([np.cos(ay / 2) * np.exp(-0.5j * az), np.sin(ay / 2) * np.exp(0.5j * az)], 1)
        psi = (psi[:, :, None] * q[:, None, :]).reshape(n, -1)
    return np.abs(psi @ U.T) ** 2


def crf_feats(a1, th, m, seed):
    g = np.random.default_rng(seed)
    X = np.stack([a1, np.cos(th), np.sin(th)], 1)
    return np.tanh(X @ (g.normal(size=(3, m)) * 1.5) + g.random(m) * 2 * np.pi)


def fit_err(Ftr, Fte, tr, te, ridge=1e-10):
    r, th, _, u, v = tr; rt, tht, _, ut, vt = te
    ur = u * np.cos(th) + v * np.sin(th); uq = -u * np.sin(th) + v * np.cos(th)
    Ftr = np.c_[Ftr, np.ones(len(r))]; Fte = np.c_[Fte, np.ones(len(rt))]
    G = Ftr.T @ Ftr + ridge * np.eye(Ftr.shape[1])
    pr = Fte @ np.linalg.solve(G, Ftr.T @ (ur / r)) * rt
    pq = Fte @ np.linalg.solve(G, Ftr.T @ (uq / r)) * rt
    pu = pr * np.cos(tht) - pq * np.sin(tht); pv = pr * np.sin(tht) + pq * np.cos(tht)
    return np.sqrt(((pu - ut)**2 + (pv - vt)**2).sum() / (ut**2 + vt**2).sum())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--N", default="4,6,8,10")
    ap.add_argument("--seeds", type=int, default=2); a = ap.parse_args()
    tr, te = data(3000, 0), data(2000, 1)
    print(f"{'N':>3} {'特徴量':>6} | {'QERC':>9} | {'古典RF(同数)':>12} | {'古典RF(N個)':>11}")
    for N in map(int, a.N.split(",")):
        for sd in range(a.seeds):
            t0 = time.time(); U = ising_U(N, sd)
            eq = fit_err(qerc_feats(tr[2], tr[1], N, U), qerc_feats(te[2], te[1], N, U), tr, te)
            m = 2**N
            ec = fit_err(crf_feats(tr[2], tr[1], m, sd), crf_feats(te[2], te[1], m, sd), tr, te)
            en = fit_err(crf_feats(tr[2], tr[1], N, sd), crf_feats(te[2], te[1], N, sd), tr, te)
            print(f"{N:>3} {m:>6} | {eq:9.2e} | {ec:12.2e} | {en:11.2e}  seed={sd} {time.time()-t0:.0f}s",
                  flush=True)


if __name__ == "__main__":
    main()
