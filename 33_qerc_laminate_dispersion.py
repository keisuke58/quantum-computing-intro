# -*- coding: utf-8 -*-
"""
33 - 高次元パラメトリック問題: 積層材料の弾性波の分散関係を量子特徴量 vs 古典特徴量で学習する

d 層の積層（材料 A, B を交互）を 1 周期とする周期構造の縦波．層 i の厚さ h_i を入力（d 次元）とし，
Bloch の分散関係
    cos(q L) = ½ tr( M_d ⋯ M_2 M_1 ),   M_i = [[cos φ_i, sin φ_i / Z_i], [−Z_i sin φ_i, cos φ_i]]
を出力とする（φ_i = ω h_i / c_i，Z_i はインピーダンス）．|½ tr| ≤ 1 なら通過帯，> 1 ならバンドギャップ．

½ tr(ΠM_i) は各層の (cos φ_i, sin φ_i) の多重線形式（2^d 項）で，入力の次元 d が増えると古典の
ランダム特徴量には難しくなる．一方，qubit i に角度 φ_i を入れた積状態の測定確率は，
各 qubit の (1, cos φ_i, sin φ_i …) のテンソル積の線形結合なので，構造が一致する．

比較（特徴量の数はすべて 2^N，N = d，最小二乗 1 回，教師あり）:
  QERC        : qubit i に RY(φ_i)（+ 位相 RZ(φ_i)），Haar リザバー，全基底確率
  tensor-RP   : 同じ積状態からの古典ランダム射影（量子と同じ関数空間）
  cos-RF      : cos(w·φ + b)（Fourier 型ランダム特徴量），w の倍率を調整
  tanh-RF     : tanh(w·z + b)，倍率を調整
"""
import argparse, json, time
import numpy as np

Z_A, Z_B = 1.0, 3.0


def target(phi):
    """phi: (n, d) → ½ tr(Π M_i)"""
    n, d = phi.shape
    M = np.tile(np.eye(2), (n, 1, 1))
    for i in range(d):
        Z = Z_A if i % 2 == 0 else Z_B
        c, s = np.cos(phi[:, i]), np.sin(phi[:, i])
        Mi = np.stack([np.stack([c, s / Z], -1), np.stack([-Z * s, c], -1)], -2)
        M = Mi @ M
    return 0.5 * (M[:, 0, 0] + M[:, 1, 1])


def sample(n, d, seed):
    g = np.random.default_rng(seed)
    x = g.random((n, d))
    return x, np.pi * (0.5 + x)          # φ_i ∈ [π/2, 3π/2]


def product_state(phi):
    n, d = phi.shape
    psi = np.ones((n, 1), complex)
    for i in range(d):
        q = np.stack([np.cos(phi[:, i] / 2), np.exp(1j * phi[:, i]) * np.sin(phi[:, i] / 2)], 1)
        psi = (psi[:, :, None] * q[:, None, :]).reshape(n, -1)
    return psi


def haar(D, seed):
    g = np.random.default_rng(seed)
    z = (g.normal(size=(D, D)) + 1j * g.normal(size=(D, D))) / np.sqrt(2)
    q, r = np.linalg.qr(z); return q * (np.diag(r) / np.abs(np.diag(r)))


def feats_qerc(phi, U):
    return np.abs(product_state(phi) @ U.T) ** 2


def feats_tensor_rp(phi, R):
    psi = product_state(phi)                       # (n, D)
    return np.real(np.einsum("ni,kij,nj->nk", np.conj(psi), R, psi, optimize=True))


def feats_cos(phi, W, b):
    return np.cos(phi @ W + b)


def feats_tanh(x, W, b):
    return np.tanh((x - 0.5) @ W + b)


def fit_err(Ftr, ytr, Fte, yte, ridge=1e-10):
    Ftr = np.c_[Ftr, np.ones(len(Ftr))]; Fte = np.c_[Fte, np.ones(len(Fte))]
    sc = np.linalg.norm(Ftr, axis=0) + 1e-30
    A = Ftr / sc; G = A.T @ A; G[np.diag_indices_from(G)] += ridge * len(A)
    w = np.linalg.solve(G, A.T @ ytr) / sc
    return np.linalg.norm(Fte @ w - yte) / np.linalg.norm(yte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ns", default="6,8,10,12")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--n_train_factor", type=float, default=4.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = []
    for N in map(int, a.Ns.split(",")):
        D = 2**N; ntr = int(a.n_train_factor * D); nte = 4000
        for sd in range(a.seeds):
            xtr, ptr = sample(ntr, N, 10 + sd); xte, pte = sample(nte, N, 99)
            ytr, yte = target(ptr), target(pte)
            res = {}
            t0 = time.time()
            U = haar(D, sd)
            res["QERC"] = fit_err(feats_qerc(ptr, U), ytr, feats_qerc(pte, U), yte)
            tq = time.time() - t0
            if N <= 8:
                g = np.random.default_rng(500 + sd)
                Zr = (g.normal(size=(D, D, D)) + 1j * g.normal(size=(D, D, D))) / np.sqrt(2 * D)
                R = (Zr + np.conj(np.transpose(Zr, (0, 2, 1)))) / 2
                res["tensor-RP"] = fit_err(feats_tensor_rp(ptr, R), ytr, feats_tensor_rp(pte, R), yte)
                del Zr, R
            best = {}
            for name, fn, inp_tr, inp_te, scales in (
                    ("cos-RF", feats_cos, ptr, pte, [0.25, 0.5, 1.0, 2.0]),
                    ("tanh-RF", feats_tanh, xtr, xte, [0.5, 1.0, 2.0, 4.0])):
                for s in scales:
                    g = np.random.default_rng(700 + sd)
                    W = g.normal(size=(N, D)) * s; b = g.random(D) * 2 * np.pi
                    e = fit_err(fn(inp_tr, W, b), ytr, fn(inp_te, W, b), yte)
                    best[name] = min(best.get(name, np.inf), e)
            res.update(best)
            res.update({"N": N, "seed": sd, "n_train": ntr, "t_qerc": tq})
            rows.append(res)
            print(f"N={N:<2} seed={sd} " + "  ".join(f"{k}={res[k]:.2e}" for k in
                  ("QERC", "tensor-RP", "cos-RF", "tanh-RF") if k in res) + f"  (QERC {tq:.0f}s)", flush=True)
            if a.out:
                json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
