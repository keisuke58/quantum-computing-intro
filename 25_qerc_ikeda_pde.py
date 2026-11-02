# -*- coding: utf-8 -*-
"""
25 - Ikeda et al. (2026) 型の QERC エンコーダで Kirsch を PDE のみで解く

Ikeda, Sakurai, Nemoto, Muramatsu, "Quantum Extreme Reservoir Computing for Phase Classification
of Polymer Alloy Microstructures" (arXiv:2601.02150) の構成:
  PCA で 2N 次元に圧縮 → qubit l に θ_l = x_l, φ_l = x_{N+l}（積状態）→ Clifford+T リザバー
  → 全基底確率 2^N 個 → 線形分類器

22 では全 qubit に同じ 2 入力を入れたため，ρ(x) の張る関数空間が 2^N より小さく，
リザバーを変えても結果が変わらなかった（N=8 でランク 152/256）。ここでは Ikeda らと同じく
qubit ごとに別の入力を入れる。PDE の入力は (ρ, θ) の 2 つしかないので，PCA の代わりに
(arccos ξ, cos θ, sin θ) のランダムなアフィン結合で 2N 個の入力を作る（周期性は cos/sin で保つ）。

仮説: 「エンコーディングの張る空間が 2^N を超えるときだけ，リザバーが効く」
リザバー: none（恒等），clifford（ランダム Clifford のみ），cliffordT（Clifford + H·T·H 層），
          ising（横磁場イジング），haar（ランダムユニタリ）
"""
import argparse, importlib.util, os, time
import numpy as np
import jax, jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
_s = importlib.util.spec_from_file_location(
    "m", os.path.join(os.path.dirname(os.path.abspath(__file__)), "22_qerc_pinn_kirsch.py"))
M = importlib.util.module_from_spec(_s); _s.loader.exec_module(M)

H1 = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
S1 = np.diag([1, 1j]); T1 = np.diag([1, np.exp(1j * np.pi / 4)])


def _op1(g, i, N):
    m = np.array([[1.]])
    for k in range(N):
        m = np.kron(m, g if k == i else np.eye(2))
    return m


def _cnot(c, t, N):
    d = 2**N; U = np.zeros((d, d))
    for b in range(d):
        bits = [(b >> (N - 1 - k)) & 1 for k in range(N)]
        if bits[c]:
            bits[t] ^= 1
        U[sum(v << (N - 1 - k) for k, v in enumerate(bits)), b] = 1
    return U


def random_clifford(N, rng, depth=None):
    """{H, S, CNOT} のランダム回路（深さ 20N）で Clifford 群を近似的に一様にサンプル"""
    U = np.eye(2**N, dtype=complex)
    for _ in range(depth or 20 * N):
        r = rng.integers(3)
        if r == 0:
            U = _op1(H1, rng.integers(N), N) @ U
        elif r == 1:
            U = _op1(S1, rng.integers(N), N) @ U
        else:
            c, t = rng.choice(N, 2, replace=False)
            U = _cnot(c, t, N) @ U
    return U


def reservoir(kind, N, seed):
    rng = np.random.default_rng(seed)
    if kind == "none":
        return np.eye(2**N)
    if kind == "ising":
        return M.Q.ising_U(N, seed)
    if kind == "haar":
        z = (rng.normal(size=(2**N,) * 2) + 1j * rng.normal(size=(2**N,) * 2)) / np.sqrt(2)
        q, r = np.linalg.qr(z); return q * (np.diag(r) / np.abs(np.diag(r)))
    U = random_clifford(N, rng)
    if kind == "cliffordT":
        HTH = H1 @ T1 @ H1
        for i in range(N):
            U = _op1(HTH, i, N) @ U
    return U


def make_ikeda(N, seed, kind):
    g = np.random.default_rng(1000 + seed)
    W = jnp.asarray(g.normal(size=(3, 2 * N))); b = jnp.asarray(g.normal(size=2 * N))
    U = jnp.asarray(reservoir(kind, N, seed))

    def feat(x, y):
        _, th, a1 = M._pre(x, y)
        z = jnp.stack([a1 - 1.57, jnp.cos(th), jnp.sin(th)]) @ W + b
        ang = jnp.pi * (jnp.tanh(z) + 1) / 2          # 0〜π（Ikeda らと同じ範囲）
        psi = jnp.ones((1,), complex)
        for l in range(N):
            tl, pl = ang[l], ang[N + l]
            psi = jnp.kron(psi, jnp.stack([jnp.cos(tl / 2), jnp.exp(1j * pl) * jnp.sin(tl / 2)]))
        return jnp.abs(U @ psi) ** 2
    return feat, 2**N


def rank(feat, n=4000):
    g = np.random.default_rng(0)
    r = np.sqrt(1 + g.random(n) * 15); th = g.random(n) * 2 * np.pi
    F = np.asarray(jax.vmap(feat)(jnp.asarray(r * np.cos(th)), jnp.asarray(r * np.sin(th))))
    sv = np.linalg.svd(F, compute_uv=False)
    return int((sv > sv[0] * 1e-10).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", default="6,8")
    ap.add_argument("--kinds", default="none,clifford,cliffordT,ising,haar")
    ap.add_argument("--seeds", type=int, default=2)
    a = ap.parse_args()
    for N in map(int, a.N.split(",")):
        for sd in range(a.seeds):
            pts = M.points(8000, 800, 100 + sd)
            for kind in a.kinds.split(","):
                feat, m = make_ikeda(N, sd, kind)
                t0 = time.time(); rk = rank(feat)
                fns = M.make_basis(feat)
                A, b = M.assemble(fns, *pts, 1.0)
                e_u, e_s, kt = M.evaluate(fns, M.solve(A, b, 1e-10))
                print(f"N={N} seed={sd} {kind:9s} rank={rk:4d}/{m}  u={e_u:.2e}  σ={e_s:.2e}  "
                      f"K_t={kt:.3f}  {time.time()-t0:.0f}s", flush=True)
            if sd == 0 or True:
                feat, m = M.make_crf(2**N, sd); fns = M.make_basis(feat)
                A, b = M.assemble(fns, *pts, 1.0)
                e_u, e_s, kt = M.evaluate(fns, M.solve(A, b, 1e-10))
                print(f"N={N} seed={sd} {'古典RF':9s} rank={rank(feat):4d}/{m}  u={e_u:.2e}  "
                      f"σ={e_s:.2e}  K_t={kt:.3f}", flush=True)


if __name__ == "__main__":
    main()
