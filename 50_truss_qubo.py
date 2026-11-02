# -*- coding: utf-8 -*-
"""
50 - トラスの変形解析を QUBO にして解く（本田ほか, Sci. Rep. 2024 の定式化の最小の再現）

全ポテンシャルエネルギー Π(u) = ½ uᵀ K u − Fᵀ u を，変位を 0/1 の変数で表して最小化する．
  変位 1 成分を N ビットで表す: u = a (2 Σ ε_α q_α / Σ ε_α − 1)，ε_α は一様乱数（論文の符号化）
  → Π は q の 2 次式（QUBO）
解き方を 4 通りで比べる:
  exact  : K u = F を直接解く（参照）
  brute  : 2^n 通りを全部調べた QUBO の最適解（符号化による誤差だけが残る）
  SA     : 古典のシミュレーテッドアニーリング
  QAOA   : JAX の状態ベクトルで QAOA（p 層），γ, β を古典の最適化で調整．最も確率の高いビット列を答えにする

例 1（QAOA まで）: 2 本の部材で吊った 1 節点（自由度 2，各 6 ビット → 12 量子ビット）
例 2（SA のみ）  : 小さい平面トラス（自由度 6，各 10 ビット → 60 ビット）
"""
import itertools, json, time
import numpy as np
import jax, jax.numpy as jnp
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)


# ==============================================================
# 1. トラスと QUBO
# ==============================================================
def stiffness(nodes, elems, EA=1.0):
    n = len(nodes); K = np.zeros((2 * n, 2 * n))
    for i, j in elems:
        d = nodes[j] - nodes[i]; L = np.linalg.norm(d); c = d / L
        k = EA / L * np.outer(np.r_[-c, c], np.r_[-c, c])
        idx = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
        K[np.ix_(idx, idx)] += k
    return K


def qubo(K, F, nbits, a, seed=0):
    """Π(u) を q（0/1）の 2 次式 c + Σ h q + Σ J q q にする．u = c0 + B q"""
    g = np.random.default_rng(seed)
    m = len(F); n = m * nbits
    B = np.zeros((m, n)); c0 = -a * np.ones(m)
    for d in range(m):
        eps = g.uniform(0.1, 1.0, nbits)
        B[d, d * nbits:(d + 1) * nbits] = 2 * a * eps / eps.sum()
    Q = 0.5 * B.T @ K @ B                                   # 2 次の項
    h = B.T @ (K @ c0) - B.T @ F                            # 1 次の項
    const = 0.5 * c0 @ K @ c0 - F @ c0
    Q = Q.copy(); h = h + np.diag(Q); np.fill_diagonal(Q, 0.0)   # q² = q を 1 次へ
    J = Q + Q.T                                             # 上三角と下三角をまとめて q_i q_j（i<j）の係数に
    return {"B": B, "c0": c0, "h": h, "J": np.triu(J, 1), "const": const}


def energy(P, q):
    q = np.asarray(q, float)
    return P["const"] + P["h"] @ q + q @ P["J"] @ q


def decode(P, q):
    return P["c0"] + P["B"] @ np.asarray(q, float)


# ==============================================================
# 2. 解き方
# ==============================================================
def brute(P):
    n = len(P["h"])
    Z = np.array(list(itertools.product([0, 1], repeat=n)), float)     # 状態の添字 = ビット列（先頭が最上位）
    E = P["const"] + Z @ P["h"] + np.einsum("si,ij,sj->s", Z, P["J"], Z)
    return Z, E


def sa(P, sweeps=4000, seed=0):
    g = np.random.default_rng(seed); n = len(P["h"])
    Jsym = P["J"] + P["J"].T
    q = g.integers(0, 2, n).astype(float)
    T0 = max(np.abs(P["h"]).max(), np.abs(P["J"]).max(), 1e-30)     # 温度はエネルギーの大きさに合わせる
    for T in np.geomspace(T0, 1e-6 * T0, sweeps):
        for i in g.permutation(n):
            dE = (1 - 2 * q[i]) * (P["h"][i] + Jsym[i] @ q)
            if dE < 0 or g.random() < np.exp(-dE / T):
                q[i] = 1 - q[i]
    return q


def qaoa(P, p, ue=None, restarts=8, seed=0):
    """コスト対角 C(z) と X ミキサー．状態ベクトルは (2,)*n のテンソルで持つ"""
    Z, E = brute(P)
    n = len(P["h"]); Ec = jnp.asarray(E - E.min())               # 定数をずらしても最適化は同じ
    scale = float(np.std(E)) or 1.0

    def state(theta):
        gam, bet = theta[:p], theta[p:]
        psi = jnp.ones(2**n, complex) / np.sqrt(2**n)
        for k in range(p):
            psi = psi * jnp.exp(-1j * gam[k] * Ec / scale)
            t = psi.reshape((2,) * n)
            cb, sb = jnp.cos(bet[k]), -1j * jnp.sin(bet[k])
            for ax in range(n):                                   # exp(-iβX) を各量子ビットに
                t0, t1 = jnp.take(t, 0, axis=ax), jnp.take(t, 1, axis=ax)
                t = jnp.stack([cb * t0 + sb * t1, sb * t0 + cb * t1], axis=ax)
            psi = t.reshape(-1)
        return psi

    cost = jax.jit(lambda th: jnp.real(jnp.vdot(state(th), Ec * state(th))))
    grad = jax.jit(jax.grad(cost))
    g = np.random.default_rng(seed); best = None
    for _ in range(restarts):
        th0 = np.r_[g.uniform(0, 1.0, p), g.uniform(0, np.pi / 2, p)]
        r = minimize(lambda t: float(cost(jnp.asarray(t))), th0, jac=lambda t: np.asarray(grad(jnp.asarray(t))),
                     method="L-BFGS-B")
        if best is None or r.fun < best.fun:
            best = r
    prob = np.abs(np.asarray(state(jnp.asarray(best.x)))) ** 2
    k_opt = int(np.argmin(E))
    good = None
    if ue is not None:                                           # 誤差 5% 以内に入るビット列の確率の合計
        U = P["c0"][None, :] + Z @ P["B"].T
        good = float(prob[np.linalg.norm(U - ue, axis=1) / np.linalg.norm(ue) < 0.05].sum())
    return {"q": Z[int(np.argmax(prob))], "p_opt": float(prob[k_opt]), "p_top": float(prob.max()), "p_good": good,
            "exp_E": float(best.fun) + E.min(), "E_opt": float(E.min())}


# ==============================================================
# 3. 例
# ==============================================================
def relerr(u, ue):
    return float(np.linalg.norm(u - ue) / np.linalg.norm(ue))


def example1(nbits=6, pmax=4):
    print("例 1: 2 本の部材で吊った 1 節点（自由度 2）")
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, -0.5]]); elems = [(0, 2), (1, 2)]
    K = stiffness(nodes, elems)[4:, 4:]; F = np.array([0.3, -1.0]) * 0.05
    ue = np.linalg.solve(K, F); a = 1.5 * np.abs(ue).max()
    P = qubo(K, F, nbits, a)
    Z, E = brute(P); qb = Z[np.argmin(E)]
    rows = {"exact_u": ue.tolist(), "nbits": nbits, "nqubits": len(P["h"])}
    rows["brute_err"] = relerr(decode(P, qb), ue)
    rows["sa_err"] = relerr(decode(P, sa(P)), ue)
    print(f"  厳密解 u = {ue}，量子ビット {len(P['h'])}")
    print(f"  全探索（符号化の誤差だけ）: 相対誤差 {rows['brute_err']:.2e}")
    U = P["c0"][None, :] + Z @ P["B"].T
    rows["uniform_good"] = float(np.mean(np.linalg.norm(U - ue, axis=1) / np.linalg.norm(ue) < 0.05))
    print(f"  参考: ランダムに選んだビット列が誤差 5% 以内に入る確率 {rows['uniform_good']:.3f}")
    print(f"  SA                         : 相対誤差 {rows['sa_err']:.2e}")
    rows["qaoa"] = []
    for p in range(1, pmax + 1):
        t0 = time.time(); r = qaoa(P, p, ue)
        e = relerr(decode(P, r["q"]), ue)
        rows["qaoa"].append({"p": p, "err": e, "p_opt": r["p_opt"], "p_top": r["p_top"], "p_good": r["p_good"]})
        print(f"  QAOA p = {p}: 最頻のビット列の誤差 {e:.2e}，最適解の確率 {r['p_opt']:.4f}，誤差 5% 以内の確率 {r['p_good']:.3f}"
              f" {time.time()-t0:.0f}s", flush=True)
    return rows


def example2(nbits=10):
    print("例 2: 平面トラス（下 3 節点・上 2 節点，左の 2 節点を固定，右下に荷重）")
    nodes = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1]], float)
    elems = [(0, 1), (1, 2), (3, 4), (0, 3), (1, 4), (0, 4), (1, 3), (2, 4)]
    K = stiffness(nodes, elems)
    nodes_free = [1, 2, 4]                                      # 節点 0, 3 を固定（左の壁）
    free = [2 * i + d for i in nodes_free for d in (0, 1)]
    K = K[np.ix_(free, free)]; F = np.zeros(len(free)); F[3] = -0.01   # 節点 2 に下向きの荷重
    ue = np.linalg.solve(K, F); a = 1.2 * np.abs(ue).max()
    P = qubo(K, F, nbits, a)
    errs = [relerr(decode(P, sa(P, sweeps=6000, seed=s)), ue) for s in range(5)]
    print(f"  自由度 {len(F)}，ビット {len(P['h'])}．SA（5 回）の相対誤差: " + ", ".join(f"{e:.1e}" for e in errs))
    return {"ndof": len(F), "nbits_total": len(P["h"]), "sa_err": errs}


if __name__ == "__main__":
    out = {"ex1": example1(), "ex2": example2()}
    json.dump(out, open("results/truss_qubo.json", "w"), indent=1)
