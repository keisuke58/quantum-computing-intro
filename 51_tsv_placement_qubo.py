# -*- coding: utf-8 -*-
"""
51 - TSV の配置最適化を QUBO にして解く（アイデア C の試作）

問題: 格子状に並んだ候補位置から K 本の TSV を選んで置く．トランジスタの位置（評価点）での
      キャリア移動度の変化 Δμ/μ の 2 乗和を最小にする（キープアウトゾーンの考え方）．
応力: TSV 1 本の応力場（平面ひずみ，無限の Si 中の Cu 円柱，ΔT = −250 K）の線形重ね合わせ
      （Jung et al. DAC 2011 と同じ近似．ピッチが小さいと誤差が出る: MORE-Stress では 5〜14%）
      Si 中: σ_rr = −S (a/r)²，σ_θθ = +S (a/r)²（S は界面の圧力）．σ_xx, σ_yy に直して足す
移動度: Δμ/μ = −(π_l σ_xx + π_t σ_yy)（x 方向のチャネル．n 型 Si <110> の代表値 π_l = −31.6，π_t = −17.6 ×1e-11 /Pa）
QUBO:  x_i ∈ {0, 1}（候補 i に置くか）．Δμ/μ(p) = Σ_i x_i m_i(p) なので
       目的 = Σ_p (Σ_i x_i m_i(p))² = Σ_ij x_i x_j G_ij，G = Mᵀ M．本数の制約は λ (Σ x_i − K)² で足す
       隣り合う候補（間隔 < s_min）に同時に置かない設計ルールもペナルティで入れられる
解き方: 全探索（C(n, K) 通り，参照），貪欲法，SA，QAOA（JAX の状態ベクトル，小さい例だけ）
"""
import itertools, json, time
import numpy as np
import jax, jax.numpy as jnp
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

# 材料（30 と同じ）: 応力の単位 MPa
E1, NU1, A1 = 110e3, 0.35, 17e-6      # Cu
E2, NU2, A2 = 130e3, 0.28, 2.6e-6     # Si
DT, RAD = -250.0, 2.5                  # ΔT [K]，ビア半径 [µm]
PI_L, PI_T = -31.6e-5, -17.6e-5        # [1/MPa]（= ×1e-11 /Pa）


def lame(E, nu):
    return E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))


def interface_pressure():
    """無限の Si 中の Cu 円柱（平面ひずみ）．Si: u = C r + D/r（遠方で応力 0），Cu: u = A r"""
    (l1, m1), (l2, m2) = lame(E1, NU1), lame(E2, NU2)
    b1, b2 = (3 * l1 + 2 * m1) * A1, (3 * l2 + 2 * m2) * A2
    C = b2 * DT / (2 * (l2 + m2))                       # 遠方の σ_rr = 0（Si の自由な熱収縮）
    # A a = C a + D/a，2(l1+m1)A − b1ΔT = 2(l2+m2)C − 2 m2 D/a² − b2ΔT
    M = np.array([[RAD, -1 / RAD], [2 * (l1 + m1), 2 * m2 / RAD**2]])
    rhs = np.array([C * RAD, 2 * (l2 + m2) * C - b2 * DT + b1 * DT])
    Acu, D = np.linalg.solve(M, rhs)
    return 2 * m2 * D / RAD**2                          # σ_rr(a) = −S（S > 0 なら圧縮）


S0 = interface_pressure()


def single_field(dx, dy):
    """TSV（原点）の Si 中の (σ_xx, σ_yy) [MPa]．r < a は Si でないので 0（評価点は Si に置く）"""
    r2 = dx**2 + dy**2
    f = np.where(r2 > RAD**2, S0 * RAD**2 / np.maximum(r2, 1e-12), 0.0)
    c2 = (dx**2 - dy**2) / np.maximum(r2, 1e-12)           # cos 2θ
    # σ_rr = −f，σ_θθ = +f → σ_xx = −f cos2θ，σ_yy = +f cos2θ
    return -f * c2, f * c2


def build(nx, ny, site_pitch, K, lam=None, s_min=None, dev_step=None):
    sites = np.array([[i * site_pitch, j * site_pitch] for j in range(ny) for i in range(nx)], float)
    dev_step = dev_step or site_pitch / 2
    gx = np.arange(-site_pitch / 2, (nx - 0.5) * site_pitch + 1e-9, dev_step)
    gy = np.arange(-site_pitch / 2, (ny - 0.5) * site_pitch + 1e-9, dev_step)
    P = np.array([[x, y] for y in gy for x in gx])
    dmin = np.min(np.linalg.norm(P[:, None] - sites[None], axis=2), axis=1)
    P = P[dmin > RAD + 1.0]                                 # トランジスタは TSV の候補から 1 µm 以上離す
    Mm = np.zeros((len(P), len(sites)))
    for i, s in enumerate(sites):
        sxx, syy = single_field(P[:, 0] - s[0], P[:, 1] - s[1])
        Mm[:, i] = -(PI_L * sxx + PI_T * syy)               # Δμ/μ
    G = Mm.T @ Mm
    n = len(sites)
    lam = lam or 2.0 * np.abs(G).max() * n
    Q = G.copy()
    # λ(Σx − K)² = λ(Σ_i x_i + 2Σ_{i<j} x_i x_j − 2K Σ x_i + K²)
    Q += lam * (np.ones((n, n)) - np.eye(n))                 # 2Σ_{i<j} を対称な Q の非対角に
    Q[np.diag_indices(n)] += lam * (1 - 2 * K)
    if s_min:
        D = np.linalg.norm(sites[:, None] - sites[None], axis=2)
        pen = (D < s_min) & (D > 0)
        Q[pen] += lam
    return {"sites": sites, "P": P, "M": Mm, "G": G, "Q": Q, "const": lam * K**2, "K": K, "n": n,
            "s_min": s_min}


def obj(prob, x):
    x = np.asarray(x, float)
    return float(x @ prob["G"] @ x)                          # 制約を満たす配置での本来の目的


def feasible(prob, x):
    x = np.asarray(x)
    if x.sum() != prob["K"]:
        return False
    if prob["s_min"]:
        idx = np.where(x)[0]
        for i, j in itertools.combinations(idx, 2):
            if np.linalg.norm(prob["sites"][i] - prob["sites"][j]) < prob["s_min"]:
                return False
    return True


def exhaustive(prob):
    best = (np.inf, None)
    for comb in itertools.combinations(range(prob["n"]), prob["K"]):
        x = np.zeros(prob["n"]); x[list(comb)] = 1
        if not feasible(prob, x):
            continue
        v = obj(prob, x)
        if v < best[0]:
            best = (v, x)
    return best


def greedy(prob):
    x = np.zeros(prob["n"])
    for _ in range(prob["K"]):
        best_v, best_i = np.inf, None
        for i in range(prob["n"]):
            if x[i]:
                continue
            y = x.copy(); y[i] = 1
            ok = True
            if prob["s_min"]:
                for j in np.where(x)[0]:
                    if np.linalg.norm(prob["sites"][i] - prob["sites"][j]) < prob["s_min"]:
                        ok = False
            if ok and obj(prob, y) < best_v:
                best_v, best_i = obj(prob, y), i
        if best_i is None:
            return None
        x[best_i] = 1
    return x


def sa(prob, sweeps=3000, seed=0):
    """本数を保つ入れ替え（1 本を空いた候補へ移す）の SA．設計ルールはペナルティで"""
    g = np.random.default_rng(seed); n, K = prob["n"], prob["K"]
    Q = prob["Q"]
    x = np.zeros(n); x[g.choice(n, K, replace=False)] = 1
    E = lambda y: y @ Q @ y
    cur = E(x); best = (cur, x.copy())
    T0 = max(np.abs(prob["G"]).max(), 1e-30)
    for T in np.geomspace(T0, 1e-6 * T0, sweeps):
        i = g.choice(np.where(x == 1)[0]); j = g.choice(np.where(x == 0)[0])
        y = x.copy(); y[i], y[j] = 0, 1
        e = E(y)
        if e < cur or g.random() < np.exp(-(e - cur) / T):
            x, cur = y, e
            if cur < best[0]:
                best = (cur, x.copy())
    return best[1]


def qaoa(prob, p, restarts=6, seed=0):
    n = prob["n"]
    Z = np.array(list(itertools.product([0, 1], repeat=n)), float)
    E = np.einsum("si,ij,sj->s", Z, prob["Q"], Z) + prob["const"]
    Ec = jnp.asarray(E - E.min()); scale = float(np.std(E))

    def state(th):
        gam, bet = th[:p], th[p:]
        psi = jnp.ones(2**n, complex) / np.sqrt(2**n)
        for k in range(p):
            psi = psi * jnp.exp(-1j * gam[k] * Ec / scale)
            t = psi.reshape((2,) * n); cb, sb = jnp.cos(bet[k]), -1j * jnp.sin(bet[k])
            for ax in range(n):
                t0, t1 = jnp.take(t, 0, axis=ax), jnp.take(t, 1, axis=ax)
                t = jnp.stack([cb * t0 + sb * t1, sb * t0 + cb * t1], axis=ax)
            psi = t.reshape(-1)
        return psi
    cost = jax.jit(lambda th: jnp.real(jnp.vdot(state(th), Ec * state(th))))
    grad = jax.jit(jax.grad(cost))
    g = np.random.default_rng(seed); best = None
    for _ in range(restarts):
        th0 = np.r_[g.uniform(0, 1, p), g.uniform(0, np.pi / 2, p)]
        r = minimize(lambda t: float(cost(jnp.asarray(t))), th0, jac=lambda t: np.asarray(grad(jnp.asarray(t))),
                     method="L-BFGS-B")
        if best is None or r.fun < best.fun:
            best = r
    prob_s = np.abs(np.asarray(state(jnp.asarray(best.x)))) ** 2
    feas = np.array([feasible(prob, z) for z in Z])
    vals = np.array([obj(prob, z) if f else np.inf for z, f in zip(Z, feas)])
    vopt = vals.min()
    return {"p_feasible": float(prob_s[feas].sum()), "p_opt": float(prob_s[np.isclose(vals, vopt)].sum()),
            "top": Z[int(np.argmax(prob_s))], "uniform_opt": float(np.isclose(vals, vopt).sum() / 2**n)}


def plot(prob, layouts, path):
    fig, axs = plt.subplots(1, len(layouts), figsize=(3.6 * len(layouts), 3.4))
    xs = np.linspace(prob["sites"][:, 0].min() - 5, prob["sites"][:, 0].max() + 5, 220)
    ys = np.linspace(prob["sites"][:, 1].min() - 5, prob["sites"][:, 1].max() + 5, 220)
    X, Y = np.meshgrid(xs, ys)
    for ax, (title, x) in zip(np.atleast_1d(axs), layouts):
        dm = np.zeros_like(X)
        for i in np.where(x)[0]:
            sxx, syy = single_field(X - prob["sites"][i, 0], Y - prob["sites"][i, 1])
            dm += -(PI_L * sxx + PI_T * syy)
        im = ax.pcolormesh(X, Y, 100 * dm, cmap="RdBu_r", vmin=-15, vmax=15, shading="auto")
        ax.scatter(*prob["sites"].T, s=8, c="#999999")
        for i in np.where(x)[0]:
            ax.add_patch(plt.Circle(prob["sites"][i], RAD, color="#C8742C"))
        ax.scatter(*prob["P"].T, s=2, c="k")
        ax.set_title(f"{title}\nΣ(Δμ/μ)² = {obj(prob, x):.3f}", fontsize=8); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axs, shrink=0.8, label="Δμ/μ (%)")
    fig.savefig(path, dpi=200, bbox_inches="tight")


def main():
    out = {"S0_MPa": S0}
    print(f"界面の圧力 S = {S0:.0f} MPa（Si 側の σ_rr = −S (a/r)²）")
    # 例 1: 小さい（QAOA まで）．候補 3×4 = 12，K = 4，隣り合う候補に置かない
    pr = build(4, 3, 10.0, 4, s_min=11.0)
    v_ex, x_ex = exhaustive(pr); x_gr = greedy(pr); x_sa = sa(pr)
    print(f"例 1（候補 12，K = 4）: 全探索 {v_ex:.4f}，貪欲 {obj(pr, x_gr):.4f}，SA {obj(pr, x_sa):.4f}")
    out["ex1"] = {"exact": v_ex, "greedy": obj(pr, x_gr), "sa": obj(pr, x_sa), "qaoa": []}
    for p in (1, 2, 3):
        t0 = time.time(); r = qaoa(pr, p)
        out["ex1"]["qaoa"].append({"p": p, **{k: v for k, v in r.items() if k != "top"}})
        print(f"  QAOA p = {p}: 制約を満たす確率 {r['p_feasible']:.3f}，最適配置の確率 {r['p_opt']:.4f}"
              f"（ランダムなら {r['uniform_opt']:.5f}） {time.time()-t0:.0f}s", flush=True)
    plot(pr, [("exhaustive (optimal)", x_ex), ("greedy", x_gr)], "figs/tsv_placement_ex1.png")
    # 例 2: 大きめ（SA と貪欲法）．候補 8×8 = 64，K = 12
    pr2 = build(8, 8, 10.0, 12, s_min=11.0)
    x_gr2 = greedy(pr2)
    sas = [sa(pr2, sweeps=20000, seed=s) for s in range(5)]
    vs = [obj(pr2, x) if feasible(pr2, x) else np.inf for x in sas]
    rnd = []
    g = np.random.default_rng(1)
    while len(rnd) < 200:
        x = np.zeros(64); x[g.choice(64, 12, replace=False)] = 1
        if feasible(pr2, x):
            rnd.append(obj(pr2, x))
    print(f"例 2（候補 64，K = 12）: 貪欲 {obj(pr2, x_gr2):.4f}，SA（5 回）最良 {min(vs):.4f} 平均 {np.mean(vs):.4f}，"
          f"ランダム配置（200 個）平均 {np.mean(rnd):.4f}")
    out["ex2"] = {"greedy": obj(pr2, x_gr2), "sa": vs, "random_mean": float(np.mean(rnd))}
    x_rnd = None
    while x_rnd is None:
        x = np.zeros(64); x[g.choice(64, 12, replace=False)] = 1
        x_rnd = x if feasible(pr2, x) else None
    plot(pr2, [("random", x_rnd), ("greedy", x_gr2), ("SA (best of 5)", sas[int(np.argmin(vs))])],
         "figs/tsv_placement_ex2.png")
    json.dump(out, open("results/tsv_placement.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
