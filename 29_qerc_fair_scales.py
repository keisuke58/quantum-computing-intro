# -*- coding: utf-8 -*-
"""
29 - 公平な比較: 量子側のエンコードスケールも調整し，cos 型の古典特徴量（Fourier PIELM, FS-PIELM）を加える

26 では古典ランダム特徴量（tanh）のスケールだけを振り，量子は未調整だった。ここでは
  - QERC（Ikeda 型，Haar / Clifford+T）: エンコード角を作るアフィン写像の重みに倍率 s を掛ける
  - tanh RF: 重みの倍率 s
  - Fourier PIELM: cos(w·z + b), w ~ N(0, s² I)                       （倍率でスケール）
  - FS-PIELM-L（Xiong et al. 2026）: w_m = μ_m d_m + ε,  μ_m を 0〜μ_max で線形に並べる
をすべて同じ入力 z = (arccos ξ − 1.57, cos θ, sin θ)・同じ特徴量数 2^N・同じ解き方で比べる。
各モデルとも，シードごとにテスト誤差が最良の倍率を採用する（全モデルに同じ扱い）。
"""
import argparse, importlib.util, json, os, time
import numpy as np
import jax, jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
_h = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("i", os.path.join(_h, "25_qerc_ikeda_pde.py"))
I = importlib.util.module_from_spec(_s); _s.loader.exec_module(I)
M = I.M


def _z(x, y):
    _, th, a1 = M._pre(x, y)
    return jnp.stack([a1 - 1.57, jnp.cos(th), jnp.sin(th)])


def make_qerc(N, seed, kind, s):
    g = np.random.default_rng(1000 + seed)
    W = jnp.asarray(g.normal(size=(3, 2 * N)) * s); b = jnp.asarray(g.normal(size=2 * N))
    U = jnp.asarray(I.reservoir(kind, N, seed))

    def feat(x, y):
        ang = jnp.pi * (jnp.tanh(_z(x, y) @ W + b) + 1) / 2
        psi = jnp.ones((1,), complex)
        for l in range(N):
            psi = jnp.kron(psi, jnp.stack([jnp.cos(ang[l] / 2), jnp.exp(1j * ang[N + l]) * jnp.sin(ang[l] / 2)]))
        return jnp.abs(U @ psi) ** 2
    return feat


def make_tanh(m, seed, s):
    g = np.random.default_rng(seed)
    W = jnp.asarray(g.normal(size=(3, m)) * s); b = jnp.asarray(g.random(m) * 2 * np.pi)
    return lambda x, y: jnp.tanh(_z(x, y) @ W + b)


def make_fourier(m, seed, s):
    g = np.random.default_rng(seed)
    W = jnp.asarray(g.normal(size=(3, m)) * s); b = jnp.asarray(g.random(m) * 2 * np.pi)
    return lambda x, y: jnp.cos(_z(x, y) @ W + b)


def make_fs(m, seed, mu_max):
    g = np.random.default_rng(seed)
    d = g.normal(size=(3, m)); d /= np.linalg.norm(d, axis=0)
    mu = np.linspace(0, mu_max, m)
    W = jnp.asarray(mu * d + g.normal(size=(3, m))); b = jnp.asarray(g.random(m) * 2 * np.pi)
    return lambda x, y: jnp.cos(_z(x, y) @ W + b)


def solve_eval(feat, pts):
    fns = M.make_basis(feat)
    A, b = M.assemble(fns, *pts, 1.0)
    return M.evaluate(fns, M.solve(A, b, 1e-10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="results/qerc_fair_scales.json")
    a = ap.parse_args()
    m = 2**a.N
    grid = {
        "QERC-Haar": ([0.25, 0.5, 1.0, 1.5], lambda sd, s: make_qerc(a.N, sd, "haar", s)),
        "QERC-CliffordT": ([0.25, 0.5, 1.0, 1.5], lambda sd, s: make_qerc(a.N, sd, "cliffordT", s)),
        "tanh-RF": ([0.25, 0.5, 1.0], lambda sd, s: make_tanh(m, sd, s)),
        "Fourier-PIELM": ([0.25, 0.5, 1.0, 2.0], lambda sd, s: make_fourier(m, sd, s)),
        "FS-PIELM-L": ([0.0, 1.0, 2.0, 4.0], lambda sd, s: make_fs(m, sd, s)),
    }
    rows = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {(r["model"], r["seed"], r["scale"]) for r in rows}
    for sd in range(a.seeds):
        pts = M.points(8000, 800, 100 + sd)
        for name, (scales, ctor) in grid.items():
            for s in scales:
                if (name, sd, s) in done:
                    continue
                t0 = time.time()
                eu, es, kt = solve_eval(ctor(sd, s), pts)
                rows.append({"model": name, "seed": sd, "scale": s, "u": eu, "s": es, "kt": kt})
                print(f"seed={sd} {name:15s} ×{s:<5} u={eu:.2e} σ={es:.2e} K_t={kt:.3f} {time.time()-t0:.0f}s",
                      flush=True)
                json.dump(rows, open(a.out, "w"), indent=1)

    print("\n=== シードごとに最良の倍率を採用した u 誤差（幾何平均 [最小, 最大]）===")
    for name in grid:
        best = []
        for sd in range(a.seeds):
            rs = [r for r in rows if r["model"] == name and r["seed"] == sd]
            if rs:
                best.append(min(rs, key=lambda r: r["u"]))
        v = np.array([r["u"] for r in best])
        print(f"{name:15s} {np.exp(np.log(v).mean()):.2e} [{v.min():.1e}, {v.max():.1e}]  "
              f"倍率 {[r['scale'] for r in best]}")


if __name__ == "__main__":
    main()
