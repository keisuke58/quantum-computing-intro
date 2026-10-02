# -*- coding: utf-8 -*-
"""
31 - TSV 1 本（30）に，量子特徴量の優位を疑う 2 つの古典対照を加える

QERC の特徴量は p_k(x) = ψ(x)† Q_k ψ(x)，Q_k = U†|k⟩⟨k|U（ランク 1 の射影），
ψ(x) = ⊗_l (cos(a_l/2), e^{iφ_l} sin(a_l/2)) は 2N 個の角度から作る積状態．

  (1) テンソル積の古典ランダム射影:  F_k(x) = Re[ψ(x)† R_k ψ(x)]，R_k はランダムなエルミート行列（GUE）．
      量子と全く同じ関数空間（ρ(x) の行列要素の張る空間）から，測定の構造なしにランダムに取り出す．
      古典計算機では 2^N × 2^N の行列が特徴量ごとに要る（量子なら N qubit の測定で済む）．
  (2) 2 層の古典ネットワークの特徴量:  h = tanh(W1 z + b1)（幅 2N，量子の角度と同じ数），
      F = tanh(W2 h + b2)（幅 2^N）．重みは固定のランダム．「積」に近い非線形の合成を持つ古典の対照．

QERC と (1) は角度の作り方（W, b）を完全に共有する．
"""
import argparse, importlib.util, json, os, time
import numpy as np
import jax, jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
_h = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("t", os.path.join(_h, "30_qerc_tsv_single.py"))
T = importlib.util.module_from_spec(_s); _s.loader.exec_module(T)


def _angles(N, seed, scale):
    g = np.random.default_rng(1000 + seed)
    W = jnp.asarray(g.normal(size=(2, 2 * N)) * scale); b = jnp.asarray(g.normal(size=2 * N))
    return g, lambda x, y: jnp.pi * (jnp.tanh(T._z(x, y) @ W + b) + 1) / 2


def _psi(ang, N):
    psi = jnp.ones((1,), complex)
    for l in range(N):
        psi = jnp.kron(psi, jnp.stack([jnp.cos(ang[l] / 2), jnp.exp(1j * ang[N + l]) * jnp.sin(ang[l] / 2)]))
    return psi


def make_tensor_rp(N, seed, scale, m=None):
    g, angf = _angles(N, seed, scale)
    d = 2**N; m = m or d
    Z = (g.normal(size=(m, d, d)) + 1j * g.normal(size=(m, d, d))) / np.sqrt(2 * d)
    R = jnp.asarray((Z + np.conj(np.transpose(Z, (0, 2, 1)))) / 2)      # GUE

    def feat(x, y, R=R):
        p = _psi(angf(x, y), N)
        return jnp.real(jnp.einsum("i,kij,j->k", jnp.conj(p), R, p))
    feat.extra = R            # jit の定数畳み込みを避けるため，R は引数として渡す
    return feat


def make_mlp2(N, seed, scale):
    g = np.random.default_rng(3000 + seed)
    W1 = jnp.asarray(g.normal(size=(2, 2 * N)) * scale); b1 = jnp.asarray(g.normal(size=2 * N))
    W2 = jnp.asarray(g.normal(size=(2 * N, 2**N)) * 2 / np.sqrt(2 * N)); b2 = jnp.asarray(g.normal(size=2**N))
    return lambda x, y: jnp.tanh(jnp.tanh(T._z(x, y) @ W1 + b1) @ W2 + b2)


def chunked_derivs(feat, chunk=256):
    """30 の derivs と同じ出力を，メモリを抑えるため点を分割して計算する"""
    extra = getattr(feat, "extra", None)
    if extra is not None:
        chunk = 32
    f = lambda p, e: feat(p[0], p[1], e) if extra is not None else feat(p[0], p[1])
    fs = [jax.jit(jax.vmap(f, (0, None))), jax.jit(jax.vmap(jax.jacfwd(f), (0, None))),
          jax.jit(jax.vmap(jax.jacfwd(jax.jacfwd(f)), (0, None)))]

    def wrap(fn):
        def g(P):
            P = np.asarray(P)
            return np.concatenate([np.asarray(fn(jnp.asarray(P[i:i + chunk]), extra))
                                   for i in range(0, len(P), chunk)])
        return g
    return tuple(wrap(fn) for fn in fs)


def run(feat_cu, feat_si, pts, ridge=1e-10):
    d1, d2 = chunked_derivs(feat_cu), chunked_derivs(feat_si)
    A, b, m1 = T.assemble(d1, d2, pts)
    return T.evaluate(d1, d2, T.solve(A, b, ridge), m1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="results/tsv_classical_controls.json")
    a = ap.parse_args()
    grid = {
        "QERC-Haar": ([1.0, 2.0, 4.0], lambda sd, s: T.make_qerc(a.N, sd, s)),
        "tensor-RP": ([2.0, 4.0], lambda sd, s: make_tensor_rp(a.N, sd, s)),
        "MLP2-RF": ([1.0, 2.0, 4.0, 8.0], lambda sd, s: make_mlp2(a.N, sd, s)),
    }
    rows = []
    for sd in range(a.seeds):
        pts = T.points(2000, 6000, 400, 400, 10 + sd)
        for name, (scales, ctor) in grid.items():
            for s in scales:
                t0 = time.time()
                eu, es = run(ctor(2 * sd, s), ctor(2 * sd + 1, s), pts)
                rows.append({"model": name, "seed": sd, "scale": s, "u": eu, "s": es})
                print(f"seed={sd} {name:10s} ×{s:<4} u={eu:.2e} σ={es:.2e} {time.time()-t0:.0f}s", flush=True)
                json.dump(rows, open(a.out, "w"), indent=1)
    print("\n=== シードごとに最良の倍率 ===")
    for name in grid:
        v = [min(r["u"] for r in rows if r["model"] == name and r["seed"] == sd) for sd in range(a.seeds)]
        print(f"{name:10s} u: 幾何平均 {np.exp(np.mean(np.log(v))):.2e}  {['%.1e' % x for x in v]}")


if __name__ == "__main__":
    main()
