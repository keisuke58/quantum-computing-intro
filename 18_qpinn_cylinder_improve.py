# -*- coding: utf-8 -*-
"""
18 - OIST Rakala 実装（円柱後流 QPINN）の improved モデルを段階的に改善する

元コード: github.com/GeetRakala/QPINN（qpinn.ipynb）。その improved モデル

    x01 → sin(2π(W1 x + b1)) [3→16] → tanh(W2 h + b2) [16→4]
        → 量子回路（AngleEmbedding + StronglyEntanglingLayers, 4 qubit × 4 層）→ ⟨Z_i⟩
        → tanh(W3 q + b3) [4→16] → W4 h + b4 [16→3] → (u, v, p)

を JAX + PennyLane で忠実に再実装し，改善を 1 つずつ積み上げて効果を測る。

  V0 baseline : 元の improved をそのまま（再現確認）
  V1 angle    : 回転角を tanh 出力（±1 rad）から π 倍（±π）に広げる
  V2 reupload : V1 + データ再アップロード（各層の前にデータを入れ直す）
  V3 readout  : V2 + 測定量を増やす（⟨Z_i⟩ に加えて ⟨X_i⟩ と ⟨Z_i Z_{i+1}⟩，計 12 個）

対照（量子回路が本当に効いているかの確認）:
  C-identity  : V1 から量子回路だけを取り除き，π·tanh(...) をそのまま後段へ渡す
  C-mlp311    : 元の古典 PINN と同じ形（sin 層 + tanh 層）で 311 パラメータにした MLP
                （3→14 sin, 14→14 tanh, 14→3：14² + 8·14 + 3 = 311）

学習・評価は元コードと同じ:
  学習点 5,000（全時空間点の 0.5%，seed 0），損失 = データ MSE + NS 残差 MSE，
  L-BFGS-B 2,000 反復，float32．評価は全 200 時刻の相対 L2 誤差の平均．
"""

import argparse
import json
import os
import time

import numpy as np
import jax
import jax.numpy as jnp
import pennylane as qml
from jax.flatten_util import ravel_pytree
from scipy import optimize
from scipy.io import loadmat

jax.config.update("jax_enable_x64", False)

NU, RHO = 0.01, 1.0
N_Q, N_L = 4, 4
RANGES = [(l % (N_Q - 1)) + 1 for l in range(N_L)]   # StronglyEntanglingLayers の既定 (1,2,3,1)


# ==============================================================
# データ（元コードと同じ処理）
# ==============================================================
def load_data(path):
    d = loadmat(path)
    U, P, t, X = d["U_star"], d["p_star"], d["t"], d["X_star"]
    N, T = X.shape[0], t.shape[0]
    xyt = np.stack([np.repeat(X[:, 0], T), np.repeat(X[:, 1], T), np.tile(t.ravel(), N)], 1)
    uvp = np.stack([U[:, 0, :].reshape(-1), U[:, 1, :].reshape(-1), P.reshape(-1)], 1)
    return {"xyt": xyt, "uvp": uvp, "X": X, "t": t, "U": U, "P": P, "N": N, "T": T}


def sample_training(data, frac=0.005, seed=0):
    np.random.seed(seed)
    idx = np.random.choice(data["xyt"].shape[0], int(round(data["xyt"].shape[0] * frac)), replace=False)
    return data["xyt"][idx].astype(np.float32), data["uvp"][idx].astype(np.float32)


# ==============================================================
# 量子回路
# ==============================================================
_dev = qml.device("default.qubit", wires=N_Q)


@qml.qnode(_dev, interface="jax", diff_method="best")
def circuit_base(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_Q))
    qml.StronglyEntanglingLayers(weights, wires=range(N_Q))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_Q)]


@qml.qnode(_dev, interface="jax", diff_method="best")
def circuit_reupload(inputs, weights):
    # 層ごとにデータを入れ直す．CNOT の結び方（range）は元の 4 層と同じ (1,2,3,1)
    for l in range(N_L):
        qml.AngleEmbedding(inputs, wires=range(N_Q))
        qml.StronglyEntanglingLayers(weights[l:l + 1], wires=range(N_Q), ranges=[RANGES[l]])
    return [qml.expval(qml.PauliZ(i)) for i in range(N_Q)]


@qml.qnode(_dev, interface="jax", diff_method="best")
def circuit_reupload_rich(inputs, weights):
    for l in range(N_L):
        qml.AngleEmbedding(inputs, wires=range(N_Q))
        qml.StronglyEntanglingLayers(weights[l:l + 1], wires=range(N_Q), ranges=[RANGES[l]])
    return ([qml.expval(qml.PauliZ(i)) for i in range(N_Q)]
            + [qml.expval(qml.PauliX(i)) for i in range(N_Q)]
            + [qml.expval(qml.PauliZ(i) @ qml.PauliZ((i + 1) % N_Q)) for i in range(N_Q)])


# ==============================================================
# モデル
# ==============================================================
VARIANTS = {
    #  name          circuit                 angle_scale  n_readout
    "V0-baseline": (circuit_base,          1.0,         4),
    "V1-angle":    (circuit_base,          np.pi,       4),
    "V2-reupload": (circuit_reupload,      np.pi,       4),
    "V3-readout":  (circuit_reupload_rich, np.pi,       12),
    "C-identity":  (None,                  np.pi,       4),
}


def make_model(name, data, key, hidden=16):
    xs, ts = data["X"], data["t"]
    off = jnp.array([xs[:, 0].min(), xs[:, 1].min(), ts.min()], dtype=jnp.float32)
    sc = jnp.array([1 / (xs[:, 0].max() - xs[:, 0].min()), 1 / (xs[:, 1].max() - xs[:, 1].min()),
                    1 / (ts.max() - ts.min())], dtype=jnp.float32)

    if name == "C-mlp311":
        H = 14
        k = jax.random.split(key, 3)

        def glorot(kk, i, o):
            lim = np.sqrt(6.0 / (i + o))
            return jax.random.uniform(kk, (i, o), minval=-lim, maxval=lim)
        params = [{"W": glorot(k[0], 3, H), "b": jnp.zeros(H)},
                  {"W": glorot(k[1], H, H), "b": jnp.zeros(H)},
                  {"W": glorot(k[2], H, 3), "b": jnp.zeros(3)}]

        def net(p, x):
            z = jnp.sin(2 * jnp.pi * (x @ p[0]["W"] + p[0]["b"]))   # 元の古典 PINN と同じく正規化なし
            z = jnp.tanh(z @ p[1]["W"] + p[1]["b"])
            return z @ p[2]["W"] + p[2]["b"]
        return params, net

    circ, ang, n_out = VARIANTS[name]
    k = jax.random.split(key, 5)
    params = {
        "w1": jax.random.normal(k[0], (3, hidden)) * jnp.sqrt(2.0 / 3), "b1": jnp.zeros(hidden),
        "w2": jax.random.normal(k[1], (hidden, N_Q)) * jnp.sqrt(2.0 / hidden), "b2": jnp.zeros(N_Q),
        "w3": jax.random.normal(k[3], (n_out, hidden)) * jnp.sqrt(2.0 / n_out), "b3": jnp.zeros(hidden),
        "w4": jax.random.normal(k[4], (hidden, 3)) * jnp.sqrt(2.0 / hidden), "b4": jnp.zeros(3),
    }
    if circ is not None:
        params["qw"] = jax.random.normal(k[2], (N_L, N_Q, 3)) * 0.01

    def net(p, x):
        h1 = jnp.sin(2 * jnp.pi * (((x - off) * sc) @ p["w1"] + p["b1"]))
        h2 = jnp.tanh(h1 @ p["w2"] + p["b2"])
        if circ is None:
            q = ang * h2
        else:
            q = jnp.array(circ(ang * h2, p["qw"]))
        h3 = jnp.tanh(q @ p["w3"] + p["b3"])
        return h3 @ p["w4"] + p["b4"]
    return params, net


# ==============================================================
# 損失（元コードと同じ）
# ==============================================================
def make_loss(net):
    def residual(p, z):
        u = lambda zz: net(p, zz)[0]
        v = lambda zz: net(p, zz)[1]
        pr = lambda zz: net(p, zz)[2]
        du, dv, dp = jax.grad(u)(z), jax.grad(v)(z), jax.grad(pr)(z)
        Hu, Hv = jax.hessian(u)(z), jax.hessian(v)(z)
        uu, vv, _ = net(p, z)
        f = du[2] + uu * du[0] + vv * du[1] + dp[0] / RHO - NU * (Hu[0, 0] + Hu[1, 1])
        g = dv[2] + uu * dv[0] + vv * dv[1] + dp[1] / RHO - NU * (Hv[0, 0] + Hv[1, 1])
        h = du[0] + dv[1]
        return f, g, h

    def loss(p, xyt, uvp):
        pred = jax.vmap(lambda z: net(p, z))(xyt)
        l_data = jnp.mean((pred - uvp) ** 2, axis=0).sum()
        f, g, h = jax.vmap(lambda z: residual(p, z))(xyt)
        l_pde = jnp.mean(f**2) + jnp.mean(g**2) + jnp.mean(h**2)
        return l_data + l_pde, (l_data, l_pde)
    return loss


def train_lbfgs(params, net, xyt, uvp, maxiter=2000):
    loss = make_loss(net)
    vg = jax.jit(jax.value_and_grad(loss, has_aux=True))
    theta0, unravel = ravel_pytree(params)
    xyt, uvp = jnp.asarray(xyt), jnp.asarray(uvp)
    hist, last = [], {"l": float("nan")}

    def obj(th):
        (l, aux), g = vg(unravel(jnp.asarray(th, dtype=jnp.float32)), xyt, uvp)
        last["l"] = float(l)
        return float(l), np.asarray(ravel_pytree(g)[0], dtype=np.float64)

    def cb(th):
        hist.append(last["l"])   # 直前に評価した損失を記録（再計算しない）

    t0 = time.time()
    res = optimize.minimize(obj, np.asarray(theta0, dtype=np.float64), jac=True, method="L-BFGS-B",
                            callback=cb, options={"maxiter": maxiter})
    return unravel(jnp.asarray(res.x, dtype=jnp.float32)), time.time() - t0, hist


def evaluate(params, net, data):
    f = jax.jit(jax.vmap(lambda z: net(params, z)))
    errs = []
    for ti in range(data["T"]):
        tv = float(data["t"][ti, 0])
        xyt = np.hstack([data["X"], np.full((data["N"], 1), tv)]).astype(np.float32)
        pr = np.asarray(f(jnp.asarray(xyt)))
        ref = np.stack([data["U"][:, 0, ti], data["U"][:, 1, ti], data["P"][:, ti]], 1)
        errs.append(np.linalg.norm(pr - ref, axis=0) / (np.linalg.norm(ref, axis=0) + 1e-12))
    return np.mean(errs, axis=0)   # (u, v, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/user/geetrakala/qpinn/data/cylinder_wake.mat")
    ap.add_argument("--variants", default="V0-baseline,V1-angle,V2-reupload,V3-readout,C-identity,C-mlp311")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--maxiter", type=int, default=2000)
    ap.add_argument("--out", required=True, help="結果を書き出す JSON")
    ap.add_argument("--save-params", default=None, help="学習済みパラメータの保存先ディレクトリ")
    a = ap.parse_args()

    data = load_data(a.data)
    xyt, uvp = sample_training(data)
    rows = json.load(open(a.out)) if os.path.exists(a.out) else []
    for name in a.variants.split(","):
        for s in [int(v) for v in a.seeds.split(",")]:
            if any(r["variant"] == name and r["seed"] == s for r in rows):
                continue
            params, net = make_model(name, data, jax.random.PRNGKey(42 + s))
            n_param = int(ravel_pytree(params)[0].size)
            pt, tt, hist = train_lbfgs(params, net, xyt, uvp, maxiter=a.maxiter)
            e = evaluate(pt, net, data)
            rows.append({"variant": name, "seed": s, "n_param": n_param, "u": float(e[0]),
                         "v": float(e[1]), "p": float(e[2]), "time": tt, "loss_hist": hist})
            json.dump(rows, open(a.out, "w"))
            if a.save_params:
                os.makedirs(a.save_params, exist_ok=True)
                np.save(os.path.join(a.save_params, f"{name}_s{s}.npy"),
                        np.asarray(ravel_pytree(pt)[0]))
            print(f"{name:<12} seed={s} params={n_param:<4} u={e[0]:.4f} v={e[1]:.4f} "
                  f"p={e[2]:.4f} loss={hist[-1] if hist else float('nan'):.3e} {tt:.0f}s", flush=True)


if __name__ == "__main__":
    main()
