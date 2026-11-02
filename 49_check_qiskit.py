# -*- coding: utf-8 -*-
"""
49 - JAX で書いた量子リザバー特徴量（30 の make_qerc）が Qiskit と一致するかを小さい N で確かめる

Qiskit 側の回路: 各量子ビットに RY(θ_l) → P(φ_l)（= cos(θ/2)|0> + e^{iφ} sin(θ/2)|1>），その後に同じユニタリ U．
JAX は jnp.kron で量子ビット 0 を最上位ビットにするので，Qiskit（量子ビット 0 が最下位）では
JAX の量子ビット l を Qiskit の量子ビット N−1−l に置く．こうすると状態ベクトルの添字の並びが一致する．
比較: (1) Statevector の厳密な確率，(2) Aer のサンプリング（ショット数 S）と統計誤差 √(p(1−p)/S) の比較
"""
import importlib.util, os
import numpy as np
import jax.numpy as jnp
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector

_h = os.path.dirname(os.path.abspath(__file__))
s = importlib.util.spec_from_file_location("t30", os.path.join(_h, "30_qerc_tsv_single.py"))
T = importlib.util.module_from_spec(s); s.loader.exec_module(T)


def params(N, seed, scale):
    """make_qerc と同じ乱数で W, b, U を作り直す"""
    g = np.random.default_rng(1000 + seed)
    W = g.normal(size=(2, 2 * N)) * scale; b = g.normal(size=2 * N)
    z = (g.normal(size=(2**N,) * 2) + 1j * g.normal(size=(2**N,) * 2)) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    return W, b, q * (np.diag(r) / np.abs(np.diag(r)))


def qiskit_probs(N, W, b, U, x, y):
    ang = np.pi * (np.tanh(np.array([x, y]) / T.B_R @ W + b) + 1) / 2
    qc = QuantumCircuit(N)
    for l in range(N):
        qc.ry(ang[l], N - 1 - l)
        qc.p(ang[N + l], N - 1 - l)
    qc.append(UnitaryGate(U), list(range(N)))
    return Statevector(qc).probabilities(), qc


rng = np.random.default_rng(0)
for N in (2, 3, 4, 6):
    W, b, U = params(N, seed=0, scale=1.0)
    feat = T.make_qerc(N, 0, 1.0)
    err = 0.0
    for _ in range(20):
        x, y = rng.uniform(-4, 4, 2)
        pj = np.asarray(feat(jnp.asarray(x), jnp.asarray(y)))
        pq, qc = qiskit_probs(N, W, b, U, x, y)
        err = max(err, np.abs(pj - pq).max())
    print(f"N = {N}: JAX と Qiskit（Statevector）の確率の最大差 {err:.1e}（20 点）")

# サンプリング（実機に近い形）: ショット数 S で数えた頻度と JAX の確率
try:
    from qiskit_aer import AerSimulator
    from qiskit import transpile
    N, S = 4, 200000
    W, b, U = params(N, 0, 1.0); feat = T.make_qerc(N, 0, 1.0)
    x, y = 1.3, -0.7
    pj = np.asarray(feat(jnp.asarray(x), jnp.asarray(y)))
    _, qc = qiskit_probs(N, W, b, U, x, y); qc.measure_all()
    sim = AerSimulator(seed_simulator=1)
    counts = sim.run(transpile(qc, sim), shots=S).result().get_counts()
    freq = np.zeros(2**N)
    for k, v in counts.items():
        freq[int(k, 2)] = v / S
    z = (freq - pj) / np.sqrt(pj * (1 - pj) / S)
    print(f"N = {N}, {S} ショット: 頻度と確率の差を統計誤差で割った値 |z| の最大 {np.abs(z).max():.2f}（2〜3 以下なら一致）")
except ImportError:
    print("qiskit-aer が無いのでサンプリングの比較は省略")
