# -*- coding: utf-8 -*-
"""
06 - VQE (Variational Quantum Eigensolver) 入門

VQE = 量子-古典ハイブリッドアルゴリズム
量子コンピュータで期待値を計算し、古典コンピュータでパラメータを最適化する。

ここでは H₂ 分子のハミルトニアンの基底状態エネルギーを求める
（量子化学の最も基本的な問題）。

簡略化: 2量子ビットのパウリハミルトニアンを直接指定。
"""

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
import numpy as np
from scipy.optimize import minimize

# ==============================================================
# 1. ハミルトニアンを定義
# ==============================================================
print("=" * 50)
print("  VQE: 変分量子固有値ソルバー")
print("=" * 50)

# H₂ 分子の簡略ハミルトニアン (2-qubit)
# H = -1.05 II + 0.39 IZ - 0.39 ZI - 0.01 ZZ + 0.18 XX
hamiltonian = SparsePauliOp.from_list([
    ("II", -1.05),
    ("IZ",  0.39),
    ("ZI", -0.39),
    ("ZZ", -0.01),
    ("XX",  0.18),
])

# 厳密解（numpy で対角化）
exact_energy = min(np.linalg.eigvalsh(hamiltonian.to_matrix().toarray()))
print(f"\n厳密な基底状態エネルギー: {exact_energy:.6f}")

# ==============================================================
# 2. Ansatz（変分回路）を定義
# ==============================================================
print("\n" + "=" * 50)
print("  Ansatz (変分回路)")
print("=" * 50)


def create_ansatz(params):
    """パラメータ付き量子回路"""
    qc = QuantumCircuit(2)
    qc.ry(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 0)
    qc.ry(params[3], 1)
    return qc


# 回路構造を表示
demo_qc = create_ansatz([0.1, 0.2, 0.3, 0.4])
print(demo_qc.draw(output="text"))
print(f"パラメータ数: 4")

# ==============================================================
# 3. コスト関数（期待値計算）
# ==============================================================

def cost_function(params):
    """ハミルトニアンの期待値を計算"""
    qc = create_ansatz(params)
    sv = Statevector(qc)
    energy = sv.expectation_value(hamiltonian).real
    return energy


# ==============================================================
# 4. VQE 最適化ループ
# ==============================================================
print("\n" + "=" * 50)
print("  VQE 最適化")
print("=" * 50)

# 初期パラメータ（ランダム）
np.random.seed(42)
initial_params = np.random.uniform(-np.pi, np.pi, 4)
print(f"初期エネルギー: {cost_function(initial_params):.6f}")

# 最適化（COBYLA）
history = []


def callback(params):
    energy = cost_function(params)
    history.append(energy)


result = minimize(
    cost_function,
    initial_params,
    method="COBYLA",
    callback=callback,
    options={"maxiter": 200},
)

print(f"VQE エネルギー: {result.fun:.6f}")
print(f"厳密解:        {exact_energy:.6f}")
print(f"誤差:          {abs(result.fun - exact_energy):.6f}")
print(f"反復回数:      {len(history)}")
print(f"最適パラメータ: [{', '.join(f'{p:.3f}' for p in result.x)}]")

# ==============================================================
# 5. 収束の可視化
# ==============================================================
print("\n収束過程:")
n_show = min(10, len(history))
step = max(1, len(history) // n_show)
for i in range(0, len(history), step):
    bar_len = int((history[i] - exact_energy) * 20)
    bar = "=" * max(0, bar_len)
    print(f"  iter {i:3d}: E = {history[i]:+.4f} |{bar}|")
print(f"  final  : E = {result.fun:+.4f} | (exact = {exact_energy:.4f})")

print("""
まとめ:
- VQE は NISQ デバイスで使える最も実用的なアルゴリズムの一つ
- 量子回路で状態を準備 → 期待値を測定 → 古典で最適化
- Ansatz（回路の形）の設計が精度を左右する
- 量子化学、材料科学、最適化問題に応用

次: 07_quantum_teleportation.py で量子テレポーテーション
""")
