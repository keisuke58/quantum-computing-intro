# -*- coding: utf-8 -*-
"""
05 - Grover の探索アルゴリズム

問題: N 個の要素から特定の要素を見つける
  古典: O(N) 回の探索
  量子: O(√N) 回で発見！

4 要素の中から「当たり」を 1 回の反復で見つける例。
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import numpy as np

simulator = AerSimulator()


def grover_oracle(n_qubits, target):
    """target 状態にマイナスの位相をつけるオラクル"""
    qc = QuantumCircuit(n_qubits, name=f"Oracle(|{target}⟩)")

    # target のビットが 0 の位置に X を適用（制御の条件を反転）
    target_bits = format(target, f"0{n_qubits}b")[::-1]
    for i, bit in enumerate(target_bits):
        if bit == "0":
            qc.x(i)

    # 多重制御 Z ゲート = CZ...Z
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)

    # X を戻す
    for i, bit in enumerate(target_bits):
        if bit == "0":
            qc.x(i)

    return qc.to_gate()


def diffusion_operator(n_qubits):
    """Grover の拡散演算子（平均周りの反転）"""
    qc = QuantumCircuit(n_qubits, name="Diffusion")

    qc.h(range(n_qubits))
    qc.x(range(n_qubits))

    # 多重制御 Z
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)

    qc.x(range(n_qubits))
    qc.h(range(n_qubits))

    return qc.to_gate()


# ==============================================================
# 2量子ビット (4要素) での Grover
# ==============================================================
n = 2
N = 2**n
target = 3  # |11⟩ を探す

print("=" * 50)
print(f"  Grover's Search (N={N}, target=|{format(target, f'0{n}b')}⟩)")
print("=" * 50)
print(f"  古典: 平均 {N/2:.0f} 回の探索")
print(f"  量子: {int(np.pi/4 * np.sqrt(N)):.0f} 回の反復\n")

# 回路を構築
qc = QuantumCircuit(n, n)

# Step 1: 均等な重ね合わせ
qc.h(range(n))

# Step 2: Grover 反復（1回）
n_iterations = 1
for _ in range(n_iterations):
    qc.append(grover_oracle(n, target), range(n))
    qc.append(diffusion_operator(n), range(n))

# Step 3: 測定
qc.measure(range(n), range(n))

print("回路図:")
print(qc.decompose().draw(output="text"))

result = simulator.run(qc, shots=1024).result()
counts = result.get_counts()

print(f"\n測定結果 (1024 shots):")
for state in sorted(counts.keys()):
    count = counts[state]
    pct = count / 1024 * 100
    bar = "#" * int(pct / 2)
    marker = " ← TARGET!" if state == format(target, f"0{n}b") else ""
    print(f"  |{state}⟩: {count:4d} ({pct:5.1f}%) {bar}{marker}")

# ==============================================================
# 3量子ビット (8要素) での Grover
# ==============================================================
print("\n" + "=" * 50)
print(f"  Grover's Search (N=8, target=|101⟩)")
print("=" * 50)

n3 = 3
target3 = 5  # |101⟩

qc3 = QuantumCircuit(n3, n3)
qc3.h(range(n3))

# 最適反復回数 ≈ π/4 * √8 ≈ 2
n_iter3 = 2
for _ in range(n_iter3):
    qc3.append(grover_oracle(n3, target3), range(n3))
    qc3.append(diffusion_operator(n3), range(n3))

qc3.measure(range(n3), range(n3))

result3 = simulator.run(qc3, shots=1024).result()
counts3 = result3.get_counts()

target_str = format(target3, f"0{n3}b")
print(f"  反復回数: {n_iter3}")
print(f"\n  測定結果:")
for state in sorted(counts3.keys()):
    count = counts3[state]
    pct = count / 1024 * 100
    bar = "#" * int(pct / 2)
    marker = " ← TARGET!" if state == target_str else ""
    print(f"  |{state}⟩: {count:4d} ({pct:5.1f}%) {bar}{marker}")

print("""
まとめ:
- Grover は「振幅増幅」で正解の確率を上げる
- 1 反復 = オラクル（位相反転）+ 拡散（平均周りの反転）
- 最適反復回数は ≈ π/4 × √N
- データベース検索、SAT、組合せ最適化に応用可能

次: 06_vqe_intro.py で変分量子アルゴリズムを学ぶ
""")
