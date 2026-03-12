# -*- coding: utf-8 -*-
"""
01 - 量子ビットの基礎

量子ビット (qubit) は古典ビット (0 or 1) と違い、
0 と 1 の「重ね合わせ」状態を取れる。

|ψ⟩ = α|0⟩ + β|1⟩   (|α|² + |β|² = 1)

このスクリプトでは:
1. 量子ビットの初期状態 |0⟩ を確認
2. アダマールゲート (H) で重ね合わせを作る
3. 測定して確率分布を見る
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# ==============================================================
# 1. 最もシンプルな回路: 1量子ビット + 1古典ビット
# ==============================================================
print("=" * 50)
print("  1. 量子ビットの初期状態 |0⟩")
print("=" * 50)

qc = QuantumCircuit(1, 1)  # 1 qubit, 1 classical bit
qc.measure(0, 0)           # 測定

print(qc.draw(output="text"))

# シミュレーション (1024回測定)
simulator = AerSimulator()
result = simulator.run(qc, shots=1024).result()
counts = result.get_counts()
print(f"測定結果: {counts}")
print("→ |0⟩ 状態なので、100% '0' が出る\n")

# ==============================================================
# 2. アダマールゲートで重ね合わせを作る
# ==============================================================
print("=" * 50)
print("  2. H ゲート → 重ね合わせ |+⟩")
print("=" * 50)

qc2 = QuantumCircuit(1, 1)
qc2.h(0)          # アダマールゲート: |0⟩ → (|0⟩ + |1⟩)/√2
qc2.measure(0, 0)

print(qc2.draw(output="text"))

result2 = simulator.run(qc2, shots=1024).result()
counts2 = result2.get_counts()
print(f"測定結果: {counts2}")
print("→ '0' と '1' がほぼ 50:50 で出る（重ね合わせ！）\n")

# ==============================================================
# 3. Statevector で状態を直接見る
# ==============================================================
print("=" * 50)
print("  3. Statevector で量子状態を確認")
print("=" * 50)

from qiskit.quantum_info import Statevector

# |0⟩ の状態ベクトル
sv0 = Statevector.from_label("0")
print(f"|0⟩ = {sv0.data}")

# H|0⟩ の状態ベクトル
qc3 = QuantumCircuit(1)
qc3.h(0)
sv_plus = Statevector(qc3)
print(f"H|0⟩ = |+⟩ = {sv_plus.data}")
print(f"各状態の確率: |0⟩={abs(sv_plus.data[0])**2:.3f}, |1⟩={abs(sv_plus.data[1])**2:.3f}")

# ==============================================================
# まとめ
# ==============================================================
print("\n" + "=" * 50)
print("  まとめ")
print("=" * 50)
print("""
- 量子ビットの初期状態は |0⟩
- H ゲートで重ね合わせ状態 |+⟩ = (|0⟩ + |1⟩)/√2 を作れる
- 測定すると重ね合わせは壊れ、0 か 1 のどちらかに確定する
- 確率は状態ベクトルの振幅の二乗で決まる

次: 02_quantum_gates.py で様々なゲートを学ぶ
""")
