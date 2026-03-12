# -*- coding: utf-8 -*-
"""
02 - 量子ゲート

量子ゲートはユニタリ行列 U（U†U = I）で表される。
古典の NOT, AND に相当する操作だが、「可逆」なのが特徴。

主要ゲート:
- X (NOT): |0⟩ ↔ |1⟩ を反転
- H (Hadamard): 重ね合わせを作る/壊す
- Z: 位相を反転 |1⟩ → -|1⟩
- CNOT: 2量子ビット制御ゲート
- Ry(θ): Y軸回転（任意の重ね合わせ比率を作る）
"""

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
import numpy as np

# ==============================================================
# 1. パウリゲート (X, Y, Z)
# ==============================================================
print("=" * 50)
print("  1. パウリゲート")
print("=" * 50)

# X ゲート（量子 NOT）
print("\nX ゲート (NOT):")
print(f"  行列:\n{Operator(QuantumCircuit(1).compose(QuantumCircuit(1, name='x').x(0))).data.real}")

qc_x = QuantumCircuit(1)
qc_x.x(0)
sv = Statevector(qc_x)
print(f"  X|0⟩ = {sv.data} → |1⟩")

# Z ゲート（位相反転）
print("\nZ ゲート (位相反転):")
qc_z = QuantumCircuit(1)
qc_z.h(0)   # まず |+⟩ を作る
qc_z.z(0)   # Z を適用
sv_z = Statevector(qc_z)
print(f"  Z|+⟩ = {sv_z.data} → |−⟩ = (|0⟩ - |1⟩)/√2")

# ==============================================================
# 2. 回転ゲート Ry(θ)
# ==============================================================
print("\n" + "=" * 50)
print("  2. 回転ゲート Ry(θ)")
print("=" * 50)

for angle_deg in [30, 45, 60, 90]:
    theta = np.radians(angle_deg)
    qc_ry = QuantumCircuit(1)
    qc_ry.ry(2 * theta, 0)  # Ry(2θ) で sin(θ)|1⟩ の確率を得る
    sv_ry = Statevector(qc_ry)
    prob_1 = abs(sv_ry.data[1]) ** 2
    print(f"  Ry({2*angle_deg}°)|0⟩ → P(|1⟩) = {prob_1:.4f} (= sin²({angle_deg}°) = {np.sin(theta)**2:.4f})")

# ==============================================================
# 3. CNOT (制御NOT) - 2量子ビットゲート
# ==============================================================
print("\n" + "=" * 50)
print("  3. CNOT ゲート")
print("=" * 50)
print("  制御ビットが |1⟩ のとき、ターゲットを反転\n")

for input_state in ["00", "01", "10", "11"]:
    qc_cx = QuantumCircuit(2)
    if input_state[0] == "1":
        qc_cx.x(1)  # qubit 1 (制御)
    if input_state[1] == "1":
        qc_cx.x(0)  # qubit 0 (ターゲット)
    qc_cx.cx(1, 0)  # CNOT: 1→0
    sv_cx = Statevector(qc_cx)
    # 最大確率の状態を取得
    probs = sv_cx.probabilities_dict()
    output = max(probs, key=probs.get)
    print(f"  CNOT|{input_state}⟩ → |{output}⟩")

# ==============================================================
# 4. 回路の可視化
# ==============================================================
print("\n" + "=" * 50)
print("  4. 複合回路の例")
print("=" * 50)

qc_demo = QuantumCircuit(3)
qc_demo.h(0)
qc_demo.cx(0, 1)
qc_demo.cx(0, 2)
qc_demo.barrier()
qc_demo.z(0)
qc_demo.x(1)
qc_demo.h(2)

print(qc_demo.draw(output="text"))

print("""
まとめ:
- X = NOT, H = 重ね合わせ, Z = 位相反転
- Ry(θ) で任意の確率比を作れる
- CNOT は 2 量子ビット間の相関を作る基本ゲート
- 全てのゲートはユニタリ（可逆）

次: 03_entanglement.py で量子もつれを体験
""")
