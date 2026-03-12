# -*- coding: utf-8 -*-
"""
07 - 量子テレポーテーション

量子テレポーテーション = もつれと古典通信を使って、
未知の量子状態を別の場所に転送するプロトコル。

注意: 情報が光速を超えて伝わるわけではない！
（古典通信 2ビットが必要 → 超光速通信は不可能）

プロトコル:
1. Alice と Bob が Bell ペアを共有
2. Alice が送りたい量子ビットと自分の Bell ビットを測定
3. 測定結果（2 classical bits）を Bob に送る
4. Bob が測定結果に基づいて補正 → 元の状態を復元
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, state_fidelity
import numpy as np

simulator = AerSimulator()

# ==============================================================
# 1. テレポーテーション回路
# ==============================================================
print("=" * 50)
print("  量子テレポーテーション")
print("=" * 50)


def teleportation_circuit(state_prep_gate=None):
    """
    量子テレポーテーション回路を作る

    qubit 0: Alice の送りたい状態 |ψ⟩
    qubit 1: Alice の Bell ペア半分
    qubit 2: Bob の Bell ペア半分
    """
    qc = QuantumCircuit(3, 2)

    # --- 送りたい状態を準備 ---
    if state_prep_gate:
        state_prep_gate(qc, 0)
    qc.barrier(label="state prep")

    # --- Step 1: Bell ペアを作る (qubit 1, 2) ---
    qc.h(1)
    qc.cx(1, 2)
    qc.barrier(label="Bell pair")

    # --- Step 2: Alice の操作 ---
    qc.cx(0, 1)   # CNOT
    qc.h(0)        # Hadamard
    qc.barrier(label="Alice")

    # --- Step 3: Alice が測定 ---
    qc.measure(0, 0)
    qc.measure(1, 1)

    # --- Step 4: Bob が補正（条件付きゲート）---
    qc.x(2).c_if(1, 1)   # classical bit 1 が 1 なら X
    qc.z(2).c_if(0, 1)   # classical bit 0 が 1 なら Z

    return qc


# ==============================================================
# 2. |0⟩ をテレポート
# ==============================================================
print("\n--- |0⟩ をテレポート ---")
qc_0 = teleportation_circuit()
print(qc_0.draw(output="text"))

# ==============================================================
# 3. 任意の状態をテレポート（Statevector で検証）
# ==============================================================
print("\n" + "=" * 50)
print("  任意の状態のテレポーテーション検証")
print("=" * 50)

# テレポートしたい状態: Ry(π/3)|0⟩
theta = np.pi / 3

# 元の状態を計算
qc_orig = QuantumCircuit(1)
qc_orig.ry(theta, 0)
original_state = Statevector(qc_orig)
print(f"\n送りたい状態: Ry({theta:.4f})|0⟩")
print(f"  |0⟩ の確率: {abs(original_state.data[0])**2:.4f}")
print(f"  |1⟩ の確率: {abs(original_state.data[1])**2:.4f}")

# テレポーテーション（測定なし版で状態忠実度を検証）
qc_tp = QuantumCircuit(3)

# 状態準備
qc_tp.ry(theta, 0)

# Bell ペア
qc_tp.h(1)
qc_tp.cx(1, 2)

# Alice の操作
qc_tp.cx(0, 1)
qc_tp.h(0)

# 理想的な補正（全4パターンの条件付き）
# 測定の代わりに、制御ゲートで実装
qc_tp.cx(1, 2)
qc_tp.cz(0, 2)

# 全体の Statevector を取得
sv_full = Statevector(qc_tp)

# qubit 2 の縮約状態を取得（qubit 0, 1 をトレースアウト）
from qiskit.quantum_info import partial_trace, DensityMatrix

dm_full = DensityMatrix(sv_full)
dm_bob = partial_trace(dm_full, [0, 1])  # qubit 0, 1 をトレース

# 忠実度を計算
fidelity = state_fidelity(original_state, dm_bob)
print(f"\n  テレポーテーション忠実度: {fidelity:.6f}")
print(f"  → {'完璧なテレポーテーション！' if fidelity > 0.999 else '忠実度が低い...'}")

# ==============================================================
# 4. 複数の状態でテスト
# ==============================================================
print("\n" + "=" * 50)
print("  様々な状態のテレポーテーション")
print("=" * 50)

test_states = {
    "|0⟩": (0, 0),
    "|1⟩": (np.pi, 0),
    "|+⟩": (np.pi / 2, 0),
    "|−⟩": (np.pi / 2, np.pi),
    "Ry(π/6)|0⟩": (np.pi / 6, 0),
    "arbitrary": (1.23, 4.56),
}

for name, (th, ph) in test_states.items():
    # 元の状態
    qc_o = QuantumCircuit(1)
    qc_o.ry(th, 0)
    qc_o.rz(ph, 0)
    state_orig = Statevector(qc_o)

    # テレポーテーション
    qc_t = QuantumCircuit(3)
    qc_t.ry(th, 0)
    qc_t.rz(ph, 0)
    qc_t.h(1)
    qc_t.cx(1, 2)
    qc_t.cx(0, 1)
    qc_t.h(0)
    qc_t.cx(1, 2)
    qc_t.cz(0, 2)

    sv_t = Statevector(qc_t)
    dm_t = partial_trace(DensityMatrix(sv_t), [0, 1])
    fid = state_fidelity(state_orig, dm_t)

    status = "OK" if fid > 0.999 else "NG"
    print(f"  {name:15s}: fidelity = {fid:.6f} [{status}]")

print("""
まとめ:
- 量子テレポーテーション = もつれ + 古典通信で量子状態を転送
- 物質やエネルギーは転送されない（情報のみ）
- 超光速通信は不可能（古典通信が必要）
- 量子ネットワーク、量子中継器、分散量子計算の基盤技術

お疲れさまでした！これで基礎は完了です。
次のステップ:
- Qiskit Textbook で実機での実行を試す
- QAOA で組合せ最適化
- 量子エラー補正
- 量子機械学習 (QML)
""")
