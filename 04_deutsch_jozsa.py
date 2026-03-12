# -*- coding: utf-8 -*-
"""
04 - Deutsch-Jozsa アルゴリズム

問題: ブラックボックス関数 f(x) が
  - 定値 (constant): 全入力に同じ値を返す (f=0 or f=1)
  - 均等 (balanced): 入力の半分に 0、残り半分に 1 を返す
のどちらかを判定せよ。

古典: 最悪 2^(n-1) + 1 回の評価が必要
量子: たった 1 回の評価で判定可能！

これが量子コンピュータの指数的高速化の最初の例。
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import numpy as np

simulator = AerSimulator()


def deutsch_jozsa_oracle(n, oracle_type="constant_0"):
    """オラクル（ブラックボックス関数）を作る"""
    qc = QuantumCircuit(n + 1, name=f"Oracle({oracle_type})")

    if oracle_type == "constant_0":
        # f(x) = 0 for all x → 何もしない
        pass
    elif oracle_type == "constant_1":
        # f(x) = 1 for all x → ancilla を反転
        qc.x(n)
    elif oracle_type == "balanced":
        # f(x) = x の最初のビット → CNOT
        for i in range(n):
            qc.cx(i, n)
    elif oracle_type == "balanced_random":
        # ランダムな balanced 関数
        rng = np.random.default_rng(42)
        for i in range(n):
            if rng.random() > 0.5:
                qc.cx(i, n)
        # 位相をランダムに加える
        if rng.random() > 0.5:
            qc.x(n)

    return qc.to_gate()


def run_deutsch_jozsa(n, oracle_type):
    """Deutsch-Jozsa アルゴリズムを実行"""
    qc = QuantumCircuit(n + 1, n)

    # Step 1: ancilla を |1⟩ にする
    qc.x(n)

    # Step 2: 全量子ビットに H を適用
    qc.h(range(n + 1))

    # Step 3: オラクルを適用
    oracle = deutsch_jozsa_oracle(n, oracle_type)
    qc.append(oracle, range(n + 1))

    # Step 4: 入力ビットに再び H
    qc.h(range(n))

    # Step 5: 入力ビットを測定
    qc.measure(range(n), range(n))

    return qc


# ==============================================================
# 実行
# ==============================================================
n_qubits = 4  # 4ビット入力 → 古典だと最悪 9 回、量子は 1 回

print("=" * 50)
print(f"  Deutsch-Jozsa Algorithm (n={n_qubits})")
print("=" * 50)
print(f"  古典計算: 最悪 {2**(n_qubits-1)+1} 回の関数評価が必要")
print(f"  量子計算: 1 回の関数評価で判定\n")

for oracle_type in ["constant_0", "constant_1", "balanced", "balanced_random"]:
    qc = run_deutsch_jozsa(n_qubits, oracle_type)

    result = simulator.run(qc, shots=1024).result()
    counts = result.get_counts()

    # 全ビットが 0 なら constant、それ以外なら balanced
    measurement = max(counts, key=counts.get)
    is_constant = all(b == "0" for b in measurement)

    verdict = "CONSTANT (定値)" if is_constant else "BALANCED (均等)"
    print(f"  Oracle: {oracle_type:20s} → 測定: {measurement} → 判定: {verdict}")

# 回路を表示
print(f"\n回路図 (oracle=balanced):")
qc_show = run_deutsch_jozsa(n_qubits, "balanced")
print(qc_show.draw(output="text"))

print("""
まとめ:
- Deutsch-Jozsa は量子の「干渉」を利用する
- 重ね合わせで全入力を同時に評価（量子並列性）
- 干渉で答えが |0...0⟩ に集中する（constant）か分散する（balanced）
- 指数的な高速化の最もクリーンな例

次: 05_grover_search.py で実用的な探索問題を解く
""")
