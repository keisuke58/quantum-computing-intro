# -*- coding: utf-8 -*-
"""
03 - 量子もつれ (Entanglement)

量子もつれ = 2つ以上の量子ビットが切り離せない相関を持つ状態。
Bell状態が最も基本的なもつれ状態:

|Φ+⟩ = (|00⟩ + |11⟩) / √2

片方を測定すると、もう片方の結果が即座に決まる。
（「怖い遠隔作用」とアインシュタインが呼んだやつ）
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector

simulator = AerSimulator()

# ==============================================================
# 1. Bell状態を作る
# ==============================================================
print("=" * 50)
print("  1. Bell 状態 |Φ+⟩ の生成")
print("=" * 50)

bell = QuantumCircuit(2, 2)
bell.h(0)       # qubit 0 を重ね合わせ
bell.cx(0, 1)   # CNOT: もつれさせる
bell.measure([0, 1], [0, 1])

print(bell.draw(output="text"))

result = simulator.run(bell, shots=1024).result()
counts = result.get_counts()
print(f"測定結果: {counts}")
print("→ '00' と '11' だけが出る（'01' や '10' は出ない！）")
print("→ 2つのビットは常に同じ値になる = もつれている\n")

# ==============================================================
# 2. 4つの Bell 状態
# ==============================================================
print("=" * 50)
print("  2. 4つの Bell 状態")
print("=" * 50)

bell_states = {
    "|Φ+⟩": [],           # H, CX
    "|Φ-⟩": ["z"],        # H, CX, Z
    "|Ψ+⟩": ["x"],        # H, CX, X
    "|Ψ-⟩": ["x", "z"],   # H, CX, X, Z
}

for name, gates in bell_states.items():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    for g in gates:
        if g == "x":
            qc.x(0)
        elif g == "z":
            qc.z(0)
    sv = Statevector(qc)
    print(f"  {name} = {sv.draw('latex_source')}")

# ==============================================================
# 3. もつれの「測定相関」を確認
# ==============================================================
print("\n" + "=" * 50)
print("  3. もつれの相関確認（10000 shots）")
print("=" * 50)

qc_corr = QuantumCircuit(2, 2)
qc_corr.h(0)
qc_corr.cx(0, 1)
qc_corr.measure([0, 1], [0, 1])

result = simulator.run(qc_corr, shots=10000).result()
counts = result.get_counts()

total = sum(counts.values())
for state, count in sorted(counts.items()):
    pct = count / total * 100
    bar = "#" * int(pct / 2)
    print(f"  |{state}⟩: {count:5d} ({pct:5.1f}%) {bar}")

# 相関チェック
correlated = sum(v for k, v in counts.items() if k[0] == k[1])
print(f"\n  同じ値の割合: {correlated/total*100:.1f}%")
print("  → もつれ状態では 100% 相関する")

# ==============================================================
# 4. もつれ vs ただの重ね合わせ
# ==============================================================
print("\n" + "=" * 50)
print("  4. もつれ vs 独立な重ね合わせ")
print("=" * 50)

# 独立な重ね合わせ（もつれなし）
qc_ind = QuantumCircuit(2, 2)
qc_ind.h(0)
qc_ind.h(1)  # 各ビットを独立に重ね合わせ
qc_ind.measure([0, 1], [0, 1])

result_ind = simulator.run(qc_ind, shots=10000).result()
counts_ind = result_ind.get_counts()

print("独立な重ね合わせ H|0⟩ ⊗ H|0⟩:")
for state, count in sorted(counts_ind.items()):
    pct = count / 10000 * 100
    print(f"  |{state}⟩: {count:5d} ({pct:5.1f}%)")
print("→ 4つの状態が均等に出る（相関なし）")

print("""
まとめ:
- H + CNOT で Bell 状態（もつれ）を作れる
- もつれた量子ビットは測定結果が 100% 相関する
- これは古典的な「コインを2枚用意」とは本質的に違う
  （Bell の不等式で証明可能）
- 量子テレポーテーション・量子暗号の基盤技術

次: 04_deutsch_jozsa.py で量子アルゴリズムの威力を体感
""")
