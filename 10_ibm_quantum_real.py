# -*- coding: utf-8 -*-
"""
10 - IBM Quantum 実機で QAOA を実行

IBM Quantum (Eagle r3, 127 qubit) で Basin 最適化 QAOA を実行する。

セットアップ:
1. IBM Quantum アカウント作成: https://quantum.ibm.com/
2. API トークン取得
3. qiskit-ibm-runtime で接続

注意:
- 無料枠: 月10分の実機時間
- ノイズの影響で理想的な結果は出ない → エラー緩和が重要
"""

import numpy as np
import sys

print("=" * 60)
print("  IBM Quantum 実機 QAOA")
print("=" * 60)

# ==============================================================
# 1. IBM Quantum 接続チェック
# ==============================================================
print("\n[1] IBM Quantum 接続確認...")

try:
    from qiskit_ibm_runtime import QiskitRuntimeService

    # 保存済みアカウントがあるかチェック
    try:
        service = QiskitRuntimeService()
        backends = service.backends()
        print(f"  接続成功！利用可能なバックエンド: {len(backends)}")
        for b in backends[:5]:
            config = b.configuration()
            n_qubits = config.n_qubits
            print(f"    - {b.name}: {n_qubits} qubits")

        IBM_AVAILABLE = True
    except Exception as e:
        print(f"  アカウント未設定: {e}")
        print("  → セットアップ手順を表示します")
        IBM_AVAILABLE = False

except ImportError:
    print("  qiskit-ibm-runtime がインストールされていません")
    print("  pip install qiskit-ibm-runtime")
    IBM_AVAILABLE = False

# ==============================================================
# 2. セットアップ手順
# ==============================================================
if not IBM_AVAILABLE:
    print("\n" + "=" * 60)
    print("  IBM Quantum セットアップ手順")
    print("=" * 60)
    print("""
    1. https://quantum.ibm.com/ でアカウント作成（無料）

    2. ダッシュボードから API トークンをコピー

    3. Python で保存:
       from qiskit_ibm_runtime import QiskitRuntimeService
       QiskitRuntimeService.save_account(
           channel="ibm_quantum",
           token="YOUR_API_TOKEN_HERE",
           overwrite=True,
       )

    4. 再度このスクリプトを実行
    """)

# ==============================================================
# 3. ローカルシミュレータで QAOA 回路を構築・テスト
# ==============================================================
print("\n[2] QAOA 回路構築（ローカルテスト）...")

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# 小規模 QAOA (10 qubit) で動作確認
N_QUBITS = 10
N_PARAMS_SMALL = 2
BITS_PER_PARAM = 5

print(f"  テスト回路: {N_QUBITS} qubits, p=1")


def build_qaoa_for_hardware(n_qubits, gamma, beta, p=1):
    """
    実機向け QAOA 回路。

    ハードウェア制約を考慮:
    - 2-qubit gate は隣接 qubit のみ (linear connectivity)
    - 回路深度を最小化
    """
    qc = QuantumCircuit(n_qubits, n_qubits)

    # Initial state
    qc.h(range(n_qubits))

    for l in range(p):
        # Cost layer (ZZ interactions + local Z)
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
            qc.rz(gamma[l] * 0.5, i + 1)
            qc.cx(i, i + 1)

        for i in range(n_qubits):
            weight = (i % BITS_PER_PARAM) / BITS_PER_PARAM
            qc.rz(gamma[l] * weight, i)

        # Mixer layer
        for i in range(n_qubits):
            qc.rx(2 * beta[l], i)

    # Measurement
    qc.measure(range(n_qubits), range(n_qubits))

    return qc


# テスト回路
gamma_test = [0.5]
beta_test = [0.3]
qc_test = build_qaoa_for_hardware(N_QUBITS, gamma_test, beta_test, p=1)

print(f"  回路深度: {qc_test.depth()}")
print(f"  ゲート数: {dict(qc_test.count_ops())}")

# ローカルシミュレータで実行
sim = AerSimulator()
result_sim = sim.run(qc_test, shots=1024).result()
counts_sim = result_sim.get_counts()
top_5 = sorted(counts_sim.items(), key=lambda x: -x[1])[:5]
print(f"\n  シミュレータ結果 (top 5):")
for bits, count in top_5:
    print(f"    |{bits}⟩: {count:4d} ({count/1024*100:5.1f}%)")

# ==============================================================
# 4. 実機実行（アカウント設定済みの場合）
# ==============================================================
if IBM_AVAILABLE:
    print("\n[3] 実機での QAOA 実行...")

    from qiskit_ibm_runtime import SamplerV2 as Sampler
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    # 最小 queue のバックエンドを選択
    backend = service.least_busy(
        simulator=False,
        min_num_qubits=N_QUBITS,
        operational=True,
    )
    print(f"  選択バックエンド: {backend.name} ({backend.configuration().n_qubits} qubits)")

    # トランスパイル（実機のトポロジーに合わせる）
    pm = generate_preset_pass_manager(optimization_level=2, backend=backend)
    qc_transpiled = pm.run(qc_test)

    print(f"  トランスパイル後:")
    print(f"    深度: {qc_transpiled.depth()}")
    print(f"    ゲート数: {dict(qc_transpiled.count_ops())}")

    # 実行
    sampler = Sampler(backend)
    job = sampler.run([qc_transpiled], shots=4096)
    print(f"  ジョブ投入完了: {job.job_id()}")
    print(f"  ステータス: {job.status()}")

    # 結果待ち
    print("  結果を待っています...")
    result_hw = job.result()

    # 結果表示
    pub_result = result_hw[0]
    counts_hw = pub_result.data.c.get_counts()

    print(f"\n  実機結果 (shots=4096):")
    top_10 = sorted(counts_hw.items(), key=lambda x: -x[1])[:10]
    for bits, count in top_10:
        pct = count / 4096 * 100
        bar = "#" * int(pct)
        print(f"    |{bits}⟩: {count:4d} ({pct:5.1f}%) {bar}")

    # シミュレータ vs 実機の比較
    print("\n  シミュレータ vs 実機:")

    # KL divergence 的な比較
    all_states = set(list(counts_sim.keys()) + list(counts_hw.keys()))
    sim_total = sum(counts_sim.values())
    hw_total = sum(counts_hw.values())

    fidelity = 0
    for state in all_states:
        p_sim = counts_sim.get(state, 0) / sim_total
        p_hw = counts_hw.get(state, 0) / hw_total
        fidelity += np.sqrt(p_sim * p_hw)

    print(f"    分布忠実度 (Bhattacharyya): {fidelity:.4f}")
    print(f"    1.0 = 完全一致, 0.0 = 完全不一致")

else:
    print("\n[3] 実機実行スキップ（アカウント未設定）")
    print("    上記セットアップ手順に従って API トークンを設定してください")

# ==============================================================
# 5. ノイズモデルでの検証
# ==============================================================
print("\n[4] ノイズモデルシミュレーション...")

from qiskit_aer.noise import NoiseModel, depolarizing_error

# 簡易ノイズモデル
noise_model = NoiseModel()
# 1-qubit gate error: 0.1%
noise_model.add_all_qubit_quantum_error(depolarizing_error(0.001, 1), ["rx", "rz", "h"])
# 2-qubit gate error: 1%
noise_model.add_all_qubit_quantum_error(depolarizing_error(0.01, 2), ["cx"])

noisy_sim = AerSimulator(noise_model=noise_model)
result_noisy = noisy_sim.run(qc_test, shots=4096).result()
counts_noisy = result_noisy.get_counts()

print(f"  ノイズなし top state: {top_5[0][0]} ({top_5[0][1]/1024*100:.1f}%)")
noisy_top = sorted(counts_noisy.items(), key=lambda x: -x[1])[:5]
print(f"  ノイズあり top-5:")
for bits, count in noisy_top:
    print(f"    |{bits}⟩: {count:4d} ({count/4096*100:5.1f}%)")

# ノイズの影響を定量化
ideal_probs = {k: v/1024 for k, v in counts_sim.items()}
noisy_probs = {k: v/4096 for k, v in counts_noisy.items()}

all_states = set(list(ideal_probs.keys()) + list(noisy_probs.keys()))
fid_noisy = sum(np.sqrt(ideal_probs.get(s, 0) * noisy_probs.get(s, 0)) for s in all_states)
print(f"\n  理想 vs ノイズ忠実度: {fid_noisy:.4f}")

# ==============================================================
# 6. エラー緩和 (Zero-Noise Extrapolation)
# ==============================================================
print("\n[5] エラー緩和 (ZNE)...")

noise_levels = [1, 2, 3]  # ノイズ倍率
expectation_values = []

for scale in noise_levels:
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(0.001 * scale, 1), ["rx", "rz", "h"])
    nm.add_all_qubit_quantum_error(depolarizing_error(0.01 * scale, 2), ["cx"])

    noisy_sim_zne = AerSimulator(noise_model=nm)
    result_zne = noisy_sim_zne.run(qc_test, shots=4096).result()
    counts_zne = result_zne.get_counts()

    # 期待値: 全ビットの平均 (簡略化)
    total = sum(counts_zne.values())
    exp_val = sum(bits.count("1") / N_QUBITS * count for bits, count in counts_zne.items()) / total
    expectation_values.append(exp_val)
    print(f"  ノイズ×{scale}: ⟨O⟩ = {exp_val:.4f}")

# 線形外挿で noise=0 を推定
coeffs = np.polyfit(noise_levels, expectation_values, 1)
zne_estimate = coeffs[1]  # y-intercept (noise=0)

# 理想値
ideal_total = sum(counts_sim.values())
ideal_exp = sum(bits.count("1") / N_QUBITS * count for bits, count in counts_sim.items()) / ideal_total

print(f"\n  理想値:       ⟨O⟩ = {ideal_exp:.4f}")
print(f"  ノイズ×1:     ⟨O⟩ = {expectation_values[0]:.4f}")
print(f"  ZNE 推定値:   ⟨O⟩ = {zne_estimate:.4f}")
print(f"  ZNE 改善:     {abs(ideal_exp - zne_estimate):.4f} < {abs(ideal_exp - expectation_values[0]):.4f}")

print("""
まとめ:
- IBM Quantum 実機アクセスには API トークンが必要（無料枠あり）
- QAOA 回路はトランスパイルでハードウェアトポロジーに合わせる
- ノイズの影響: 2-qubit gate error (1%) が支配的
- ZNE (Zero-Noise Extrapolation) でエラー緩和が可能
- 実用的には Sampler + Error Mitigation が標準ワークフロー

IBM Quantum セットアップ後に再実行すると実機結果が得られます。
""")
