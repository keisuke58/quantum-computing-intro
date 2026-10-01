# Quantum Computing 入門

量子コンピューティング初心者のための学習リポジトリ。
Qiskit を使って、量子ゲート・量子回路・量子アルゴリズムを段階的に学べます。

## 対象者

- プログラミング（Python）の基礎がある人
- 量子コンピューティングに興味があるけど何から始めればいいかわからない人
- 線形代数の基礎（行列・ベクトル）をなんとなく知っている人

## セットアップ

```bash
pip install -r requirements.txt
```

## コンテンツ

| # | ファイル | 内容 |
|---|---------|------|
| 1 | `01_qubit_basics.py` | 量子ビット、重ね合わせ、測定 |
| 2 | `02_quantum_gates.py` | 基本ゲート（X, H, CNOT）と回路の可視化 |
| 3 | `03_entanglement.py` | Bell状態、量子もつれ、非局所性 |
| 4 | `04_deutsch_jozsa.py` | Deutsch-Jozsaアルゴリズム（量子の優位性を初体験） |
| 5 | `05_grover_search.py` | Groverの探索アルゴリズム（√N 高速化） |
| 6 | `06_vqe_intro.py` | 変分量子固有値ソルバー（VQE）入門 |
| 7 | `07_quantum_teleportation.py` | 量子テレポーテーション |

### 応用編（バイオフィルム × 量子）

| # | ファイル | 内容 |
|---|---------|------|
| 8 | `08_qaoa_basin_optimization.py` | QAOA × 多峰性 Basin 最適化（12 qubit） |
| 9 | `09_quantum_kernel_bo.py` | 量子カーネル × ベイズ最適化（20D パラメータ） |
| 10 | `10_ibm_quantum_real.py` | IBM Quantum 実機 QAOA + ノイズ緩和 |

### 研究編（QPINN × 固体力学）

| # | ファイル | 内容 |
|---|---------|------|
| 11 | `11_qpinn_lame.py` | QPINN ベンチ #1: 厚肉円筒（Lamé 解）。古典 PINN / Fourier feature PINN / QPINN の比較 |
| 12 | `12_qpinn_ablation.py` | QPINN アブレーション: 表現力（教師ありフィット）と最適化可能性（PDE 学習）を分離 |

```bash
python 11_qpinn_lame.py              # 単一シードで3モデル比較
python 11_qpinn_lame.py --seeds 5    # 古典2モデルを5シードで平均±標準偏差
python 12_qpinn_ablation.py          # 回路構成を振るスイープ（CPU で約1時間）
python 12_qpinn_ablation.py --quick  # 構成を減らした短縮版
```

必要: `torch`, `pennylane`

各ファイルは **そのまま実行可能** で、コメントで解説付きです。

## 実行

```bash
python 01_qubit_basics.py
python 02_quantum_gates.py
# ...
```

## ドキュメント

- [docs/research_landscape_2025.md](docs/research_landscape_2025.md) — 量子コンピューティング研究の最新動向
- [docs/project_ideas.md](docs/project_ideas.md) — プロジェクト案とロードマップ
- [docs/qpinn_rocket_research_plan.md](docs/qpinn_rocket_research_plan.md) — QPINN × 固体変形（ロケット構造）研究構想（2026/10–2027/5）
- [docs/qpinn_plan_review.md](docs/qpinn_plan_review.md) — 上記計画のレビュー・文献確認・ベンチ #1 の実測結果
- [docs/qpinn_ablation_results.md](docs/qpinn_ablation_results.md) — 回路構成のアブレーション結果（表現力 vs 最適化可能性）

## 参考リソース

- [Qiskit Textbook](https://learning.quantum.ibm.com/)
- [IBM Quantum](https://quantum.ibm.com/)
- [Quantum Computing: An Applied Approach (Hidary)](https://link.springer.com/book/10.1007/978-3-030-83274-2)

## License

MIT
