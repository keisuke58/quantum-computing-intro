# 量子コンピューティング プロジェクト案

既存の研究（バイオフィルム TMCMC, FEM）との接点を活かした提案。

---

## A. 量子最適化 (Optimization)

### A1. QAOA でバイオフィルムパラメータ最適化
- **概要**: TMCMC の 20 パラメータ空間を QAOA で探索
- **面白い点**: 多峰性ランドスケープ（Basin sensitivity で確認済み）に量子トンネル効果が効くか？
- **難易度**: ★★★☆☆
- **必要**: `qiskit-optimization`, `qiskit-algorithms`
- **スケジュール**: 2-3 週間

### A2. 量子アニーリング × FEM メッシュ最適化
- **概要**: 3D conformal mesh のノード配置を量子アニーリングで最適化
- **面白い点**: 産総研の「トラス構造最適化」と同系統、FEM との直接接続
- **難易度**: ★★★★☆
- **必要**: `dwave-ocean-sdk` or QAOA エミュレーション

---

## B. 量子ベイズ推論 (Bayesian)

### B1. 量子カーネル × ベイズ最適化 (BO)
- **概要**: DeepONet サロゲート + 量子カーネル BO でハイパーパラメータ探索
- **面白い点**: 古典 BO と比較して高次元（20D）での収束速度を検証
- **難易度**: ★★★☆☆
- **必要**: `pennylane`, `scikit-optimize`
- **スケジュール**: 1-2 週間

### B2. Quantum-enhanced MCMC
- **概要**: 量子ウォークベースの proposal で TMCMC の混合を加速
- **面白い点**: 多峰性 posterior（dh_baseline の bimodal DI）での性能
- **難易度**: ★★★★★（理論的にもオープン）
- **必要**: `qiskit`, カスタム実装

---

## C. 量子機械学習 (QML)

### C1. 変分量子回路 (VQC) サロゲートモデル 【おすすめ】
- **概要**: DeepONet の代わりに VQC で ODE サロゲートを構築
- **面白い点**: 既存の `quantum_surrogate.py` を拡張、DeepONet と精度・速度を比較
- **難易度**: ★★☆☆☆（既にプロトタイプあり）
- **必要**: `pennylane` (自動微分が強い) or `qiskit-machine-learning`
- **スケジュール**: 1 週間

### C2. 量子カーネル PCA × バイオフィルム条件分類
- **概要**: 4 条件（CS/CH/DH/DS）の posterior 分布を量子カーネルで分類
- **面白い点**: 古典 PCA vs 量子 PCA の分離度を比較（既に `quantum_kernel_pca.py` あり）
- **難易度**: ★★☆☆☆
- **必要**: `qiskit-machine-learning`
- **スケジュール**: 3-5 日

### C3. 量子リザバーコンピューティング × 時系列予測
- **概要**: バイオフィルム成長の時系列データを量子リザバーで予測
- **面白い点**: 少データ（各条件 ~50 時点）での性能、古典 LSTM と比較
- **難易度**: ★★★☆☆
- **必要**: `pennylane`

---

## D. 量子シミュレーション (Simulation)

### D1. VQE × 材料パラメータ推定 【産総研テーマに近い】
- **概要**: E(DI) 構成則のパラメータを変分量子固有値問題として定式化
- **面白い点**: 産総研の「一般化固有値問題 × FEM」論文と直接関連
- **難易度**: ★★★★☆
- **必要**: `qiskit-algorithms`, `qiskit-nature`

### D2. 量子 ODE ソルバー × Hamilton 系
- **概要**: 5 種バイオフィルム Hamilton ODE を量子回路で解く
- **面白い点**: 産総研の「量子非線形微分方程式」と同系統
- **難易度**: ★★★★★
- **必要**: カスタム実装、HHL ベース or 変分法

---

## 推奨ロードマップ

```
Week 1:  C1 (VQC サロゲート) + C2 (量子カーネル PCA)
         → 既存コードの拡張、すぐに結果が出る

Week 2:  A1 (QAOA パラメータ最適化)
         → 最適化の量子的アプローチ

Week 3:  B1 (量子カーネル BO)
         → ベイズとの融合

Week 4:  チュートリアル整備 + 結果まとめ
         → リポジトリを公開可能な状態に
```

## 環境

現在インストール済み:
- `qiskit 2.3.0` + `qiskit-aer 0.17.2` (CPU シミュレータ)
- `qiskit-ibm-runtime` (IBM Quantum 実機アクセス)
- `qiskit-algorithms` (VQE, QAOA, Grover 等)
- `qiskit-machine-learning` (VQC, Quantum Kernel)
- `qiskit-optimization` (QAOA, MinimumEigenOptimizer)
- `pennylane 0.44.1` (自動微分、ハードウェア非依存)
- `cirq 1.6.1` (Google 量子フレームワーク)
