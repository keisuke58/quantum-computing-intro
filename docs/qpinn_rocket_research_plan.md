# QPINN × 固体変形（ロケット構造）研究構想

量子物理情報ニューラルネットワーク（QPINN）の現状整理と、2027年5月までの研究計画
（KQCC系ワークショップ発表想定）

作成: 2026-10-01 ／ Keisuke（Keio × LUH）

---

## 要点

- QPINN は**シミュレータ上で小規模 PDE を古典 PINN 並みの精度・より少ないパラメータで解ける**段階。
  計算速度・精度で古典を超えた例はまだない。
- 2D 問題は解けている（Maxwell、lid-driven cavity、Helmholtz など）。ただし
  **固体力学（弾性・塑性）への QPINN 適用はほぼ空白**。
- 提案: **ロケット構造**を題材に、厚肉円筒（Lamé）→ 固体ロケットモータの星形グレイン（応力集中）
  → 推進剤弾性率のベイズ同定（TMCMC）、の3段構成。
- 主張は「古典超え」ではなく、**固体力学への初適用・表現力の限界の定量化・ベイズ逆解析との統合**に置く。

---

## 1. QPINN の現状（2026年時点）

### できていること

- **対象 PDE**: Burgers、熱方程式、Helmholtz、Klein-Gordon、移流拡散、2D 時間依存 Maxwell、
  2D lid-driven cavity（非線形 NS）など。
- **主な成果 = パラメータ効率**: 古典 PINN と同程度の精度・収束を、より少ない学習パラメータで
  達成という主張が中心（QCPINN 等）。
- **アーキテクチャの工夫**: 学習可能なエンコーディング（FNN/PQC による trainable embedding）、
  対称性を回路に埋め込む Geometric QPINN、エネルギー保存則のペナルティ（Maxwell）。
- **ツール**: PyTorch ベースの GPU 量子シミュレータ（TorQ 等）、古典/量子 PINN 統合の
  PINNACLE（multi-GPU）。

### 限界

- ほぼ全てが**古典シミュレータ上**（4〜10 qubit 程度）。実機での本格的な結果はほぼない。
- **Barren plateau**（qubit 数・深さに対して勾配が指数的に消失）。
- 実機では backprop 不可 → parameter-shift 則で評価回数が増え、**高階微分ほど高コスト**。
  shot noise が PDE 残差を汚す。
- 精度は相対誤差 1e-2〜1e-3 程度が多く、古典 PINN のベスト（≲1e-4）に届かないことが多い。
- 公平な比較（同等の古典モデル、特に Fourier feature PINN との比較）が少ない。

---

## 2. 古典 PINN との違い

PDE 残差＋境界条件の loss を最小化する枠組みは同じ。違うのは
**関数近似器（u(x) を出す部分）が NN か量子回路か**だけ。

```
古典PINN:  x, y → MLP(θ) → u(x) → 自動微分で ∂u, ∂²u → 残差 loss

QPINN:     x, y → エンコード R(x) → 変分量子回路 U(θ), n qubit → 測定 ⟨Z⟩ → u(x)
                                  ※実機: parameter-shift / シミュレータ: 自動微分
```

多くは前後に小さな古典 NN を挟むハイブリッド構成
（古典 NN で trainable embedding → PQC → 古典デコーダ）。

| 観点 | 古典 PINN | QPINN |
|---|---|---|
| 関数クラス | MLP（汎用近似、スペクトルバイアスあり） | 角度エンコードでは実質 x のフーリエ級数。周波数はエンコード回数で決まり、係数を回路が学習 → 滑らかな解は得意、局所的急変は苦手 |
| パラメータ数 | 多い（数千〜数万） | 少ない（状態空間は 2ⁿ 次元） |
| 微分 | 自動微分で安価 | 実機では parameter-shift（1パラメータあたり回路2回評価）、高階で爆発 |
| ノイズ | なし | shot noise・ゲート誤差（実機） |
| 学習の難しさ | loss 不均衡、スペクトルバイアス | barren plateau |

---

## 3. 古典を超えているか？

**結論: 現状 No。** 古典で解けない問題は扱っておらず、精度・時間・メモリで古典を上回った例はない。

- シミュレータで回している時点で古典で解ける規模。優位の主張はパラメータ数のみで、
  シミュレーションコストは 2ⁿ で増える。
- **Dequantization**: 角度エンコードの変分回路はフーリエ級数モデルなので、古典のフーリエ特徴量
  モデルで近似できる（classical surrogate）という指摘がある。
- 理論上の優位候補は、HHL 系線形ソルバー（誤り耐性量子計算機前提、解の全読み出しで加速消失）、
  超高次元 PDE、量子ネイティブ問題（Schrödinger 方程式等）。いずれも QPINN で実証済みではない。

**研究上の含意**: 「古典超え」は主張しない。新分野への初適用・表現力とパラメータ効率の定量評価・
古典 Fourier feature PINN との公平比較で論文を組むのが誠実で通りやすい。

---

## 4. 2D 問題と固体力学の空白

### 2D は解ける（条件付き）

- 2D 時間依存 Maxwell、2D lid-driven cavity、2D Helmholtz 等で実績あり。
- 条件: 数 qubit・シミュレータ・矩形ドメイン・滑らかな解。
- 穴あき・き裂・応力集中など複雑形状はほぼ未検証。

### 固体力学 QPINN はほぼ手つかず

- QPINN のベンチマークは流体・波動・熱に偏っている。
- 古典 PINN では線形弾性〜von Mises 弾塑性、動的弾性の材料同定まで確立済み
  （Haghighat et al. 2020 ほか）。
- → **「固体力学初の QPINN ベンチマーク」に十分な新規性。**

### 固体で QPINN が難しい理由（＝評価すべき論点）

- **ベクトル場・多出力**: 変位 u, v（＋応力）を同時出力 → 少数 qubit の読み出し設計が課題
  （観測量を分ける or 回路を複数本）。
- **高階微分**: Navier 方程式は2階、板曲げは4階。
- **応力集中・境界層**: 局所的急勾配はフーリエ級数的な関数クラスが苦手。

支配方程式（線形弾性 Navier）:

```
μ ∇²u + (λ+μ) ∇(∇·u) + f = 0
```

---

## 5. 研究対象案: ロケット構造

全て軸対称 or 2D 断面に落とせるため、現行 QPINN の規模（数 qubit）に合う。

| # | 対象 | 内容 | 参照解 | 役割 |
|---|---|---|---|---|
| 1 | 厚肉円筒の内圧（Lamé） | 燃焼室・推進剤タンクの基本形。軸対称で r のみの問題、2D 平面ひずみ断面でも可 | Lamé 解析解 | **確実** 最初のベンチ |
| 2 | ノズル壁の熱応力 | 内面高温・外面低温の半径方向温度勾配による熱弾性。熱伝導→変形の連成にすればマルチフィジックス化 | 厚肉円筒熱応力の解析解 | 拡張 |
| 3 | 固体ロケットモータの星形グレイン | 星形内孔の推進剤が内圧＋熱収縮で変形、星先端の応力集中→き裂が実際の故障モード | FEM（参照解） | **本命** 弱点評価 |
| 4 | COPV（CFRP 巻き圧力容器） | 異方性弾性の厚肉円筒。CFRP-GNN 研究と接続 | Lekhnitskii 解 | 拡張 |
| 5 | 推進剤弾性率の同定 × TMCMC | 推進剤は温度・経年（エージング）で特性変化 → 表面ひずみ計測から E, ν の事後分布を推定。QPINN をサロゲートに | 合成データ | **独自性** 先行例ほぼなし |

### Lamé 解（ベンチ#1 の検証用）

内半径 a、外半径 b、内圧 pᵢ、外圧 0 のとき:

```
σ_r = A − B/r²,   σ_θ = A + B/r²
A = pᵢa² / (b² − a²),   B = pᵢa²b² / (b² − a²)
```

### 推奨ストーリー: #1 → #3 → #5

「2D/軸対称で解ける（Lamé）」→「QPINN の弱点＝応力集中を定量化（星形グレイン）」→
「ベイズ逆解析と組み合わせて使いどころを示す（推進剤 E, ν 同定）」で一本筋が通る。
5月までに #1 と #3 は確実に、#5 は小規模でも1ケース入れられれば発表として強い。

### 比較・評価設計

- **ベースライン**: 古典 PINN（MLP）、**Fourier feature PINN**（査読で必ず聞かれる）、FEM。
- **評価指標**: 変位・応力の相対 L2 誤差、応力集中係数の再現精度、パラメータ数、学習時間、収束安定性。
- **アブレーション**: qubit 数、回路深さ、エンコード方式（角度／trainable embedding）、
  出力設計（u, v の読み出し方法）。
- **実装**: PyTorch ＋ GPU 量子シミュレータ（PennyLane / TorchQuantum / CUDA-Q から選定）。
  可能なら KQCC 経由の IBM 実機で小問題を1ケース。

---

## 6. スケジュール（2026年10月 → 2027年5月）

| タスク | 10月 | 11月 | 12月 | 1月 | 2月 | 3月 | 4月 | 5月 |
|---|---|---|---|---|---|---|---|---|
| 文献整理・シミュレータ選定 | ■ | | | | | | | |
| 古典 PINN で Lamé・平面ひずみベンチ | ■ | ■ | | | | | | |
| QPINN 実装（1D 軸対称 → 2D） | | ■ | ■ | | | | | |
| **#1 Lamé 厚肉円筒 完了** | | | ★ | | | | | |
| #3 星形グレイン（FEM 参照解作成含む） | | | | ■ | ■ | | | |
| アブレーション・Fourier feature 比較 | | | | | ■ | ■ | | |
| #5 E, ν 同定 × TMCMC | | | | | | ■ | ■ | |
| IBM 実機で小問題（任意） | | | | | | ◇ | ◇ | |
| **結果確定・図表・アブスト** | | | | | | | | ★ |

■ 作業期間 ／ ★ マイルストーン ／ ◇ 任意

---

## 7. 注意点・TODO

- **発表先の確認**: KQCC は延世大・台湾大・慶應の3大学合同ワークショップを8月に開催している
  （2023年は 8/24–25）。2027年の日程・アブスト締切は山本研 or KQCC 事務局に確認。
- **修論との両立**: LUH 修論（TMCMC × バイオフィルム）の締切を先に固め、QPINN はサブ研究として
  週の配分を決める。#5 で TMCMC 資産を共有すると効率的。
- **実機アクセス**: KQCC の IBM Q Hub アカウント取得方法を早めに確認。
- **主張の範囲**: 「古典超え」は書かない。parameter efficiency の主張にはシミュレーションコスト
  （2ⁿ）も併記。
- **先行研究チェック**: 2026年8月の台湾イベント等で GPU 量子シミュレータ×固体変形の発表がないか
  確認（あれば直接の先行例）。

---

## 参考文献

1. QPINN with Quantum Trainable Embeddings for the Lid-Driven Cavity Problem, arXiv:2605.13892 (2026).
2. Quantum-Assisted Trainable-Embedding PINNs for Parabolic PDEs, arXiv:2602.14596 (2026).
3. Geometric Quantum Physics-Informed Neural Network (GQPINN), arXiv:2605.02352 (2026).
4. Chen et al., QPINNs for Maxwell's Equations: Circuit Design, Barren Plateaus Mitigation, and GPU Acceleration, arXiv:2506.23246; Quantum Mach. Intell. 8, 21 (2026).
5. Farea et al., QCPINN: Quantum-Classical PINNs for Solving PDEs, arXiv:2503.16678.
6. Panichi et al., QPINNs for multivariable PDEs (CV quantum computing), Phys. Rev. Applied 25 (2026); arXiv:2503.12244.
7. Klement, Eyring, Schwabe, Explaining the advantage of quantum-enhanced PINNs, arXiv:2601.15046 (2026).
8. PINNACLE: Open-Source Framework for Classical and Quantum PINNs, arXiv:2604.15645 (2026).
9. Haghighat et al., A deep learning framework for solution and discovery in solid mechanics, arXiv:2003.02751.
10. Kag & Gopinath, PINN for modeling dynamic linear elasticity, arXiv:2312.15175.
11. 慶應義塾大学 量子コンピューティングセンター 2023年度事業報告, keio.ac.jp/ja/org/kgri/research-centers/2024/A24-05-2/
