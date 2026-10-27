# QPINN 研究計画のレビューと文献確認

対象: `docs/qpinn_rocket_research_plan.md`（2026-10-01 版）
作成: 2026-10-01

---

## 1. 計画の強いところ

- **主張の置き方が正しい。** 「古典超え」を主張しないという判断は、この分野の現状
  （シミュレータ上・数 qubit・古典で解ける規模）に照らして唯一通る立て方。査読で最初に
  突かれるのがそこなので、先に閉じているのは良い。
- **題材の選び方が妥当。** 厚肉円筒は軸対称で 1D に落ち、解析解（Lamé）があり、しかも
  ロケット構造の実体がある。「ベンチマークのためのベンチマーク」に見えない。
- **空白の主張が裏付けられる。** 下記 §3 の通り、QPINN × 固体力学は文献上ほぼ空白で、
  新規性の主張は成立する。
- **Fourier feature PINN をベースラインに入れている。** 角度エンコード VQC が実質
  フーリエ級数である以上、ここを外すと dequantization の指摘で論文が落ちる。入っているのは重要。

## 2. 弱点・先に潰すべき点

### 2.1 「空白＝新規性」は弱い主張になりうる

固体力学 QPINN が無いのは、誰もやっていないからではなく**やっても勝てないから**という
可能性がある。査読者もそう読む。対策は、空白を埋めること自体ではなく
**「なぜ難しいかを定量化した」**を主成果に据えること:

- 応力集中係数の再現誤差が qubit 数・エンコード周波数に対してどうスケールするか
- 同じ関数クラスを持つ古典 Fourier feature PINN との差がどこから来るか

この形なら「負の結果」でも論文になる。計画の §4「固体で QPINN が難しい理由」を
**結論ではなく測定対象**として前に出すべき。

### 2.2 ベンチ #1 で既に出た実測上の落とし穴（本リポジトリ `11_qpinn_lame.py` で確認）

最小実装を走らせた段階で、計画に書かれていない実務的な障害が2つ出た。どちらも
#3 星形グレインでさらに悪化するので、先に対策を決めておくべき。

1. **エンコード周波数を上げると PDE 損失が壊れる。**
   qubit ごとに (w+1) 倍の角度を入れる「周波数を稼ぐ」定番の設計にすると、初期状態で
   u'' が O(10) になり、PDE 残差の最小化が「高周波成分の抑制」に支配されて自明解に
   落ちる（rel L2 ≳ 1）。全 qubit 同一角度の低周波エンコードにすると収束した。
   → **表現力（周波数）と最適化可能性のトレードオフ**が実在する。これ自体が測定対象。

2. **残差の書き方で条件数が変わる。**
   `u'' + u'/r − u/r²` をそのまま使うより、Euler 型 `r²u'' + ru' − u` にした方が
   古典・量子どちらも安定した。論文に出す時はこの正規化を明記しないと再現できない。

3. **出力スケーリングの初期値が効く。** `u = w·⟨Z⟩ + c` の c を解の平均値付近に
   初期化しないと初期損失が桁で跳ね、収束が遅い。ハイパラとして記録すべき。

### 2.3 評価指標に応力（微分量）を必ず入れる

変位 u の相対 L2 だけを見ると QPINN は実際より良く見える。応力は u の1階微分なので
誤差が増幅する。計画の「応力集中係数の再現精度」は正しいが、**#1 の段階から σ_r の
相対 L2 を併記**しておくと #3 への接続が自然になる。`11_qpinn_lame.py` は両方出す。

### 2.4 スケジュールの現実性

- 10–11月に「古典 PINN ベンチ」は妥当。ただし **QPINN 実装（11–12月）は 1D 軸対称で
  2週間、2D ベクトル場で1ヶ月**を見た方がいい。多出力（u, v）の読み出し設計は
  計画 §4 が指摘する通り未解決で、ここが最大のリスク。
- **#3 の FEM 参照解作成を 1月に置いているのは遅い。** 星形グレインのメッシュと
  FEM 解は QPINN とは独立に作れるので、**11–12月に前倒し**して、1月は QPINN 側に
  専念できる形にすべき。FEM が間に合わないと #3 全体が落ちる。
- #5（TMCMC）は修論資産の再利用が前提なので、**修論側のコードが固まる時期**に
  依存する。修論の締切を先に確定させるという計画の判断は正しい。
- IBM 実機（任意）は、parameter-shift での2階微分コストを考えると #1 の 1D 問題
  以外は現実的でない。**「1D Lamé を実機で1ケース」に限定**して期待値を下げておく。

### 2.5 リスクと撤退ライン

| リスク | 兆候 | 代替 |
|---|---|---|
| 2D ベクトル場の読み出しが設計できない | 1月末時点で u, v 同時出力が動かない | 1D 軸対称のみで深掘り（#1 + #2 熱応力）に縮退 |
| 星形グレインで QPINN が全く収束しない | 応力集中部で rel L2 > 0.5 が改善しない | それ自体を「表現力の限界」として報告（負の結果として出す） |
| FEM 参照解が間に合わない | 12月末にメッシュ未完 | 円孔付き板など解析解のある応力集中問題に差し替え |
| 先行研究が出る | 文献チェックで直接の先行例 | ベイズ逆解析（#5）に主軸を移す。こちらは先行例がほぼない |

## 3. 文献確認の結果（2026-10-01 時点）

計画の参考文献のうち、ウェブ検索で実在を確認できたもの・できなかったものを分けた。
**2026年の arXiv 番号を持つ項目は原稿に載せる前に必ず現物確認すること。**

### 確認できたもの

| 文献 | 状況 |
|---|---|
| Farea, Khan, Celebi, *QCPINN: Quantum-Classical PINNs for Solving PDEs*, arXiv:2503.16678 | **実在**。Istanbul Technical University + Rutherford Appleton Lab。古典 PINN の 10–30% の学習パラメータで同等の精度・収束、Helmholtz / Klein-Gordon / 移流拡散で相対誤差 4–64% 改善。計画の「パラメータ効率が主な成果」という整理と一致。v6 まで改訂されている |
| Chen, Shaviner, Chandravamsi, Pisnoy, Frankel, Pereg, *QPINNs for Maxwell's Equations*, arXiv:2506.23246 | **実在**。Technion。2D 時間依存 Maxwell、エネルギー保存則を損失に入れる、"black hole" barren plateau という新種の損失地形を報告、PyTorch ベースの GPU 量子シミュレータを自作。計画の記述と一致 |
| QPINN for multivariable PDEs, arXiv:2503.12244 | **実在**（連続変数量子計算） |
| Haghighat et al., *A deep learning framework for solution and discovery in solid mechanics*, arXiv:2003.02751 | **実在**。古典 PINN 側の固体力学の基準文献 |
| Kag & Gopinath, *PINN for modeling dynamic linear elasticity*, arXiv:2312.15175 | **実在**（SSRN 版も確認） |
| Trainable embedding QPINN | **実在**。ただし計画が挙げる arXiv 番号ではなく、*Trainable embedding quantum physics informed neural networks for solving nonlinear PDEs*, Scientific Reports (2025), s41598-025-02959-z として出版済み。**引用はこちらの出版版に差し替えるべき** |
| PINNACLE（古典/量子 PINN 統合フレームワーク） | 言及を確認（arXiv:2604.15645 として参照されている） |

### 現物確認が必要なもの

計画が 2026年の arXiv 番号で挙げている以下は、検索では一次情報に到達できなかった。
番号の取り違え・プレプリント段階・記憶違いのいずれもありうる。

- arXiv:2605.13892（Lid-Driven Cavity の QPINN with Quantum Trainable Embeddings）
- arXiv:2602.14596（Parabolic PDE の Quantum-Assisted Trainable-Embedding PINN）
  — 上記 Scientific Reports 版と同一内容の可能性が高い
- arXiv:2605.02352（Geometric QPINN / GQPINN）
- arXiv:2601.15046（Klement, Eyring, Schwabe）
- arXiv:2604.15645（PINNACLE）
- Panichi et al., Phys. Rev. Applied 25 (2026)

**TODO**: arXiv の listing で著者名から直接引き当て、番号・タイトル・年を原稿に写す前に確定する。

### 固体力学 QPINN の空白について

「quantum physics-informed neural network solid mechanics elasticity」「quantum neural
network linear elasticity Lamé thick-walled cylinder」等で検索した範囲では、
**QPINN を線形弾性・固体力学に適用した研究は見つからなかった**。古典 PINN 側は
Physics-Informed Holomorphic NN（arXiv:2407.01088）、Finite-PINN（arXiv:2412.09453）等、
線形弾性で層が厚い。計画の「固体力学 QPINN はほぼ空白」という認識は支持される。

ただし検索は英語の一般ウェブ検索であり、**arXiv の全文検索・Google Scholar・
2026年8月の台湾イベント（計画 §7 が挙げているもの）の予稿集は別途当たる必要がある**。

## 4. ベンチ #1 の実測結果（`11_qpinn_lame.py`）

無次元化した厚肉円筒（a=1, b=2, p_i=1, E=1, ν=0.3）、collocation 64 点、Adam + cosine
annealing、PDE 残差は Euler 型、境界条件の重み 10。CPU 実行。

### 単一シード（seed=0）

| モデル | params | rel L2(u) | rel L2(σ_r) | 時間 [s] | 学習ステップ |
|---|---|---|---|---|---|
| 古典 PINN（MLP 16×2） | 321 | 7.09e-4 | 1.06e-3 | 7.5 | 4000 |
| Fourier feature PINN（8 周波数） | 18 | 1.09e-1 | 2.75e-2 | 6.9 | 4000 |
| QPINN（4 qubit, 3 層, 2 再アップロード） | 50 | **5.70e-3** | **3.29e-3** | 428 | 3000 |

### 5 シード平均 ± 標準偏差（古典2モデル）

| モデル | params | rel L2(u) | rel L2(σ_r) |
|---|---|---|---|
| 古典 PINN | 321 | 3.45e-4 ± 2.1e-4 | 5.28e-4 ± 3.3e-4 |
| Fourier feature PINN | 18 | 8.16e-2 ± 7.7e-2 | 2.72e-2 ± 2.2e-2 |

### 読み取れること

- **QPINN は 50 パラメータで rel L2(u) = 5.7e-3 に到達**し、同程度に小さい古典 Fourier
  feature PINN（18 params, 8.2e-2 ± 7.7e-2）を**1桁以上上回った**。計画が狙う
  parameter efficiency の主張は、少なくとも #1 では成立する方向。
- 一方で **古典 PINN（321 params）には1桁負ける**（7.1e-4 対 5.7e-3）。計画の
  「古典超えは主張しない」という方針と整合する。
- **計算時間は 60 倍（7.5s 対 428s）**。パラメータ効率を主張する時は必ず併記すべき。
- **Fourier feature PINN は初期値依存が極端に強い**（5 シードで 4.4e-3 〜 1.9e-1、
  標準偏差が平均と同オーダー）。単一シードの比較は信用できない。**論文の表は必ず
  複数シードの平均±標準偏差で出すこと。** QPINN 側も同様に複数シードが必要
  （シミュレータが重いので計算資源の確保を先に）。
- σ_r（1階微分）の誤差が u の誤差とほぼ同オーダーに収まっているのは、解が滑らかで
  応力集中がないため。**#3 星形グレインではここが崩れるはず**で、その差こそが
  測定したい量。

### アブレーション（`12_qpinn_ablation.py`）

§5 の優先度2に挙げた「エンコード周波数 vs 最適化可能性」のアブレーションを実施した。
全結果は [docs/qpinn_ablation_results.md](qpinn_ablation_results.md)。要点:

- **表現力はボトルネックではない。** どの構成も解析解を 1e-3 前後で表現できるのに、
  PDE 損失で学習すると 5.2e-2〜1.6e+0 にしかならない。QPINN の誤差の 98〜99.9% は
  表現力ではなく最適化で失われている。
- **ladder エンコード（qubit ごとに (w+1) 倍の周波数）は表現力を上げずに最適化だけ壊す。**
  uniform と表現力はほぼ同じ（8.9e-4 vs 7.2e-4）なのに PDE 誤差は 30 倍悪い
  （1.51 vs 5.2e-2）。§2.2 で観測した失敗は表現力の問題ではないと確定した。
- **6 qubit は表現力が最良（4.3e-4）なのに PDE 性能は 4 qubit の 10 倍悪い。**
  barren plateau が 6 qubit で既に効き始めている。
- **動く構成の幅が狭い。** 層を 3→2 にするだけで 5.2e-2→1.4e+0 に崩れる。
  成功は 4 qubit / 3〜5 層 / 再アップロード 2 回の狭い谷のみ。

→ #3 に進む前に、この谷が Lamé 固有か問題非依存かを確認する必要がある。
また、アブレーションの軸は「回路サイズ」ではなく「損失設計」に移すべき。

### このベンチで判明した実装上の注意

§2.2 に記載。要約すると、(1) エンコード周波数を上げると PDE 損失が壊れる、
(2) 残差は Euler 型に正規化する、(3) 出力シフトの初期値を解の平均付近に置く、の3点。
いずれも #3 で再発するのでスクリプトにコメントとして残してある。

---

## 5. 推奨する優先順位の修正

計画の #1 → #3 → #5 という筋は維持。そのうえで:

1. **FEM 参照解（#3 用）の作成を 11–12月に前倒し**。QPINN 実装と並行させる。
2. ~~**#1 の段階で「エンコード周波数 vs 最適化可能性」のアブレーションを完了させる**~~
   → **実施済み**（§4 と [qpinn_ablation_results.md](qpinn_ablation_results.md)）。
   結論は「表現力ではなく最適化が律速」。以後のアブレーションは回路サイズではなく
   損失設計（境界条件の重み、残差の正規化、適応的な損失バランス）を振ること。
3. **多出力（u, v）の読み出し設計を 12月中に決める**。2D に進む前提条件であり、
   最大のリスク。観測量を分ける案・回路を複数本持つ案の両方を小問題で試す。
4. IBM 実機は 1D Lamé 1ケースに限定し、他は任意のまま。

---

## 6. 参照

- [QCPINN (arXiv:2503.16678)](https://arxiv.org/abs/2503.16678)
- [QPINNs for Maxwell's Equations (arXiv:2506.23246)](https://arxiv.org/abs/2506.23246)
- [QPINN for multi-variable PDEs (arXiv:2503.12244)](https://arxiv.org/html/2503.12244v1)
- [Trainable embedding QPINN, Scientific Reports (2025)](https://www.nature.com/articles/s41598-025-02959-z)
- [PIHNN: Linear Elasticity (arXiv:2407.01088)](https://arxiv.org/pdf/2407.01088)
- [Finite-PINN for Solid Mechanics (arXiv:2412.09453)](https://arxiv.org/pdf/2412.09453)
- [PINN for Dynamic Linear Elasticity (SSRN)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4841930)
