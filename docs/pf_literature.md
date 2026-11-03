# Phase-field（Allen–Cahn / Cahn–Hilliard）の文献調査（2026-11-03）

村松先生の方針（QPINN を簡単な PF 方程式に．AC 型の次に CH 型）に沿って，「同じ例題で比べられる相手」を中心に整理．

## 1. 古典 PINN の標準例題（直接比べられる数値）

| 文献 | 例題 | 報告されている誤差 |
|---|---|---|
| Raissi, Perdikaris, Karniadakis, J. Comput. Phys. 378 (2019) | AC 1D：`u_t − 1e-4 u_xx + 5u³ − 5u = 0`，`u0 = x² cos πx`，周期（式 12） | 6.99e-3（t = 0.1 のデータから t = 0.9 を 100 段 RK 1 ステップで予測） |
| Wight & Zhao, Commun. Comput. Phys. 29 (2021) 930（arXiv 2007.04542） | **同じ AC 1D**（t ∈ [0, 1] 全体を時空間 PINN で） | 素の PINN 0.99，適応サンプリング最良 **2.33e-2** |
| 同上 | **CH 1D**：`u_t − (γ2(u³ − u) − γ1 u_xx)_xx = 0`，γ2 = 0.01，γ1 = 1e-6，`u0 = −cos 2πx`，周期，μ を別ネットワーク（混合形） | 空間サンプリングだけでは収束せず，時間方向の適応で最良 **9.51e-3** |
| Mattey & Ghosh, CMAME (2022)（bc-PINN，arXiv 2106.07606） | AC・CH（1D, 2D） | 時間を区切って順に学習．素の PINN より高精度 |
| Chen ら, J. Comput. Phys. (2025)（PF-PINNs） | AC と CH の連成（腐食） | — |

→ AC は **Raissi / Wight–Zhao と同じ例題**（今の `52_allen_cahn_qerc.py`）．CH も **Wight–Zhao の例題に合わせる**（`54_cahn_hilliard_qerc.py --eps2 1e-6 --g2 0.01 --ic cos2 --ref_modes 1024`，界面の幅 ≈ √(γ1/γ2) = 0.01 と鋭い）．
今の AC の暫定値（学習なし，t = 1 で 1e-3〜1e-2）は，適応 PINN の 2.33e-2 より良い．

## 2. 今回と同じ枠組みの古典手法（最も近い先行研究）

- **Zhou, Fu, Wang, Feng, arXiv 2604.25502 (2026)**：乱数特徴量（random feature method）＋ IMEX Runge–Kutta で時間を進め，各段は線形最小二乗．AC・CH・Burgers・KdV で誤差 1e-6 程度，IMEX-PINN より高精度で安い．
  → **今回の方法の古典版そのもの**．量子リザバーを特徴量に使う点だけが違う．論文ではこれを必ず引用し，比較対象に入れる
- Chen ら, random feature method（arXiv 2207.13380）：乱数特徴量で PDE を解く方法の元論文
- CARE-SAV（arXiv 2608.25300）：勾配流（PF）をエネルギー安定に解く乱数特徴量法
- 細胞の PF に乱数特徴量 ＋ 陽的 RK（arXiv 2412.05920）

## 3. 量子

| 文献 | 内容 | 今回との違い |
|---|---|---|
| Zabihi, Montazeri, Suiker, arXiv 2606.04679 (2026) | 古典 NN ＋ 量子回路のハイブリッド PINN．Burgers・**AC**・KdV．AC で古典 PINN より誤差が約 5 分の 1 | 変分（勾配法で学習）．古典側は PINN だけで，乱数特徴量や同規模の対照はない |
| An, Fang, Jordan, Liu, Low, Wang, arXiv 2205.01141 | Carleman 線形化による量子アルゴリズム（反応拡散，Fisher–KPP と **AC**） | 誤り耐性量子計算機向け．PINN ではない |
| arXiv 2411.16233 | PF への Carleman 線形化（長時間で発散する問題の対策） | 同上 |
| arXiv 2510.19855 | 反応拡散（AC を含む）の Chebyshev 型量子アルゴリズム | 同上 |

**量子リザバー（学習なし）で AC・CH を解いた論文は見つからなかった．**

### 研究室内：佐々木翔矢くん（村松研）の CLS

- Sasaki, Endo, Muramatsu, "Hamiltonian simulation for nonlinear partial differential equation by Schrödingerization", Sci. Rep. 16, 11743 (2026)（arXiv 2508.01640）
- **CLS = Carleman 線形化 ＋ Schrödingerization**（warped phase transformation）．非線形の反応拡散を線形化し，Schrödinger 方程式にしてハミルトニアンシミュレーションで解く
- 例題：`φ_t = D φ_xx + φ − φ²`（Fisher–KPP 型，2 次の非線形），x ∈ (0, 1)，Dirichlet，`φ0 = 0.5 − 0.5 cos 2πx`．差分 36 点，Carleman 3 次
- 論文の数値は CLS の古典シミュレーション（実機・回路シミュレーションなし）．2026 年は実機向けの具体的な量子回路設計（WCCM-ECCOMAS 2026，第 31 回計算工学講演会，門脇・鈴木と共著）
- 今回との関係：同じ「非線形の反応拡散を量子で」でも，佐々木くんは誤り耐性向けのハミルトニアンシミュレーション，こちらは近い将来の機械向けの量子リザバー ＋ 最小二乗．**重ならず補い合う**
  - 共通の例題（佐々木くんの Fisher–KPP 型，または AC）で両方を並べると研究室としてまとまる．こちらの方法は Dirichlet 境界も境界条件の行を足せば扱える
  - 進める前に佐々木くんと例題・パラメータをすり合わせる
 量子で AC を扱った PINN は Zabihi ら（変分のハイブリッド）だけ．

## 4. 論文での位置づけ（案）

- 新しさ：量子リザバー PINN を PF（AC → CH）に初めて適用．学習なしで，適応 PINN（Wight–Zhao）と同じ例題で比べる
- 公平さ：同じ枠組みの古典の乱数特徴量（Zhou ら 2026 の方法）と，同じ数・同じ倍率調整で比べる
- 正直な見通し：今のところ古典の乱数特徴量の方が良い．「量子が勝つ」ではなく，「どこまで届き，何が足りないか」（脱量子化の議論，ショットノイズ）を示す論文になる
- TSV（固体）と PF を 1 本にまとめると，問題の幅があって主張が強くなる
