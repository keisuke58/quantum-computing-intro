# TSV の熱応力・キープアウトゾーンと機械学習の先行研究（2026-11-03 調べ）

Web 検索で書誌情報を確かめたものだけを載せる．本文まで読めていないものは「要旨のみ」と書いた．
論文に引用する前に，原典で数値と書誌を確認すること．

## 1. TSV の熱応力とキープアウトゾーン（KOZ）

| 文献 | 内容 | 確認の程度 |
|---|---|---|
| K. H. Lu, X. Zhang, S.-K. Ryu, J. Im, R. Huang, P. S. Ho, "Thermo-mechanical reliability of 3-D ICs containing through silicon vias", ECTC 2009 | TSV 周りの熱応力が Si のき裂とデバイス性能の両方を脅かす．小さい TSV ほど応力は距離とともに速く減衰．TSV 同士の干渉で KOZ は軸対称でなくなる．背景として「約 100 MPa の応力で MOSFET の移動度が 7% 以上変わる」を引用 | 一部（PDF は UT Austin の Huang 研のサイト） |
| S.-K. Ryu, K. H. Lu, T. Jiang, J. Im, R. Huang, P. S. Ho, "Effect of thermal stresses on carrier mobility and keep-out zone around through-silicon vias for 3-D integration", IEEE Trans. Device Mater. Reliab. 12(2), 2012 | 表面付近の応力を FEA で求め，ピエゾ抵抗効果で移動度の変化を評価．KOZ = 移動度の変化が基準（5%）を超える範囲．ピッチが小さいと干渉で KOZ が広がる．Cu の塑性で応力と KOZ が小さくなる | 要旨のみ |
| S.-K. Ryu ほか（同グループ），IEEE Trans. Device Mater. Reliab.，2011 | 表面付近の応力分布は 2 次元の解とはっきり違う（こちらの結果「2D は表面付近を表せない」と同じ） | 要旨のみ（題名は要確認） |
| T. Jiang ほか（同グループ），ICSICT 2012 | TSV 近くのデバイスに対する KOZ への熱応力の影響 | 要旨のみ |
| Zhu ほか（同グループ），IEEE Trans. Device Mater. Reliab.，DOI 10.1109/TDMR.2015.2401035 | ラマン分光と FEA で高温での表面付近の応力を測定．KOZ = 移動度の変化 10% 以上 | 要旨のみ |
| C. S. Selvanayagam, J. H. Lau, X. Zhang ほか, "Nonlinear thermal stress/strain analyses of copper filled TSV (through silicon via) and their flip-chip microbumps", IEEE Trans. Adv. Packag. 32(4), 720–728, 2009 | Cu の TSV の非線形（塑性）熱応力．Si の厚さとビア径の比（アスペクト比）を変えて解析 | 書誌のみ |
| M. Jung, J. Mitra, D. Z. Pan, S. K. Lim, "TSV stress-aware full-chip mechanical reliability analysis and optimization for 3D IC", DAC 2011；拡張版 IEEE Trans. CAD 31(8), 1194–1207, 2012 | TSV 1 本の応力テンソルの線形重ね合わせでチップ全体を評価（FEA と比較）．von Mises で信頼性を判定．ランディングパッド・ライナーの厚さ・KOZ・配置を最適化 | 要旨・一部 |
| T. Zhu, Q. Wang, Y. Lin, R. Wang, R. Huang, "MORE-Stress", DATE 2025, arXiv:2411.12690 | 単位ブロックのモデル次数削減．ANSYS より 153〜504 倍速く誤差 1% 未満．線形重ね合わせは誤差 5〜14% | 本文を読んだ（`docs/qpinn_next_plan.md`） |

## 2. 機械学習・ニューラルネットによる TSV / 3D IC の解析

| 文献 | 内容 | TSV の熱応力か |
|---|---|---|
| Zhang, Wang, Tan, "Fast stress analysis for runtime reliability enhancement of 3D IC using artificial neural network", ISQED 2016 | 熱と応力のデータで ANN をオフライン学習し，動作中に応力を推定 | 3D IC の応力（データ駆動） |
| 同グループ, "Runtime stress estimation for three-dimensional IC reliability management using artificial neural network", ACM（DOI 10.1145/3363185） | 3 種類の ANN を比較し，CNN が精度と計算量で最良 | 3D IC の応力（データ駆動） |
| Wang ほか（MORE-Stress の文献 [15]） | TSV 配列の動作中の熱応力を深層学習で解析．MORE-Stress は「信頼性が保証されず，学習した条件の外に使えない」と批判 | TSV の熱応力（データ駆動）．書誌は要確認 |
| Gharib, Popryho, Partin-Vaisband, "TSV-PhGNN", arXiv:2603.29268, 2026 | 物理を組み込んだグラフニューラルネットで TSV 配列の S パラメータと等価熱伝導率を予測．15×15 配列で HFSS の 2,998 s に対し 1.73 ms | 電気・熱（応力ではない） |
| arXiv:2508.16999, "Physics-Informed Kolmogorov-Arnold Networks for multi-material elasticity problems in electronic packaging" | 電子実装の多材料弾性を物理情報付きの KAN で解く | 実装の弾性（TSV かは未確認） |
| Zhang, Liu, Chen, arXiv:2503.04049, 2025 | HBM の接合部温度とホットスポットを NN で推定（FEM の 13,494 条件で学習） | 温度（応力ではない） |

## 3. 位置づけ（こちらの研究との関係）

- TSV の熱応力と KOZ の基礎（Lu 2009，Ryu 2011/2012，Selvanayagam 2009）と，高速な配列解析（Jung 2011/2012 の線形重ね合わせ，MORE-Stress 2025）は確立している
- 機械学習は主に**データ駆動**（FEM の結果で学習）．**PINN（物理情報）で TSV の熱応力を解いた論文は見つからなかった**
- 量子リザバー + 最小二乗（PIELM 型）で TSV を解き，同じ寸法（MORE-Stress）で Abaqus と比べるのは新しい
- Ryu らの「表面付近は 2D と違う」は，こちらの軸対称・3D の結果と一致する．KOZ の定義は移動度の変化（5% や 10%）が標準 → こちらの「von Mises 300 MPa」は仮の目安なので，論文ではピエゾ抵抗係数から移動度の変化に直すのが筋
