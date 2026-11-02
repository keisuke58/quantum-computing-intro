# QAOA の引き継ぎ（本田さん，村松研）メモ

Web で見つかった本田理央さん（村松研）の関連業績（2026-11-03 調べ）．本文はまだ読んでいない．

| 業績 | 内容 |
|---|---|
| R. Honda, K. Endo, T. Kaji, Y. Suzuki, Y. Matsuda, S. Tanaka, M. Muramatsu, "Development of optimization method for truss structure by quantum annealing", Scientific Reports 14, 13872 (2024), DOI 10.1038/s41598-024-64588-2 | トラスのトポロジー最適化を量子アニーリングで．節点変位と断面積を 2 値変数で表し，ハミルトニアン = 弾性ひずみエネルギー + 位置エネルギー．有限要素の問題は古典で解き，残差の最小化に量子アニーリングを使うハイブリッド |
| R. Honda, K. Endo, Y. Suzuki, Y. Matsuda, S. Tanaka, M. Muramatsu, "Development of Hamiltonian for structural applications by quantum annealing assuming finite element method", Int. J. Numer. Methods Eng. (2026) | 有限要素法を前提にした構造解析用のハミルトニアン（村松研サイトに掲載．巻号は未確定） |
| 本田理央ほか（慶應・産総研・Fixstars Amplify），「周期トラス構造解析における量子アニーリングと量子近似最適化アルゴリズムの性能比較」，第 29 回計算工学講演会（OS15） | **QAOA の話はこれ**．周期トラスの構造解析で量子アニーリングと QAOA を比較．要旨は未確認 |

関連: 村松先生の産総研 G-QuAT での講演資料（2024-05-10，`unit.aist.go.jp/g-quat/.../20240510_07_Muramatsu.pdf`），
Xiao, Muramatsu ほか（ガウス型 on-off エンコーディングによるトラス解析，計算工学講演会 2025）．

## 引き継ぎで本田さんに聞くこと

- 計算工学講演会の QAOA の発表資料と講演論文（PDF），コード一式（Qiskit か Amplify か），使った環境（シミュレータ・実機）
- 周期トラスの問題設定（何を 2 値変数にしたか，QUBO の作り方），比較の結果，うまくいかなかったこと
- 4 年生に渡すなら，どこから始めるのがよいか

## 今の研究とのつながり

- 「構造解析を量子（アニーリング / QAOA）で解く」は，QPINN と同じく固体力学 × 量子．比較の作法（古典の手法も同じ手間で調整する）がそのまま使える
- 周期トラスは周期構造なので，TSV の単位セルの考え方とも近い
