# 慶應のクラスタ（copaam）で回すジョブ（指示書）

copaam 側でやることだけを書いた指示書です．上から順に，**状態が「未実行」のもの**を回してください．
copaam はログインノードなので，計算は必ず `qsub` で計算ノードに投げます（詳細は `docs/server_guide.md` と `CLAUDE.md`）．

## 毎回の手順

```bash
cd ~/quantum-computing-intro
git pull                                   # 最新のコードとこの指示書を取り込む
qstat -a                                   # 空きを確認（同時は 10 本まで）
qsub -v JOB=<ジョブ名> scripts/pbs_job.sh   # 投入（既定ノード vancouver01 = RTX 4090）
tail -f logs/pbs_<ジョブ名>.log             # 先頭に devices: [CudaDevice(...)] と出れば GPU で動いている
```

終わったら（ログの最後に `=== done ===`）:

```bash
git add results/server
git commit -m "Cluster run: <ジョブ名>"
git push
```

push できないときは `results/server/*.json` をチャットにアップロードしてください．
回し終えたら，下の表の「状態」を「完了（日付）」に書き換えて一緒に push してください．

## ジョブ一覧

| 順 | ジョブ名 | 内容 | 出力 | 目安 | 状態 |
|---|---|---|---|---|---|
| 1 | `axisym` | 軸対称 (r, z) の TSV 1 本．MORE-Stress の寸法（直径 5 µm，高さ 50 µm，ピッチ 15 µm），上下固定 / 上面自由の 2 通り，5 シード | `results/server/tsv_axisym.json` | 約 50 分（vancouver01） | 完了（2026-11-03） |
| 2 | `tsv3d_plane` | 3D 単位セルの TSV（`44_tsv_3d.py`）．参照は中央面の平面ひずみ近似（上下固定のみ）．N = 8, 10，3 シード | `results/server/tsv_3d_plane_N8.json`，`_N10.json` | 約 55 分 | 完了（2026-11-03） |
| 3 | `tsv3d` | 3D 単位セルの TSV．参照は fifa の Abaqus．上下固定 `qsub -v JOB=tsv3d scripts/pbs_job.sh`，上面自由 `qsub -v JOB=tsv3d,BC3D=free scripts/pbs_job.sh`（Abaqus の結果はどちらも push 済み） | `results/server/tsv_3d_abaqus_clamped_N8.json`，`_N10.json`（free も同様） | 約 1 時間 | 完了（2026-11-03） |
| 4 | `bigN` | 量子ビット数を増やした TSV 単体（N = 10, 12，3 シード） | `results/server/tsv_single_N10.json`，`_N12.json` | 未計測（N = 12 は重い） | 完了（2026-11-03） |
| 5 | `shots` | ショットノイズ対策（`39_shot_mitigation.py`）．N = 10，強形式 vs エネルギー形，5 シード．**先に `.venv/bin/python -m pip install pennylane` と `.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu` を 1 回だけ実行**（copaam で実行してよい．import に必要） | `results/server/shot_mitigation_5s.json` | 約 30 分 | 完了（2026-11-03） |
| 6 | `edge_single` | **倍率の範囲の端で止まっていたものの回し直し**（以下 6〜9 共通．最良の倍率が探索範囲の端だった）．TSV 単体 N = 10 の Fourier を ×16〜×128，テンソル積射影を ×0.5, ×1 | `results/server/tsv_single_N10_fourier_wide.json`，`tsv_controls_rp_low.json` | 数時間（テンソル積射影が重い） | 完了（2026-11-03） |
| 7 | `edge_axisym` | 軸対称の QERC-Haar と MLP2-RF を ×0.25, ×0.5 | `results/server/tsv_axisym_low.json` | 約 30 分 | 完了（2026-11-03） |
| 8 | `edge_3d` | 3D（Abaqus 参照）の QERC-Haar・MLP2-RF・tanh-RF を ×0.25, ×0.5, ×1．上下固定 `qsub -v JOB=edge_3d scripts/pbs_job.sh`，上面自由 `qsub -v JOB=edge_3d,BC3D=free scripts/pbs_job.sh` | `results/server/tsv_3d_abaqus_<bc>_N8_low.json`，`_N10_low.json` | 数時間 | 完了（2026-11-03） |
| 9 | `edge_plane` | 3D（平面ひずみ参照）の QERC-Haar と MLP2-RF を ×0.25, ×0.5 | `results/server/tsv_3d_plane_N8_low.json`，`_N10_low.json` | 数時間 | 完了（2026-11-03） |
| 10 | `edge_periodic` | 周期配列 L = 3 の Fourier を ×32, ×64, ×128（5 シードとも最良が上限 ×16 の端だった）．`qsub -v JOB=edge_periodic scripts/pbs_job.sh` | `results/server/tsv_periodic_L3_fourier_wide.json` | 約 30 分 | 完了（2026-11-03）：×32 で 7.7e-2 と悪化、最良は ×16（1.4e-5）のまま |
| 11 | `pf_ac` | **Phase-field：Allen–Cahn**（Raissi 2019 の式 12）．量子リザバー・tanh・Fourier，5 シード，倍率は広めの範囲．N = 8 と 10 を別々に投げる：`qsub -v JOB=pf_ac,PFN=8 scripts/pbs_job.sh` と `qsub -v JOB=pf_ac,PFN=10,PFSEEDS=3 scripts/pbs_job.sh`（ログは `logs/pbs_pf_ac_N8.log` のように N 付き） | `results/server/allen_cahn_N8.json`，`_N10.json` | N = 8：1〜2 時間，N = 10：半日 | 完了（2026-11-04，GPU 版で再実行） |
| 12 | `pf_ch` | **Phase-field：Cahn–Hilliard**（混合形，ε² = 0.02）．N = 8 は 5 シード `qsub -v JOB=pf_ch,PFN=8 scripts/pbs_job.sh`，N = 10 は重いので 3 シード `qsub -v JOB=pf_ch,PFN=10,PFSEEDS=3 scripts/pbs_job.sh`（ログは `logs/pbs_pf_ch_N8.log` など） | `results/server/cahn_hilliard_N8.json`，`_N10.json` | N = 8：数時間，N = 10：1 日程度 | 完了（2026-11-04，GPU 版で再実行） |
| 13 | `pf_ch_wz` | **Cahn–Hilliard を Wight & Zhao（2021）の例題で**（γ2 = 0.01，γ1 = 1e-6，u0 = −cos 2πx．界面が鋭い）．古典 PINN の報告値 9.51e-3 と直接比べるため．N = 8 は 5 シード `qsub -v JOB=pf_ch_wz,PFN=8 scripts/pbs_job.sh`，N = 10 は 3 シード `qsub -v JOB=pf_ch_wz,PFN=10,PFSEEDS=3 scripts/pbs_job.sh` | `results/server/cahn_hilliard_wz_N8.json`，`_N10.json` | N = 8：数時間，N = 10：1 日程度 | 完了（2026-11-04） |
| 14 | `pf_ac_vqpinn` | **Allen–Cahn の比較対象**：変分 QPINN（6 量子ビット，6 層）と普通の PINN（同じパラメータ数の小さい MLP と，4 層 × 64）を同じ損失・Adam 20000 回で学習．3 シード．`qsub -v JOB=pf_ac_vqpinn scripts/pbs_job.sh` | `results/server/ac_vqpinn.json` | 数時間〜半日（VQPINN が重い） | 完了（2026-11-04） |
| 15 | `pf_ac2d` | **2 次元の Allen–Cahn（縮む円）**．最初に `--check`（厳密な Fourier 基底で参照解と時間刻みを確認，ログに出るだけ）．続けて量子リザバー・tanh・Fourier を 3 シード．N = 8 と 10 を別々に：`qsub -v JOB=pf_ac2d,PFN=8 scripts/pbs_job.sh`，`qsub -v JOB=pf_ac2d,PFN=10 scripts/pbs_job.sh`（GPU は 2 本だけ使う） | `results/server/ac2d_N8.json`，`_N10.json`，ログの check の行 | N = 8：数時間，N = 10：半日 | 完了（2026-11-04） |
| 16 | `pf_ac_product` | **命題の数値確認（Allen–Cahn）**：量子リザバーと，同じエンコーディングの古典の積の乱数特徴量を N = 8，5 シードで比べる．`qsub -v JOB=pf_ac_product scripts/pbs_job.sh` | `results/server/allen_cahn_N8_product.json` | 1〜2 時間 | 完了（2026-11-04） |
| 17 | `pf_ac2d_low` | **2D Allen–Cahn の倍率を下に広げる**（15 番で最良の倍率が 3 手法とも範囲の下の端だった）．3 手法を ×0.25, ×0.5, ×1 で 3 シード．N = 8 と 10 を別々に：`qsub -v JOB=pf_ac2d_low,PFN=8 scripts/pbs_job.sh`，`qsub -v JOB=pf_ac2d_low,PFN=10 scripts/pbs_job.sh` | `results/server/ac2d_N8_low.json`，`_N10_low.json` | N = 8：1〜2 時間，N = 10：数時間 | 完了（2026-11-04） |
| 18 | `pf_ac_data` | **Allen–Cahn をデータありで**（Rakala 実装と同じ「データ＋物理」）．参照解の値 1000 点（t = 0.1, …, 1.0 の各時刻に 100 点）を足す．最小二乗の 3 手法（N = 8，データの重み 0.1 と 1，3 シード）と，変分 QPINN・PINN（3 シード）．`qsub -v JOB=pf_ac_data scripts/pbs_job.sh` | `results/server/allen_cahn_N8_data_w0.1.json`，`_w1.json`，`ac_vqpinn_data.json` | 半日（変分 QPINN が重い） | 完了（2026-11-05） |
| 19 | `pf_vq_bench` | **18 番の変分 QPINN が遅い件の切り分け**（データ 1000 点を足すと 1 ステップ 4〜6 倍遅い）．`56_ac_vqpinn.py` は点の配列を jit の引数で渡すように直した（CPU ではデータの項は +6 % だけ）．データありとなしの 1 ステップの時間を測って表示するだけ（学習しない）．`qsub -v JOB=pf_vq_bench scripts/pbs_job.sh` | ログの `bench ...` の 4 行（VQPINN / PINN × データなし / あり） | 10 分 | 完了（2026-11-05） |
| 20 | `pf_ac_data_vq` | 19 番でデータありが遅くないと確かめてから：**18 番の変分 QPINN・PINN だけを回し直す**（直したコード）．走っている 18 番は止めてよい（最小二乗の分 `allen_cahn_N8_data_w*.json` はもう出ている）．`qsub -v JOB=pf_ac_data_vq scripts/pbs_job.sh` | `results/server/ac_vqpinn_data.json` | 数時間 | 完了（2026-11-05） |
| 21 | `te_setup`（qsub ではない） | **著者のコード（TE-QPINN）と専用の環境を用意する**．copaam のログインノードで `bash scripts/setup_te_venv.sh` を 1 回実行（ダウンロードとインストールだけ）．著者のリポジトリを `~/te-qpinns` に clone し，`.venv_te`（torch ＋ PennyLane）を作る | 最後に `torch ... pennylane ... cuda build ...` と `=== setup done ===` | 10〜20 分 | 未実行 |
| 22 | `te_smoke` | **動作確認**（小さな格子で 2 ステップだけ．結果ではない）．`qsub -v JOB=te_smoke scripts/pbs_job.sh` | ログに `'device': 'cuda'` と，Burgers・Allen–Cahn それぞれ 6 行のモデルの行 | 5 分 | 未実行 |
| 23 | `te_burgers` | **元論文の再現：Burgers**（著者の設定のまま：4 量子ビット，5 層，L-BFGS 500 ステップ）．5 シード．3 本に分けて投げる：`qsub -v JOB=te_burgers,TEMODEL=TE scripts/pbs_job.sh`，`TEMODEL=fixed`，`TEMODEL=classical` | `results/server/te_authors_burgers_TE.json`，`_fixed.json`，`_classical.json`（＋ `_fields.npz`，`te_authors_ref_*.npy`） | TE と fixed は 1 シード 1〜5 時間，classical は全部で 1 時間ほど | 未実行 |
| 24 | `te_ac` | **本番：Allen–Cahn**（著者の Allen–Cahn のノートブックのまま：因果性つきの学習，200 × 200 点，L-BFGS 最大 300 ステップ）．5 シード．同じく 3 本：`qsub -v JOB=te_ac,TEMODEL=TE scripts/pbs_job.sh`，`TEMODEL=fixed`，`TEMODEL=classical` | `results/server/te_authors_ac_TE.json`，`_fixed.json`，`_classical.json`（＋ `_fields.npz`） | TE と fixed は 1 シード 2〜10 時間（長くてよい），classical は数時間 | 未実行 |
| 25 | `te_ch` | **Cahn–Hilliard（Wight & Zhao の問題）を同じ枠組みで**．著者のコードに CH は無いので，損失（混合形：φ と μ を別々のネットワークで表す）だけこちらで書いた．回路・埋め込みの NN・格子・因果性つきの学習・L-BFGS は 24 番と同じ（著者の Allen–Cahn の設定）．5 シード．3 本：`qsub -v JOB=te_ch,TEMODEL=TE scripts/pbs_job.sh`，`TEMODEL=fixed`，`TEMODEL=classical`．**24 番が終わってから**投げる（`git pull` を先に） | `results/server/te_authors_ch_TE.json`，`_fixed.json`，`_classical.json`（＋ `_fields.npz`） | 24 番の 2 倍ほど（ネットワークが 2 つ） | 未実行 |
| 26 | `te_*`（`TEMODEL=fixedmatch`） | **固定の埋め込みで，パラメータ数を TE-QPINN にそろえたもの**（変分回路を深くする：Burgers は 20 層 = 240，Allen–Cahn・CH は 17 層 = 204．著者の回路のコードは層数を変えるだけ）．5 シード．`git pull` してから 3 本：`qsub -v JOB=te_burgers,TEMODEL=fixedmatch scripts/pbs_job.sh`，`JOB=te_ac,TEMODEL=fixedmatch`，`JOB=te_ch,TEMODEL=fixedmatch`．GPU が空きしだい，23〜25 番と並べてよい（同時に 4 本まで） | `results/server/te_authors_burgers_fixedmatch.json`，`te_authors_ac_fixedmatch.json`，`te_authors_ch_fixedmatch.json` | 同じ問題の fixed の 3〜4 倍 | 未実行 |
| 27 | `pf_ch2d` | **2 次元の Cahn–Hilliard を最小二乗の 3 手法で比べる**（量子リザバー，tanh，Fourier）．楕円の液滴が円に戻る問題（ε = 0.05，t = 0.4 まで，64 × 64 点，混合形）．`68_ch2d_qerc.py`．2 本：`qsub -v JOB=pf_ch2d,PFN=8 scripts/pbs_job.sh` と `PFN=10`．これは `.venv`（JAX）で動く．23〜26 番と並べてよい（同時に 4 本まで） | `results/server/ch2d_N8.json`，`ch2d_N10.json` | N = 8 は 1〜2 時間，N = 10 は半日ほど | 未実行 |

（`seeds5` / `tsv` / `array` は Claude 側の環境で回したので不要）

## 21〜25 番について（学習する QPINN の埋め込みの比較，著者のコードをそのまま使う）

村松先生の提案（2026-11-17）：固定の埋め込み，表現力を高めた埋め込み，パラメータ数をそろえた古典 NN で性能を比べる．

- 元論文：S. Berger, N. Hosters, M. Möller, "Trainable embedding quantum physics informed neural networks for solving nonlinear PDEs", Sci. Rep. 15, 18823 (2025), doi:10.1038/s41598-025-02959-z
- 著者のコード：https://git.rwth-aachen.de/berger.st.11.11/te-qpinns （commit `f7d0177`，PennyLane ＋ PyTorch．ログインなしで clone できる）
- `67_te_qpinn_authors.py` は，著者のノートブックのセル（回路・埋め込みの NN・コロケーション点・損失）を**書き換えずに読み込んで実行**する．
  こちらで書いたのは，学習のループ（著者のループと同じ `torch.optim.LBFGS(..., line_search_fn="strong_wolfe")`・同じ回数・同じ停止条件），
  参照解（スペクトル法），著者のコードに無いモデルだけ．著者のコードはこのリポジトリには入れない（`~/te-qpinns` に置く）

| モデル | 中身 |
|---|---|
| `TE` | 学習する埋め込み（小さな NN が量子ビットごとの倍率を出す）＋ 変分回路．著者の TE-QPINN |
| `NONE` / `CHEBYSHEV` / `TOWER` | 固定の埋め込み（座標をそのまま／Chebyshev 型）＋ 同じ変分回路．著者のコードにある選択肢 |
| `NONE-match` など | 同じ固定の埋め込みで，変分回路を深くしてパラメータ数を `TE` にそろえたもの（26 番） |
| `PINN-3x10` / `PINN-2x13` / `PINN-match` | パラメータ数をそろえた古典 PINN（著者の FNN クラス） |
| `PINN-4x64` | 大きめの古典 PINN（参考） |
| `TE-twin` | `TE` の回路だけを，同じ形の古典の関数（角度の三角関数の積の線形和）に置き換えたもの（こちらで追加） |

- **主な比較は，パラメータ数をそろえた 4 つ**：`TE`，`*-match`（固定の埋め込みを深くしたもの，26 番），`PINN-match`（Burgers は `PINN-2x13`・`PINN-3x10`），`TE-twin`．
  5 層のままの固定の埋め込み（60 パラメータ）と `PINN-4x64` は，著者の設定の再現と参考の値として残す
- 順番：21 番（1 回だけ）→ 22 番 → 23 番（3 本同時）→ 24 番（3 本同時）→ 25 番（3 本同時）．自分の GPU ジョブは同時に 4 本までにする
- 23・24 番は `TEMODEL` を必ず付ける（付けないとすぐ止まる）．ログは `logs/pbs_te_ac_<TEMODEL>.log` など
- 途中で止まっても，同じコマンドで投げ直せば終わったシードは飛ばして続きから回る（結果は 1 シードごとに保存）
- 21 番で「pinned versions did not install」と出たら，新しい版で入っている．そのまま進めてよい（版はログと結果に記録される）
- 22 番のログが `'device': 'cpu'` のときは，23 番を投げずに報告する（GPU 版の torch が入っていない）
- 回し終えたら，各ログのモデルごとの最後の行（`relL2=` のある行）を報告し，`results/server/te_authors_*` を push する

## うまくいかないとき

| 症状 | 対処 |
|---|---|
| ログに `CpuDevice` と出る | GPU が使えていない．結果は同じなので続けてよい（遅いだけ）．気になるときは `CLAUDE.md` の `LD_LIBRARY_PATH` の項を確認 |
| `No module named ...` | `.venv` が無い．`bash scripts/server_setup.sh gpu` を一度実行 |
| メモリ不足で落ちる | 別のノード（`-l nodes=1:ppn=2:gpus=1:stuttgart01` など）で投げ直す |
| ジョブがすぐ終わる | `logs/pbs_<ジョブ名>.log` の最後の数行をチャットに貼る |
