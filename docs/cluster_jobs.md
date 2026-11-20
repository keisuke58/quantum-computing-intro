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
| 21 | `te_setup`（qsub ではない） | **著者のコード（TE-QPINN）と専用の環境を用意する**．copaam のログインノードで `bash scripts/setup_te_venv.sh` を 1 回実行（ダウンロードとインストールだけ）．著者のリポジトリを `~/te-qpinns` に clone し，`.venv_te`（torch ＋ PennyLane）を作る | 最後に `torch ... pennylane ... cuda build ...` と `=== setup done ===` | 10〜20 分 | 完了（2026-11-18） |
| 22 | `te_smoke` | **動作確認**（小さな格子で 2 ステップだけ．結果ではない）．`qsub -v JOB=te_smoke scripts/pbs_job.sh` | ログに `'device': 'cuda'` と，Burgers・Allen–Cahn それぞれ 6 行のモデルの行 | 5 分 | 完了（2026-11-18） |
| 23 | `te_burgers` | **元論文の再現：Burgers**（著者の設定のまま：4 量子ビット，5 層，L-BFGS 500 ステップ）．5 シード．3 本に分けて投げる：`qsub -v JOB=te_burgers,TEMODEL=TE scripts/pbs_job.sh`，`TEMODEL=fixed`，`TEMODEL=classical` | `results/server/te_authors_burgers_TE.json`，`_fixed.json`，`_classical.json`（＋ `_fields.npz`，`te_authors_ref_*.npy`） | TE と fixed は 1 シード 1〜5 時間，classical は全部で 1 時間ほど | 完了（2026-11-18） |
| 24 | `te_ac` | **本番：Allen–Cahn**（著者の Allen–Cahn のノートブックのまま：因果性つきの学習，200 × 200 点，L-BFGS 最大 300 ステップ）．5 シード．同じく 3 本：`qsub -v JOB=te_ac,TEMODEL=TE scripts/pbs_job.sh`，`TEMODEL=fixed`，`TEMODEL=classical` | `results/server/te_authors_ac_TE.json`，`_fixed.json`，`_classical.json`（＋ `_fields.npz`） | TE と fixed は 1 シード 2〜10 時間（長くてよい），classical は数時間 | 完了（2026-11-18） |
| 25 | `te_ch` | **Cahn–Hilliard（Wight & Zhao の問題）を同じ枠組みで**．著者のコードに CH は無いので，損失（混合形：φ と μ を別々のネットワークで表す）だけこちらで書いた．回路・埋め込みの NN・格子・因果性つきの学習・L-BFGS は 24 番と同じ（著者の Allen–Cahn の設定）．5 シード．**24 番の結果から，300 ステップでは足りない見込みなので `TEOUTER=1500` を付ける**（例：`qsub -v JOB=te_ch,TEMODEL=TE,TEOUTER=1500 scripts/pbs_job.sh`．出力は `..._outer1500.json`）．3 本：`qsub -v JOB=te_ch,TEMODEL=TE scripts/pbs_job.sh`，`TEMODEL=fixed`，`TEMODEL=classical`．**24 番が終わってから**投げる（`git pull` を先に） | `results/server/te_authors_ch_TE.json`，`_fixed.json`，`_classical.json`（＋ `_fields.npz`） | 24 番の 2 倍ほど（ネットワークが 2 つ） | 完了（2026-11-18） |
| 26 | `te_*`（`TEMODEL=fixedmatch`） | **固定の埋め込みで，パラメータ数を TE-QPINN にそろえたもの**（変分回路を深くする：Burgers は 20 層 = 240，Allen–Cahn・CH は 17 層 = 204．著者の回路のコードは層数を変えるだけ）．5 シード．`git pull` してから 3 本：`qsub -v JOB=te_burgers,TEMODEL=fixedmatch scripts/pbs_job.sh`，`JOB=te_ac,TEMODEL=fixedmatch`，`JOB=te_ch,TEMODEL=fixedmatch`．GPU が空きしだい，23〜25 番と並べてよい（同時に 4 本まで） | `results/server/te_authors_burgers_fixedmatch.json`，`te_authors_ac_fixedmatch.json`，`te_authors_ch_fixedmatch.json` | 同じ問題の fixed の 3〜4 倍 | 完了（2026-11-19） |
| 27 | `pf_ch2d` | **2 次元の Cahn–Hilliard を最小二乗の 3 手法で比べる**（量子リザバー，tanh，Fourier）．楕円の液滴が円に戻る問題（ε = 0.05，t = 0.4 まで，64 × 64 点，混合形）．`68_ch2d_qerc.py`．2 本：`qsub -v JOB=pf_ch2d,PFN=8 scripts/pbs_job.sh` と `PFN=10`．これは `.venv`（JAX）で動く．23〜26 番と並べてよい（同時に 4 本まで） | `results/server/ch2d_N8.json`，`ch2d_N10.json` | N = 8 は 1〜2 時間，N = 10 は半日ほど | 完了（2026-11-19） |
| 28 | `te_ac`（`TEOUTER=1500`） | **Allen–Cahn を長く学習する**．24 番は 300 ステップで打ち切られ，因果性の重みが最後の時刻まで開かなかった（`min w` が 0.00 のまま，誤差 70〜110 %）．著者の図のデータは 762 ステップ分ある．上限を 1500 にして，重みが開いたら止まる著者の停止条件のまま回す．2 本：`qsub -v JOB=te_ac,TEMODEL=TE,TEOUTER=1500 scripts/pbs_job.sh` と `TEMODEL=classical`（固定の埋め込みは 24 番でロスが 0.11 から動かないので回さない）．`git pull` を先に | `results/server/te_authors_ac_TE_outer1500.json`，`te_authors_ac_classical_outer1500.json` | TE は 1 シード 最大 4 時間 | 完了（2026-11-18） |
| 29 | `te_*`（`TEMODEL=rakala`） | **固定の埋め込み = Rakala さんのモデル**（村松先生の訂正，2026-11-18：固定の埋め込みは `github.com/GeetRakala/QPINN` のモデルを指す）．古典の層 → 量子回路（AngleEmbedding ＋ StronglyEntanglingLayers，4 量子ビット × 4 層）→ 古典の層．元の形（65 パラメータ），improved（261），improved を TE-QPINN の数にそろえたもの（Burgers 248，Allen–Cahn 209）の 3 つ．5 シード．`git pull` してから 3 本：`qsub -v JOB=te_burgers,TEMODEL=rakala scripts/pbs_job.sh`，`qsub -v JOB=te_ac,TEMODEL=rakala,TEOUTER=1500 scripts/pbs_job.sh`，`qsub -v JOB=te_ch,TEMODEL=rakala,TEOUTER=1500 scripts/pbs_job.sh` | `results/server/te_authors_burgers_rakala.json`，`te_authors_ac_rakala_outer1500.json`，`te_authors_ch_rakala_outer1500.json` | Burgers は数時間，Allen–Cahn・CH は 1 シード最大数時間 | 完了（2026-11-20） |
| 30 | `pf_ac_reup` | **量子の優位性が出るかの実験：入力をエンタングルさせながら入れ直す量子リザバー**．今の量子リザバーは入力を量子ビットごとに別々に入れるので，古典でも安く作れる関数の範囲に収まる（論文の命題）．「入力を入れる → Haar ユニタリ」を 2 回・3 回くり返すと，その範囲の外に出る．Allen–Cahn（1 次元，最小二乗，これまでと同じ条件）で，くり返し 1・2・3 回を比べる．`.venv`（JAX）で動く．2 本：`qsub -v JOB=pf_ac_reup,PFN=8 scripts/pbs_job.sh` と `qsub -v JOB=pf_ac_reup,PFN=10,PFSEEDS=3 scripts/pbs_job.sh`．`git pull` を先に | `results/server/allen_cahn_N8_reup.json`，`allen_cahn_N10_reup.json` | N = 8 は 1 時間ほど，N = 10 は半日ほど | 完了（2026-11-19） |
| 31 | `te_*`（`TEMODEL=tran`） | **Tran らの学習する量子の埋め込み（LQNN-TE，arXiv:2609.33700）**．補助の量子回路（入力を入れ直す回路）が量子ビットごとの特徴量を出し，主回路の角度に足す．論文から書き起こした（公開コードは見つかっていない）．パラメータ数は TE-QPINN にそろえた（Burgers 240，Allen–Cahn 204）．5 シード．`git pull` してから 3 本：`qsub -v JOB=te_burgers,TEMODEL=tran scripts/pbs_job.sh`，`qsub -v JOB=te_ac,TEMODEL=tran,TEOUTER=1500 scripts/pbs_job.sh`，`qsub -v JOB=te_ch,TEMODEL=tran,TEOUTER=1500 scripts/pbs_job.sh` | `results/server/te_authors_burgers_tran.json`，`te_authors_ac_tran_outer1500.json`，`te_authors_ch_tran_outer1500.json` | 回路が 2 つあるので TE の 3〜5 倍．Burgers は 1 シード数時間 | 打ち切り（2026-11-20） |
| 32 | `pf_ac_reup_more` | **30 番の続き**．N = 8 では入れ直しの回数を増やすほど誤差が下がり（1 回 2.9e-2，2 回 5.7e-3，3 回 3.2e-3），最良の倍率が範囲の上端（8，16）に寄った．(a) 2 回・3 回を倍率 32，64 で，(b) 4 回・6 回を倍率 4〜64 で回す．`qsub -v JOB=pf_ac_reup_more,PFN=8 scripts/pbs_job.sh`．N = 10 は 30 番の N = 10 の結果を見てから | `results/server/allen_cahn_N8_reup_wide.json`，`allen_cahn_N8_reup_deep.json` | 2〜3 時間 | 完了（2026-11-19） |
| 33 | `te_ac`（`TEVAR=sched`） | **Allen–Cahn を，解ける設定で比べ直す**．著者の公開版の設定（単精度，因果性の強さ 200 の 1 段）では 1500 ステップでもどのモデルも解けなかった（24・28 番）．こちらの設定：倍精度，因果性の強さを 0.01 → 0.1 → 1 → 10 → 100 と段階的に上げる（Wang ら 2024）．回路・NN・格子・損失・L-BFGS は著者のまま．古典 PINN での確認（1 シード，各段 200 ステップ）：4 × 64 が 0.61 → 0.10，小さい PINN が 1.0 → 0.30．各段の上限を 500 ステップにして 4 本：`qsub -v JOB=te_ac,TEMODEL=classical,TEVAR=sched,TEOUTER=500 scripts/pbs_job.sh`，同じく `TEMODEL=TE`，`TEMODEL=rakala`，`TEMODEL=tran`．classical を先に投げる（速い）．`git pull` を先に | `results/server/te_authors_ac_<TEMODEL>_sched_outer500.json` | classical は数時間，量子は 1 シード数時間〜半日 | 打ち切り（2026-11-20） |
| 34 | `pf_reup_dt` | **入れ直しの量子リザバーと古典の tanh のどちらが上かを決める**．N = 10 では両方が時間刻みの誤差の下限（5e-6）に当たっている（入れ直し 6 回 5.4e-6，tanh 5.3e-6）．時間刻みを 1/4（2.5e-4）にして回す．3 シード．`qsub -v JOB=pf_reup_dt scripts/pbs_job.sh`．`.venv`（JAX） | `results/server/allen_cahn_N10_dt_classical.json`，`allen_cahn_N10_dt_reup.json` | 半日〜1 日（ステップ数が 4 倍） | 完了（2026-11-20） |
| 35 | `pf_deep_rf` | **古典側の公平な相手：層を重ねた乱数特徴量**（tanh 2 層・3 層，Fourier 2 層．重みはすべて乱数で固定）．入れ直しは回路を深くすることにあたるので，古典も深くして比べる．Allen–Cahn，これまでと同じ条件．2 本：`qsub -v JOB=pf_deep_rf,PFN=8 scripts/pbs_job.sh` と `qsub -v JOB=pf_deep_rf,PFN=10,PFSEEDS=3 scripts/pbs_job.sh` | `results/server/allen_cahn_N8_deep.json`，`allen_cahn_N10_deep.json` | N = 8 は 1 時間，N = 10 は数時間 | 完了（2026-11-19） |
| 36 | `pf_ch_reup` | **Cahn–Hilliard（1 次元，Wight & Zhao）でも入れ直しが効くか**．入れ直し 2・3・4・6 回と tanh 2 層．2 本：`qsub -v JOB=pf_ch_reup,PFN=8 scripts/pbs_job.sh` と `qsub -v JOB=pf_ch_reup,PFN=10,PFSEEDS=3 scripts/pbs_job.sh` | `results/server/cahn_hilliard_wz_N8_reup.json`，`cahn_hilliard_wz_N10_reup.json` | N = 8 は数時間，N = 10 は半日〜1 日 | 完了（2026-11-21） |
| 37 | `pf_2d_reup` | **2 次元（Allen–Cahn の縮む円，Cahn–Hilliard の楕円）でも入れ直しが効くか**．N = 10，入れ直し 2・3・4 回と tanh 2 層，3 シード．`qsub -v JOB=pf_2d_reup scripts/pbs_job.sh` | `results/server/ac2d_N10_reup.json`，`ch2d_N10_reup.json` | 1 日ほど | 完了（2026-11-20） |
| 38 | `pf_reup_shallow` | **浅い回路のリザバーでも入れ直しが効くか**．Haar のリザバーは量子ビットを増やすと出力が平らになって使えなくなる（Xiong ら）．代わりに浅い乱数回路（各量子ビットにランダムな回転 → 隣どうしに CZ，を 1 層または 2 層）を使い，入れ直し 1・3・6 回で比べる．Allen–Cahn，これまでと同じ条件．2 本：`qsub -v JOB=pf_reup_shallow,PFN=8 scripts/pbs_job.sh` と `qsub -v JOB=pf_reup_shallow,PFN=10,PFSEEDS=3 scripts/pbs_job.sh`．`.venv`（JAX） | `results/server/allen_cahn_N8_reup_shallow.json`，`allen_cahn_N10_reup_shallow.json` | N = 8 は 2 時間ほど，N = 10 は半日ほど | 完了（2026-11-20） |
| 39 | `pf_fair_1d`，`pf_fair_2d` | **比較を公平にするための確認**．(a) Cahn–Hilliard（1 次元）に古典の深い特徴量（tanh 3 層，Fourier 2 層）を足す．(b) 最良のスケールが範囲の端にあったモデルを，範囲を広げて回し直す（1 次元：tanh・Fourier を 64〜256，Allen–Cahn の Fourier を 128・256．2 次元：入れ直しを小さい側へ，tanh 2 層も小さい側へ）．3 本：`qsub -v JOB=pf_fair_1d,PFN=8 scripts/pbs_job.sh`，`qsub -v JOB=pf_fair_1d,PFN=10,PFSEEDS=3 scripts/pbs_job.sh`，`qsub -v JOB=pf_fair_2d scripts/pbs_job.sh`．`.venv`（JAX）．終わったら `python 71_fig_reservoir_summary.py` の出力（6 行）も報告 | `results/server/*_fair_*.json` | N = 8 は数時間，N = 10 と 2 次元は半日〜1 日 | 未実行 |

（`seeds5` / `tsv` / `array` は Claude 側の環境で回したので不要）

## 31・33 番の打ち切り（2026-11-20，西岡さんの判断）

copaam向け：次のとおり止めて，GPU を空けてください．止めたものは「打ち切り（2026-11-20）」として，終わった分の結果だけ push する．

| ジョブ | 対応 | 理由 |
|---|---|---|
| 31 番 Allen–Cahn（tran） | **回さない**（CPU でも回さない） | Burgers で 5 シードとも誤差 1.15〜1.19．解は t = 0 の初期条件だけ合い，次の時刻の行（t = 0.019）ですでに x < 0 側が 0 につぶれている．コロケーション点の時刻の間で急に変わる解に落ちて，ロスは低い（2.6e-3）のに解は外れる |
| 31 番 CH（tran，4699） | **今のシードが終わったら止める**（`qdel`） | 同じ理由．1 シード 16 時間 |
| 33 番 rakala（4704） | **すぐ止める** | 2 シードとも誤差 5.6〜7.5 で発散 |
| 33 番 tran（4705） | **すぐ止める** | 31 番と同じ理由 |
| 33 番 TE（4703） | **3 シードで止める**（42，43，44 が終わったら `qdel`） | 1 シード 20.5 時間．シード 42 で 0.21（著者の設定では 0.74）．3 シードあれば傾向は言える |

- GPU の故障（stuttgart01 の 0000:1a:00，stuttgart03 の 0000:b3:00，Xid 79）は，西岡さんから研究室の管理者に連絡する．直るまで，その GPU には投げない

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
| `RAKALA` / `RAKALA-imp` / `RAKALA-match` | **固定の埋め込み（先生の意図したもの）**：Rakala さんの QPINN．古典の層で前後を挟んだ回路で，回路への入力は決まった角度埋め込み（29 番） |
| `LQNN-TE` | Tran らの学習する量子の埋め込み（補助の量子回路の出力を角度に足す）．31 番 |
| `NONE-match` など | 同じ固定の埋め込みで，変分回路を深くしてパラメータ数を `TE` にそろえたもの（26 番） |
| `PINN-3x10` / `PINN-2x13` / `PINN-match` | パラメータ数をそろえた古典 PINN（著者の FNN クラス） |
| `PINN-4x64` | 大きめの古典 PINN（参考） |
| `TE-twin` | `TE` の回路だけを，同じ形の古典の関数（角度の三角関数の積の線形和）に置き換えたもの（こちらで追加） |

- **主な比較は，パラメータ数をそろえたもの**：`RAKALA-match`（固定の埋め込み，29 番），`TE`，`*-match`（Chebyshev 型の固定の埋め込みを深くしたもの，26 番），`PINN-match`（Burgers は `PINN-2x13`・`PINN-3x10`），`TE-twin`．
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
