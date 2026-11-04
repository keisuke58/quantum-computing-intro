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
| 13 | `pf_ch_wz` | **Cahn–Hilliard を Wight & Zhao（2021）の例題で**（γ2 = 0.01，γ1 = 1e-6，u0 = −cos 2πx．界面が鋭い）．古典 PINN の報告値 9.51e-3 と直接比べるため．N = 8 は 5 シード `qsub -v JOB=pf_ch_wz,PFN=8 scripts/pbs_job.sh`，N = 10 は 3 シード `qsub -v JOB=pf_ch_wz,PFN=10,PFSEEDS=3 scripts/pbs_job.sh` | `results/server/cahn_hilliard_wz_N8.json`，`_N10.json` | N = 8：数時間，N = 10：1 日程度 | 未実行（N = 8 は完了 2026-11-04，N = 10 は実行中） |
| 14 | `pf_ac_vqpinn` | **Allen–Cahn の比較対象**：変分 QPINN（6 量子ビット，6 層）と普通の PINN（同じパラメータ数の小さい MLP と，4 層 × 64）を同じ損失・Adam 20000 回で学習．3 シード．`qsub -v JOB=pf_ac_vqpinn scripts/pbs_job.sh` | `results/server/ac_vqpinn.json` | 数時間〜半日（VQPINN が重い） | 完了（2026-11-04） |
| 15 | `pf_ac2d` | **2 次元の Allen–Cahn（縮む円）**．最初に `--check`（厳密な Fourier 基底で参照解と時間刻みを確認，ログに出るだけ）．続けて量子リザバー・tanh・Fourier を 3 シード．N = 8 と 10 を別々に：`qsub -v JOB=pf_ac2d,PFN=8 scripts/pbs_job.sh`，`qsub -v JOB=pf_ac2d,PFN=10 scripts/pbs_job.sh`（GPU は 2 本だけ使う） | `results/server/ac2d_N8.json`，`_N10.json`，ログの check の行 | N = 8：数時間，N = 10：半日 | 未実行 |

（`seeds5` / `tsv` / `array` は Claude 側の環境で回したので不要）

## うまくいかないとき

| 症状 | 対処 |
|---|---|
| ログに `CpuDevice` と出る | GPU が使えていない．結果は同じなので続けてよい（遅いだけ）．気になるときは `CLAUDE.md` の `LD_LIBRARY_PATH` の項を確認 |
| `No module named ...` | `.venv` が無い．`bash scripts/server_setup.sh gpu` を一度実行 |
| メモリ不足で落ちる | 別のノード（`-l nodes=1:ppn=2:gpus=1:stuttgart01` など）で投げ直す |
| ジョブがすぐ終わる | `logs/pbs_<ジョブ名>.log` の最後の数行をチャットに貼る |
