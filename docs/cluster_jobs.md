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
| 1 | `axisym` | 軸対称 (r, z) の TSV 1 本．MORE-Stress の寸法（直径 5 µm，高さ 50 µm，ピッチ 15 µm），上下固定 / 上面自由の 2 通り，5 シード | `results/server/tsv_axisym.json` | 数時間 | 未実行 |
| 2 | `tsv3d_plane` | 3D 単位セルの TSV（`44_tsv_3d.py`）．参照は中央面の平面ひずみ近似（上下固定のみ）．N = 8, 10，3 シード | `results/server/tsv_3d_plane_N8.json`，`_N10.json` | 数時間（N = 10 はメモリ 10 GB 程度） | 未実行 |
| 3 | `tsv3d` | 3D 単位セルの TSV．参照は fifa の Abaqus．上下固定 `qsub -v JOB=tsv3d scripts/pbs_job.sh`，上面自由 `qsub -v JOB=tsv3d,BC3D=free scripts/pbs_job.sh`（Abaqus の結果はどちらも push 済み） | `results/server/tsv_3d_abaqus_clamped_N8.json`，`_N10.json`（free も同様） | 数時間 | 未実行 |
| 4 | `bigN` | 量子ビット数を増やした TSV 単体（N = 10, 12，3 シード） | `results/server/tsv_single_N10.json`，`_N12.json` | 未計測（N = 12 は重い） | 完了（2026-11-03） |

（`seeds5` / `tsv` / `array` / `shots` は Claude 側の環境で回しているので不要）

## うまくいかないとき

| 症状 | 対処 |
|---|---|
| ログに `CpuDevice` と出る | GPU が使えていない．結果は同じなので続けてよい（遅いだけ）．気になるときは `CLAUDE.md` の `LD_LIBRARY_PATH` の項を確認 |
| `No module named ...` | `.venv` が無い．`bash scripts/server_setup.sh gpu` を一度実行 |
| メモリ不足で落ちる | 別のノード（`-l nodes=1:ppn=2:gpus=1:stuttgart01` など）で投げ直す |
| ジョブがすぐ終わる | `logs/pbs_<ジョブ名>.log` の最後の数行をチャットに貼る |
