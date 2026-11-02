# 研究室サーバー（fifa / copaam）で実験を回す手順

**方針（2026-11-02）: 重い計算は基本 IKM クラスタの GPU で回す（copaam から qsub）．** 回すジョブの一覧と状態は `docs/cluster_jobs.md`． 新しい実験はジョブとして `scripts/server_run.sh` に追加する．

## まずはこれ：copaam から計算ノードの GPU に投げる（最短手順）

**copaam はログインノード（GPU なし，計算禁止）．計算は必ず `qsub` で計算ノードに投げる．**
`scripts/server_run.sh`（nohup）は copaam では使わない．詳細は `CLAUDE.md` の IKM クラスタの節．

```bash
# 1. 取得とセットアップ（初回のみ．copaam で実行してよい．最後に CpuDevice と CUDA のエラーが出るが正常）
cd ~
git clone https://github.com/keisuke58/quantum-computing-intro.git
cd quantum-computing-intro
git checkout ccr-e0b1f2e9-h1hyaz
bash scripts/server_setup.sh gpu

# 2. 空きを確認して投入（既定は vancouver01 = RTX 4090）
pbsnodes -a | less
qstat -a
qsub -v JOB=bigN scripts/pbs_job.sh
#   別ノード: qsub -v JOB=bigN -l nodes=1:ppn=2:gpus=1:stuttgart01 scripts/pbs_job.sh

# 3. 進み具合
qstat -a
tail -f logs/pbs_bigN.log       # 先頭に devices: [CudaDevice(...)] と出れば GPU で動いている
```

- 計算ノード: vancouver01/02 = RTX 4090，stuttgart01–03 = RTX 3090（celtic03 は GPU 故障）．同時は 10 本まで
- 終わったら `results/server/*.json` を push するか，チャットにアップロード

以下は詳しい説明です．

QPINN / QERC-PINN の実験を，研究室のサーバーで回すための指示書です．
コマンドは上から順にコピーして貼り付ければ動きます．

## 0. 前提

- サーバーに SSH でログインできること
- Python 3.10 以上（`python3 --version` で確認）
- GPU を使う場合: NVIDIA GPU と CUDA 12 系のドライバ（`nvidia-smi` で確認）
- GitHub からクローンできること（非公開リポジトリなら GitHub の認証が要る）

## 1. 取得（初回のみ）

```bash
cd ~
git clone https://github.com/keisuke58/quantum-computing-intro.git
cd quantum-computing-intro
git checkout ccr-e0b1f2e9-h1hyaz        # 最新の実験コードがあるブランチ
```

2 回目以降は最新を取り込むだけ:

```bash
cd ~/quantum-computing-intro
git pull
```

## 2. セットアップ（初回のみ）

仮想環境 `.venv` を作り，必要なパッケージ（numpy, scipy, matplotlib, jax）だけ入れます．
qiskit や torch は今回の実験には不要なので入れません．

```bash
bash scripts/server_setup.sh          # CPU のみ
# または
bash scripts/server_setup.sh gpu      # GPU を使う
```

最後に `devices: [CpuDevice(...)]` または `[CudaDevice(...)]` と出れば成功です．
GPU を指定したのに CPU と出る場合は，CUDA のバージョンが合っていません（CPU のままでも動きます）．

## 3. 実験を回す

```bash
bash scripts/server_run.sh <ジョブ名>
```

裏で動くので，SSH を切っても止まりません．

| ジョブ名 | 内容 | 目安（CPU 4 コア） |
|---|---|---|
| `seeds5` | 5 シードで数値を確定（下の 3 つを順に） | 5〜6 時間 |
| `tsv` | TSV 単体 + 古典対照（テンソル積射影・2 層ネット） | 1〜2 時間 |
| `array` | TSV 周期配列（L = 3, 4, 6） | 1〜2 時間 |
| `shots` | ショットノイズ対策（N = 10，強形式 vs エネルギー形） | 2〜3 時間 |
| `bigN` | 量子ビット数を増やした TSV 単体（N = 10, 12） | 未計測（N = 12 は重い） |
| `axisym` | 軸対称 (r, z) の TSV 1 本（MORE-Stress の寸法，上下固定 / 上面自由） | 数時間（FE の参照解が重い） |

同じサーバーで同時に 2 つ以上回すと遅くなります．1 つずつがおすすめです．

## 4. 進み具合の確認

```bash
tail -f logs/<ジョブ名>.log      # Ctrl+C で表示をやめる（実験は止まらない）
```

最後に `=== 完了 ===` と出たら終わりです．
途中で止めたいときは:

```bash
pkill -f "python 3"              # このリポジトリの実験（3x_*.py）をまとめて止める
```

## 5. 結果の受け渡し

結果は `results/server/*.json` に出ます．次のどちらかで共有してください．

- そのままコミットして push する
  ```bash
  git add results/server
  git commit -m "Server runs: <ジョブ名>"
  git push
  ```
- または json ファイルをチャットにアップロードする

受け取ったら，図と表の更新（`docs/qpinn_next_plan.md`，`paper/main.tex`）はこちらでやります．

## 困ったとき

| 症状 | 対処 |
|---|---|
| `No module named 'numpy'` など | `source .venv/bin/activate` を忘れている．`server_run.sh` は自動で読み込むので，手で python を叩くときだけ必要 |
| メモリ不足で落ちる | `bigN` の N = 12 は重い．`nvidia-smi` / `free -g` で空きを確認し，他の人のジョブが少ないときに回す |
| GPU で遅い | 実験は倍精度（float64）なので，GPU によっては CPU と大差ないことがある |
| `git clone` で認証を求められる | GitHub のアカウントでログインするか，個人用アクセストークンを使う |
