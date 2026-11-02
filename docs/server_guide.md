# 研究室サーバー（fifa / copaam）で実験を回す手順

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
