# CLAUDE.md（quantum-computing-intro）

## IKM クラスタ（copaam）で実験を回すとき

- **copaam はログインノード**（GPU なし、計算は禁止）。編集・投入・確認だけにする。
- **`scripts/server_run.sh`（nohup で裏で動かす）は copaam では使わない**。GPU は他の利用者と共有しているので、
  計算は必ず PBS/Torque の `qsub` で計算ノードに投げる（vancouver01/02 = RTX 4090、stuttgart01–03 = RTX 3090）。
- ジョブスクリプトの要点（`server_run.sh` の各ジョブの中身をそのまま書く）：
  ```bash
  #PBS -l nodes=1:ppn=2:gpus=1:vancouver01      # ノードを固定（celtic03 は GPU 故障）
  #PBS -j oe
  #PBS -o <repo>/logs/<job>.log
  unset LD_LIBRARY_PATH                         # PBS の /usr/local/cuda/lib64 が pip の CUDA とぶつかる
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export CUDA_VISIBLE_DEVICES=$(grep -oE "gpu[0-9]+" "$PBS_GPUFILE" | head -1 | grep -oE "[0-9]+")
  cd <repo> && .venv/bin/python 30_qerc_tsv_single.py --N 10 --seeds 3 --out results/server/tsv_single_N10.json
  ```
- 投入前に `pbsnodes -a` / `qstat -a` で空きを見る。同時は 10 本まで。
- **ジョブスクリプトは ASCII だけで書く**。日本語のコメントが 1 文字でもあると Torque が `qsub:  file must be an ascii script` で受け付けない（`scripts/pbs_job.sh` がそうだった）。
- `bash scripts/server_setup.sh gpu` は copaam で実行してよい（`.venv` を作るだけ）。copaam には GPU が無いので、
  最後の確認は `CpuDevice` と CUDA のエラーを出すが正常。GPU の確認は計算ノードで行う。
- 動作確認済み（2026-11-03）：`jax[cuda12]` = JAX 0.12.0.1 が vancouver01（RTX 4090、ドライバ 550.90.07、CUDA 12.4）で
  `CudaDevice` を認識。numpy 2.5.4、scipy 1.18.1。
- fifa はホームが別で Rocky 9。copaam の `.venv` はそのまま使えない（未確認）。

## クラスタに回してほしい計算を頼むとき

- 指示は必ず `docs/cluster_jobs.md`（指示書）に書く．チャットにコマンドを並べるだけにしない
- 新しいジョブは `scripts/server_run.sh` の `run_job` に追加し，指示書の「ジョブ一覧」に 1 行足して状態を「未実行」にする
- copaam 側は指示書を上から読み，回し終えたら状態を「完了（日付）」にして結果と一緒に push する
