#!/bin/bash
# IKM クラスタ（copaam から qsub で投げる）用のジョブスクリプト．
#   使い方（copaam で）:
#     qsub -v JOB=bigN scripts/pbs_job.sh
#     qsub -v JOB=seeds5 -l nodes=1:ppn=2:gpus=1:stuttgart01 scripts/pbs_job.sh   # ノードを変える場合
#   JOB は scripts/server_run.sh と同じ名前（seeds5 / tsv / array / shots / bigN）．
#   ログは logs/pbs_<JOB>.log．進み具合: qstat -a，tail -f logs/pbs_<JOB>.log
#PBS -N qpinn
#PBS -l nodes=1:ppn=2:gpus=1:vancouver01
#PBS -j oe

cd "${PBS_O_WORKDIR:?qsub から投げてください}"
mkdir -p logs results/server
exec > "logs/pbs_${JOB:?JOB を -v JOB=... で指定}.log" 2>&1

unset LD_LIBRARY_PATH                         # PBS の CUDA ライブラリが pip の CUDA とぶつかるため
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=$(grep -oE "gpu[0-9]+" "$PBS_GPUFILE" | head -1 | grep -oE "[0-9]+")
source .venv/bin/activate
python -c "import jax; print('devices:', jax.devices())"

# server_run.sh のジョブ定義を読み込んで，その場で（nohup なしで）実行する
eval "$(sed -n '/^run_job() {/,/^}/p' scripts/server_run.sh)"
run_job "$JOB"
echo "=== 完了 ==="
