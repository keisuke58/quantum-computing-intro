#!/bin/bash
# Job script for the Keio cluster (submit with qsub from copaam).
# NOTE: Torque rejects job scripts with non-ASCII bytes ("file must be an ascii script"), so keep this file ASCII only.
#   Usage (on copaam, from the repo root):
#     qsub -v JOB=bigN scripts/pbs_job.sh
#     qsub -v JOB=seeds5 -l nodes=1:ppn=2:gpus=1:stuttgart01 scripts/pbs_job.sh   # other node
#   JOB uses the same names as scripts/server_run.sh (seeds5 / tsv / array / shots / bigN / axisym).
#   Log: logs/pbs_<JOB>.log. Progress: qstat -a, tail -f logs/pbs_<JOB>.log
#PBS -N qpinn
#PBS -l nodes=1:ppn=2:gpus=1:vancouver01
#PBS -j oe

cd "${PBS_O_WORKDIR:?submit with qsub}"
mkdir -p logs results/server
exec > "logs/pbs_${JOB:?set JOB with -v JOB=...}.log" 2>&1

unset LD_LIBRARY_PATH                         # PBS sets the system CUDA libs, which clash with the pip CUDA wheels
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=$(grep -oE "gpu[0-9]+" "$PBS_GPUFILE" | head -1 | grep -oE "[0-9]+")
source .venv/bin/activate
python -c "import jax; print('devices:', jax.devices())"

# Load the job definitions from server_run.sh and run them here (no nohup)
eval "$(sed -n '/^run_job() {/,/^}/p' scripts/server_run.sh)"
run_job "$JOB"
echo "=== done ==="
