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
exec > "logs/pbs_${JOB:?set JOB with -v JOB=...}${BC3D:+_$BC3D}${PFN:+_N$PFN}.log" 2>&1   # BC3D/PFN in the name: parallel runs must not share a log

unset LD_LIBRARY_PATH                         # PBS sets the system CUDA libs, which clash with the pip CUDA wheels
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Keep CPU threads within the reserved cores (ppn). Without this, numpy/BLAS and XLA's CPU backend
# used 4-6 cores per job while only 2 were reserved (52/54 ran on CPU, GPU at 0%).
NT=${PBS_NUM_PPN:-2}
export OMP_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT MKL_NUM_THREADS=$NT NUMEXPR_NUM_THREADS=$NT
export XLA_FLAGS="${XLA_FLAGS:-} --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=$NT"
export CUDA_VISIBLE_DEVICES=$(grep -oE "gpu[0-9]+" "$PBS_GPUFILE" | head -1 | grep -oE "[0-9]+")
source .venv/bin/activate
python -c "import jax; print('devices:', jax.devices())"

# Shared GPUs (about 15 people in the lab): do not start on a GPU someone else is already using.
# Jobs started outside PBS (Jupyter, ssh) can land on a GPU that PBS gave to us.
GPU_UUID=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=uuid --format=csv,noheader 2>/dev/null)
foreign_on_gpu() {
  local pid owner
  for pid in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null \
               | awk -F', ' -v u="$GPU_UUID" '$1 == u {print $2}'); do
    owner=$(ps -o user= -p "$pid" 2>/dev/null)
    [ -n "$owner" ] && [ "$owner" != "$(id -un)" ] && { echo "$owner pid $pid"; return 0; }
  done
  return 1
}
if who=$(foreign_on_gpu); then
  echo "=== GPU $CUDA_VISIBLE_DEVICES is already used by $who; not starting (resubmit later) ==="
  exit 3
fi

# Load the job definitions from server_run.sh and run them here (no nohup)
eval "$(sed -n '/^run_job() {/,/^}/p' scripts/server_run.sh)"
run_job "$JOB"
echo "=== done ==="
