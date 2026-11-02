#!/bin/bash
# 研究室サーバー（fifa / copaam など）での初回セットアップ
#   使い方:  bash scripts/server_setup.sh          # CPU のみ
#            bash scripts/server_setup.sh gpu      # NVIDIA GPU（CUDA 12）を使う
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}
"$PY" -m venv .venv
source .venv/bin/activate
python -m pip install -q --upgrade pip
python -m pip install -q numpy scipy matplotlib
if [ "${1:-cpu}" = "gpu" ]; then
  python -m pip install -q "jax[cuda12]"
else
  python -m pip install -q jax
fi
python - <<'EOF'
import jax, numpy, scipy
print("numpy", numpy.__version__, "| scipy", scipy.__version__, "| jax", jax.__version__)
print("devices:", jax.devices())
EOF
echo "セットアップ完了．次は: bash scripts/server_run.sh <ジョブ名>"
