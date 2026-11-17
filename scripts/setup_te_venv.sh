#!/bin/bash
# Set up the authors' TE-QPINN code (Berger, Hosters, Moeller, Sci. Rep. 15, 18823 (2025)) and a separate venv for it.
# Run once on copaam from the repo root (login node is fine: this only downloads and installs).
#   bash scripts/setup_te_venv.sh
# The clone goes to ~/te-qpinns (outside this repo), the venv to .venv_te (torch + PennyLane; .venv stays JAX only).
set -e
REPO=${TE_REPO:-$HOME/te-qpinns}
COMMIT=f7d0177ed33ebd674fe9ed0fd846d18a5bf0129e
[ -d "$REPO/.git" ] || git clone https://git.rwth-aachen.de/berger.st.11.11/te-qpinns.git "$REPO"
git -C "$REPO" checkout -q "$COMMIT"
echo "authors' code: $REPO at $(git -C "$REPO" rev-parse --short HEAD)"
[ -d .venv_te ] || python3 -m venv .venv_te
P=.venv_te/bin/pip
$P install -q --upgrade pip
# the versions in the authors' requirements.txt (torch 2.4.0+cu124, PennyLane 0.37.0)
if $P install -q torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124 \
   && $P install -q PennyLane==0.37.0 autoray==0.6.12 numpy==1.26.3 scipy==1.14.0 matplotlib==3.9.1 pandas==2.2.2; then
  echo "installed the authors' pinned versions"
else
  echo "the pinned versions did not install (Python too new?); installing current versions instead"
  $P install -q torch pennylane numpy scipy matplotlib pandas
fi
.venv_te/bin/python -c "import torch, pennylane; print('torch', torch.__version__, '| pennylane', pennylane.__version__, '| cuda build', torch.version.cuda)"
echo "=== setup done ==="
