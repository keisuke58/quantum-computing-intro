#!/bin/bash
# 研究室サーバーで実験を回す．結果は results/server/，ログは logs/ に出る．
#   使い方:  bash scripts/server_run.sh <ジョブ名>
#   ジョブ:
#     seeds5   5 シードで数値を確定（TSV 単体・古典対照・TSV 周期配列・ショットノイズ N=10）
#     tsv      TSV 単体 + 古典対照のみ（5 シード）
#     array    TSV 周期配列のみ（5 シード）
#     shots    ショットノイズ対策のみ（N=10，強形式 vs エネルギー形，5 シード）
#     bigN     量子ビット数を増やした TSV 単体（N=10, 12，3 シード）
#     tsv3d_plane  3D unit cell TSV vs approx. plane-strain reference (N=8,10)
#     tsv3d    3D unit cell TSV vs Abaqus reference (needs results/abaqus from fifa)
#     edge_single / edge_axisym / edge_3d / edge_plane / edge_periodic  re-runs
#     pf_ac / pf_ch  phase field: Allen-Cahn and Cahn-Hilliard (PFN=8|10, PFSEEDS) with wider scale grids (best scale was at the grid edge)
#     axisym   軸対称 (r, z) の TSV 1 本（MORE-Stress の寸法，上下固定 / 上面自由，5 シード）
#   接続が切れても止まらないよう，nohup で裏で動かす．進み具合: tail -f logs/<ジョブ名>.log
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p results/server logs
JOB=${1:?ジョブ名を指定してください（seeds5 / tsv / array / shots / bigN / axisym / tsv3d_plane / tsv3d）}
export XLA_PYTHON_CLIENT_PREALLOCATE=false     # GPU メモリを一度に確保しない

run_job() {
  case "$1" in
    tsv)
      python 30_qerc_tsv_single.py --seeds 5 --out results/server/tsv_single_5s.json
      python 31_tsv_classical_controls.py --seeds 5 --out results/server/tsv_controls_5s.json ;;
    array)
      python 34_tsv_periodic.py --seeds 5 --out results/server/tsv_periodic_5s.json ;;
    shots)
      python 39_shot_mitigation.py --Ns 10 --methods strong,ritz --seeds 5 --out results/server/shot_mitigation_5s.json ;;
    seeds5)
      run_job tsv; run_job array; run_job shots ;;
    axisym)
      python 42_tsv_axisym.py --seeds 5 --h 0.02 --out results/server/tsv_axisym.json ;;
    tsv3d_plane)
      for N in 8 10; do
        python 44_tsv_3d.py --ref plane --bc clamped --N $N --seeds 3 --out results/server/tsv_3d_plane_N${N}.json
      done ;;
    tsv3d)
      for N in 8 10; do
        python 44_tsv_3d.py --ref abaqus --bc ${BC3D:-clamped} --N $N --seeds 3 --out results/server/tsv_3d_abaqus_${BC3D:-clamped}_N${N}.json
      done ;;
    edge_single)
      python 30_qerc_tsv_single.py --N 10 --models Fourier --scales 16,32,64,128 --seeds 3 --out results/server/tsv_single_N10_fourier_wide.json
      python 31_tsv_classical_controls.py --models tensor-RP --scales 0.5,1 --seeds 5 --out results/server/tsv_controls_rp_low.json ;;
    edge_axisym)
      python 42_tsv_axisym.py --models QERC-Haar,MLP2-RF --scales 0.25,0.5 --seeds 5 --h 0.02 --out results/server/tsv_axisym_low.json ;;
    edge_3d)
      for N in 8 10; do
        python 44_tsv_3d.py --ref abaqus --bc ${BC3D:-clamped} --N $N --models QERC-Haar,MLP2-RF,tanh-RF --scales 0.25,0.5,1 --seeds 3 --out results/server/tsv_3d_abaqus_${BC3D:-clamped}_N${N}_low.json
      done ;;
    edge_plane)
      for N in 8 10; do
        python 44_tsv_3d.py --ref plane --bc clamped --N $N --models QERC-Haar,MLP2-RF --scales 0.25,0.5 --seeds 3 --out results/server/tsv_3d_plane_N${N}_low.json
      done ;;
    edge_periodic)
      python 34_tsv_periodic.py --L 3 --models Fourier --scales 32,64,128 --seeds 5 --out results/server/tsv_periodic_L3_fourier_wide.json ;;
    pf_ac)
      python 52_allen_cahn_qerc.py --D 1e-4 --N ${PFN:-8} --seeds 5 --out results/server/allen_cahn_N${PFN:-8}.json ;;
    pf_ch)
      python 54_cahn_hilliard_qerc.py --eps2 0.02 --N ${PFN:-8} --seeds ${PFSEEDS:-5} --out results/server/cahn_hilliard_N${PFN:-8}.json ;;
    bigN)
      for N in 10 12; do
        python 30_qerc_tsv_single.py --N $N --seeds 3 --out results/server/tsv_single_N${N}.json
      done ;;
    *) echo "不明なジョブ: $1"; exit 1 ;;
  esac
}

nohup bash -c "$(declare -f run_job); run_job $JOB; echo '=== 完了 ==='" > "logs/$JOB.log" 2>&1 &
echo "開始しました（PID $!）．進み具合: tail -f logs/$JOB.log"
