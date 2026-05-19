#!/bin/bash
# Run real-data (MIMIC-III sepsis) experiments locally.
# Usage: bash run_realdata.sh
#
# HPC/SLURM users: add your #SBATCH directives here and submit with sbatch.
# GPU is recommended; set --device cpu below if unavailable.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES=0

Q_HIDDEN="512,512,256"
BRIDGE_HIDDEN="512,512,256"
AUX_HIDDEN="256,256"
COMMON="--device cuda --gamma 1.0 --bridge_steps 6000 --q_steps 8000 --batch_size 4096 \
  --q_hidden $Q_HIDDEN --bridge_hidden $BRIDGE_HIDDEN --aux_hidden $AUX_HIDDEN"

# Step 1: clean raw data → 48 state features, T=10
# Skip if already done (requires realdata/sepsis_processed_state_action.csv from MIMIC-III)
if [ ! -f realdata/sepsis_T10.csv ]; then
    echo "[$(date '+%H:%M:%S')] Step 1: cleaning raw sepsis data..."
    python3 -u sepsis/clean_sepsis.py \
        --input realdata/sepsis_processed_state_action.csv \
        --outdir realdata
fi

# Step 2: train Double DQN target policy
# Skip if checkpoint already exists
if [ ! -f realdata/dqn_sepsis.pt ]; then
    echo "[$(date '+%H:%M:%S')] Step 2: training DQN target policy..."
    python3 -u sepsis/train_dqn_sepsis.py \
        --input realdata/sepsis_T10.csv \
        --outdir realdata \
        --gamma 1.0 --seed 42 --n_steps 50000
fi

# Step 3: apply MNAR mechanism at four missing rates (20%, 40%, 60%, 80%)
echo "[$(date '+%H:%M:%S')] Step 3: applying MNAR missingness..."
python3 -u sepsis/apply_mnar.py \
    --input realdata/sepsis_T10_with_targets.csv \
    --outdir realdata \
    --miss-rates 0.2,0.4,0.6,0.8

# Step 4: evaluate all OPE methods — run four missing rates in parallel
echo "[$(date '+%H:%M:%S')] Step 4: evaluating OPE methods (4 rates in parallel)..."
mkdir -p realdata/results/mnar20 realdata/results/mnar40 \
         realdata/results/mnar60 realdata/results/mnar80

python3 -u sepsis/eval_ope.py \
    --input realdata/sepsis_T10_mnar20_ope.csv --seed 42 $COMMON \
    --outdir realdata/results/mnar20 > realdata/results/mnar20.log 2>&1 &
PID1=$!

python3 -u sepsis/eval_ope.py \
    --input realdata/sepsis_T10_mnar40_ope.csv --seed 42 $COMMON \
    --outdir realdata/results/mnar40 > realdata/results/mnar40.log 2>&1 &
PID2=$!

python3 -u sepsis/eval_ope.py \
    --input realdata/sepsis_T10_mnar60_ope.csv --seed 42 $COMMON \
    --outdir realdata/results/mnar60 > realdata/results/mnar60.log 2>&1 &
PID3=$!

python3 -u sepsis/eval_ope.py \
    --input realdata/sepsis_T10_mnar80_ope.csv --seed 42 $COMMON \
    --outdir realdata/results/mnar80 > realdata/results/mnar80.log 2>&1 &
PID4=$!

echo "[$(date '+%H:%M:%S')] PIDs: mnar20=$PID1 mnar40=$PID2 mnar60=$PID3 mnar80=$PID4"

wait $PID1; echo "[$(date '+%H:%M:%S')] mnar20 done (exit=$?)"
wait $PID2; echo "[$(date '+%H:%M:%S')] mnar40 done (exit=$?)"
wait $PID3; echo "[$(date '+%H:%M:%S')] mnar60 done (exit=$?)"
wait $PID4; echo "[$(date '+%H:%M:%S')] mnar80 done (exit=$?)"

echo "[$(date '+%H:%M:%S')] Generating figures..."
python3 -m scripts.plot_sepsis_ope
echo "[$(date '+%H:%M:%S')] All real-data experiments finished."
