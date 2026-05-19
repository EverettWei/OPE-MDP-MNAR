#!/bin/bash
# Run all simulation sweeps locally.
# Usage: bash run_simulation.sh
#
# HPC/SLURM users: add your #SBATCH directives here and submit with sbatch.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[$(date '+%H:%M:%S')] Starting simulation sweeps (3 parallel jobs)..."

# Sweep 1: varying sample size (Figure 2 and Appendix)
python -u -m scripts.eval_grid \
    --mode size_x_missrate \
    --ns 64,128,256,512,1024,2048 --Ts 8 \
    --seeds 321:370 --mnar-c0s "0.3,-0.7,-1.5,-2.8" \
    --device cpu --n_workers 4 &
PID1=$!

# Sweep 2: varying horizon (Appendix)
python -u -m scripts.eval_grid \
    --mode horizon_x_missrate \
    --ns 512 --Ts 2,4,8,16,32 \
    --seeds 321:370 --mnar-c0s "0.3,-0.7,-1.5,-2.8" \
    --device cpu --n_workers 4 &
PID2=$!

# Sweep 3: varying reward type (Appendix)
python -u -m scripts.eval_grid \
    --mode reward_x_missrate \
    --ns 512 --Ts 8 \
    --seeds 321:370 --mnar-c0s "0.3,-0.7,-1.5,-2.8" \
    --reward-types "sigmoid,linear" \
    --device cpu --n_workers 4 &
PID3=$!

echo "[$(date '+%H:%M:%S')] PIDs: size=$PID1 horizon=$PID2 reward=$PID3"

wait $PID1; echo "[$(date '+%H:%M:%S')] size_x_missrate sweep done (exit=$?)"
wait $PID2; echo "[$(date '+%H:%M:%S')] horizon_x_missrate sweep done (exit=$?)"
wait $PID3; echo "[$(date '+%H:%M:%S')] reward_x_missrate sweep done (exit=$?)"

echo "[$(date '+%H:%M:%S')] Generating figures..."
python -m scripts.plot_ope_results --which all \
    --tables results/tables --outdir results/figures
echo "[$(date '+%H:%M:%S')] All simulation experiments finished."
