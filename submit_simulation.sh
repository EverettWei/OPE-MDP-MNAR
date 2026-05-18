#!/bin/bash
#SBATCH --job-name=ope-mnar-sim
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=256G
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm-sim-%j.out
#SBATCH --error=slurm-sim-%j.err

# ── Cluster configuration: edit these two lines for your system ──────────────
CONDA_SH="${HOME}/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV="ope_mnar"
# ─────────────────────────────────────────────────────────────────────────────

source "$CONDA_SH"
conda activate "$CONDA_ENV"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[$(date '+%H:%M:%S')] Starting simulation sweeps (3 parallel jobs)..."

# Sweep 1: varying sample size (Figure 2 and Appendix)
python -u -m scripts.eval_grid \
    --mode size_x_missrate \
    --ns 64,128,256,512,1024,2048 --Ts 8 \
    --seeds 321:370 --mnar-c0s "0.3,-0.7,-1.5,-2.8" \
    --device cpu --n_workers 30 &
PID1=$!

# Sweep 2: varying horizon (Appendix)
python -u -m scripts.eval_grid \
    --mode horizon_x_missrate \
    --ns 512 --Ts 2,4,8,16,32 \
    --seeds 321:370 --mnar-c0s "0.3,-0.7,-1.5,-2.8" \
    --device cpu --n_workers 30 &
PID2=$!

# Sweep 3: varying reward type (Appendix)
python -u -m scripts.eval_grid \
    --mode reward_x_missrate \
    --ns 512 --Ts 8 \
    --seeds 321:370 --mnar-c0s "0.3,-0.7,-1.5,-2.8" \
    --reward-types "sigmoid,linear" \
    --device cpu --n_workers 30 &
PID3=$!

echo "[$(date '+%H:%M:%S')] PIDs: size=$PID1 horizon=$PID2 reward=$PID3"

wait $PID1; echo "[$(date '+%H:%M:%S')] size_x_missrate sweep done (exit=$?)"
wait $PID2; echo "[$(date '+%H:%M:%S')] horizon_x_missrate sweep done (exit=$?)"
wait $PID3; echo "[$(date '+%H:%M:%S')] reward_x_missrate sweep done (exit=$?)"

echo "[$(date '+%H:%M:%S')] Generating figures..."
python -m scripts.plot_ope_results --which all \
    --tables results/tables --outdir results/figures
echo "[$(date '+%H:%M:%S')] All simulation experiments finished."
