# Off-Policy Evaluation for Missingness-Aware Policies in MDPs with Rewards Missing Not at Random

This repository contains code to reproduce the simulation and real-data experiments in:

> **Off-Policy Evaluation for Missingness-Aware Policies in MDPs with Rewards Missing Not at Random**
> Ziheng Wei, Annie Qu, Rui Miao. *ICML 2026.*

---

## Repository structure

```
src/                    Core library
  configs.py            Environment configuration
  utils.py              Shared utilities
  generate_data.py      Offline data generation under behavior policy
  envs/sim_envs.py      Simulated MDP with MNAR reward mechanism
  policies/             Behavior and target policy definitions
  OPE/
    rkhs.py             RKHS kernel primitives for min-max bridge estimation
    fqe.py              OPE estimators: ProxFQE, NaiveFQE, WeightedFQE, ImputeFQE, SCOPE
    nn_bridge.py        Neural network bridge estimator (AGMM)

scripts/                Experiment entry points
  eval_grid.py          Simulation sweep over (n, T, seed, missing rate)
  simulation.py         True value computation via target policy rollout
  run_simulation.py     Quick single-run wrapper for simulation.py
  plot_ope_results.py   Simulation figure generation
  plot_sepsis_ope.py    Sepsis figure generation
  viz_data.py           Data diagnostics and overview panel

sepsis/                 Real-data (MIMIC-III) experiment pipeline
  clean_sepsis.py       Clean raw sepsis data to 48-feature, T=10 format
  train_dqn_sepsis.py   Train Double DQN target policy
  apply_mnar.py         Apply MNAR mechanism at target missing rates
  eval_ope.py           Evaluate all OPE methods on sepsis data
  nn_fqe.py             Neural network FQE estimators for high-dimensional state

realdata/               Pre-computed sepsis results
  dqn_sepsis.pt         Trained DQN model checkpoint
  results/mnar{20,40,60,80}/ope_results.csv

results/                Pre-computed simulation results
  tables/               Raw and summary CSVs for all three sweeps
```

---

## 1. Setup

Recommended: Python 3.11

```bash
conda create -n ope_mnar python=3.11 -y
conda activate ope_mnar
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Cluster submission (SLURM)

Two SLURM batch scripts are provided for running on a cluster or GPU node.
Before submitting, open each script and set `CONDA_SH` and `CONDA_ENV` to
match your cluster's conda installation.

```bash
# Simulation studies (CPU, ~7 days for full 50-seed sweep)
sbatch submit_simulation.sh

# Real-data experiment (GPU, ~2 days for full pipeline)
sbatch submit_realdata.sh
```

Both scripts use relative paths and are designed to be submitted from the
repository root. Each script runs all sub-jobs in parallel internally and
prints per-job exit codes for easy debugging.

---

## 3. Simulation studies

### 3.1 Run the evaluation grid

The three sweeps correspond to the three sets of results reported in the paper
(varying sample size, varying horizon, and varying reward type), each crossed
with four MNAR missing rates (~20%, 40%, 60%, 80%).

**Full replication** (50 seeds, 30 parallel workers — requires a multi-core machine):

```bash
# Sweep 1: varying sample size  (Figure 2 and Appendix)
python -m scripts.eval_grid \
    --mode size_x_missrate \
    --ns 64,128,256,512,1024,2048 --Ts 8 \
    --seeds 321:370 --mnar-c0s "0.3,-0.7,-1.5,-2.8" \
    --gamma 1 --device cpu --n_workers 30

# Sweep 2: varying horizon  (Appendix)
python -m scripts.eval_grid \
    --mode horizon_x_missrate \
    --ns 512 --Ts 2,4,8,16,32 \
    --seeds 321:370 --mnar-c0s "0.3,-0.7,-1.5,-2.8" \
    --gamma 1 --device cpu --n_workers 30

# Sweep 3: varying reward type  (Appendix)
python -m scripts.eval_grid \
    --mode reward_x_missrate \
    --ns 512 --Ts 8 \
    --seeds 321:370 --mnar-c0s "0.3,-0.7,-1.5,-2.8" \
    --reward-types "sigmoid,linear" \
    --gamma 1 --device cpu --n_workers 30
```

**Quick test** (5 seeds, 5 workers):

```bash
python -m scripts.eval_grid \
    --mode size_x_missrate \
    --ns 64,128,256,512,1024,2048 --Ts 8 \
    --seeds 321:325 --mnar-c0s "0.3,-0.7,-1.5,-2.8" \
    --gamma 1 --device cpu --n_workers 5
```

Results are saved to `results/tables/`. Pre-computed results from the paper are already included.

### 3.2 Plot figures

```bash
python -m scripts.plot_ope_results --which all \
    --tables results/tables --outdir results/figures
```

Figures are saved to `results/figures/`.

---

## 4. Real-data experiment (MIMIC-III sepsis)

**Data access:** The raw MIMIC-III data requires credentialed access via
[PhysioNet](https://physionet.org/content/mimiciii/). Follow the pre-processing
pipeline of Raghu et al. (2017) to obtain `sepsis_processed_state_action.csv`,
then place it under `realdata/`.

### 4.1 Full pipeline

```bash
# Step 1: clean raw data → 48 state features, T=10
python3 sepsis/clean_sepsis.py \
    --input realdata/sepsis_processed_state_action.csv \
    --outdir realdata

# Step 2: train Double DQN target policy
python3 sepsis/train_dqn_sepsis.py \
    --input realdata/sepsis_T10.csv \
    --outdir realdata \
    --gamma 1.0 --seed 42 --n_steps 50000

# Step 3: apply MNAR mechanism at four missing rates
python3 sepsis/apply_mnar.py \
    --input realdata/sepsis_T10_with_targets.csv \
    --outdir realdata \
    --miss-rates 0.2,0.4,0.6,0.8

# Step 4: evaluate all OPE methods (run once per missing rate)
for RATE in 20 40 60 80; do
    python3 sepsis/eval_ope.py \
        --input realdata/sepsis_T10_mnar${RATE}_ope.csv \
        --outdir realdata/results/mnar${RATE} \
        --gamma 1.0 --seed 42 --device cuda \
        --bridge_steps 6000 --q_steps 8000 --batch_size 4096 \
        --q_hidden 512,512,256 --bridge_hidden 512,512,256 --aux_hidden 256,256
done
```

### 4.2 Plot figures

```bash
python3 -m scripts.plot_sepsis_ope
```

Figures are saved to `realdata/results/figures/`. Pre-computed results from the
paper are already included under `realdata/results/`.

---

## Citation

```bibtex
@inproceedings{wei2026ope,
  title     = {Off-Policy Evaluation for Missingness-Aware Policies in {MDP}s with Rewards Missing Not at Random},
  author    = {Wei, Ziheng and Qu, Annie and Miao, Rui},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026}
}
```
