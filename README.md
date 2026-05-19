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

run_simulation.sh       Run simulation studies locally (bash run_simulation.sh)
run_realdata.sh         Run real-data experiment locally (bash run_realdata.sh)
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

## 2. Running the experiments

Activate your environment and run the scripts directly from the repository root:

```bash
# Simulation studies
bash run_simulation.sh

# Real-data experiment (GPU recommended; edit --device cpu inside the script if unavailable)
bash run_realdata.sh
```

Each script runs all sub-jobs in parallel internally and prints per-job exit codes for easy debugging.

**HPC/SLURM users:** add `#SBATCH` directives at the top of each script and submit with `sbatch` instead.

---

## 3. Pre-computed results

Pre-computed results from the paper are already included under `results/tables/`
(simulation) and `realdata/results/` (real-data). The scripts above will
regenerate them from scratch if needed.

---

## Citation

```bibtex
@inproceedings{wei2026ope,
  title     = {Off-Policy Evaluation for Missingness-Aware Policies in {MDP}s with Rewards Missing Not at Random},
  author    = {Wei, Ziheng and Qu, Annie and Miao, Rui},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  year      = {2026}
}
```
