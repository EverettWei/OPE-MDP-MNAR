# OPE for MDPs with reward MNAR simulation

This repository contains code to reproduce the simulation results for proximal off-policy evaluation (OPE) with MNAR rewards.

## 1. Setup

Recommended: Python 3.11

Create a virtual environment and install dependencies:

```bash
conda create -n sim_mnar python=3.11 -y
conda activate sim_mnar
python -m pip install --upgrade pip
pip install -r requirements.txt
```



## 2. Run simulations (generate tables)

```bash
python -m scripts.eval_grid --mode size --ns 64,128,256,512,1024,2048 --gamma 1 --Ts 8 --seeds 321:325 --device cpu --n_workers 5
python -m scripts.eval_grid --mode horizon --ns 512 --gamma 1 --Ts 2,4,8,16,32 --seeds 321:325 --device cpu --n_workers 5
```

Tables will be saved under:

- results/tables

## 3. Plot figures

```bash
python -m scripts.plot_ope_results --which size --tables results/tables --outdir results/figures
python -m scripts.plot_ope_results --which horizon --tables results/tables --outdir results/figures
python -m scripts.plot_ope_results --which panel --tables results/tables --outdir results/figures
```

Figures will be saved under:

- results/figures
