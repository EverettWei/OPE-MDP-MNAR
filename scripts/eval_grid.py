from __future__ import annotations
import os, argparse, time
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from scripts.simulation import compute_true_value_via_target_rollout
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Tuple


from src.OPE.fqe import ProxFQE, NaiveFQE, WeightedFQE
from src.generate_data import collect_episodes
from src.configs import EnvConfig
from src.envs.sim_envs import MNARMDP
from src.policies.behavior_policy import BehaviorPolicy
from src.policies.target_policy import TargetPolicy

def parse_int_list(spec: str):
    spec = str(spec).strip()
    if ":" in spec:
        lo, hi = spec.split(":")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",") if x]



def _run_single_job(job_args):
    """
    Run one (n, T, seed) job of the OPE grid.

    Parameters
    ----------
    job_args : tuple
        (job_idx, total_jobs, n, T, s, args_dict)

    Returns
    -------
    rows : list[dict]
        Result rows (naive / prox / ipw) for this job.
    """
    (
        job_idx,
        total_jobs,
        n,
        T,
        s,
        args_dict,
    ) = job_args

    # reconstruct a lightweight args object inside the worker
    class _Args:
        pass

    args = _Args()
    for k, v in args_dict.items():
        setattr(args, k, v)

    rows = []
    start_job = time.time()
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Job {job_idx}/{total_jobs}: "
          f"n={n}, T={T}, seed={s} ...")

    # ----- dataset generation -----
    lam_grid_abs = np.logspace(-7, 1, 30).tolist()

    try:
        cfg = EnvConfig(horizon=T, seed=s, gamma=args.gamma)
        env = MNARMDP(cfg)
        pi_b = BehaviorPolicy(seed=s + 11)
        ds = collect_episodes(env, pi_b, n_episodes=n, seed=s)
        if isinstance(ds, tuple):
            ds = ds[0]
        print(f"  ✓ Generated dataset with {len(ds['t'])} transitions.")
    except Exception as e:
        print(f"  ✗ Dataset generation failed: {e}")
        return rows 

    # ----- true value -----
    v_true = compute_true_value_via_target_rollout(T, args.gamma, s, n_eval=5000)
    print(f"  ✓ True value: {v_true:.4f}")

    # eval states / missing rate
    tvec = ds["t"].astype(int)
    S1 = ds["obs"][tvec == 1, :2]
    O0 = np.zeros((S1.shape[0],), dtype=np.float32)
    miss_rate = float(1.0 - np.mean(ds["o"]))

    # ----- NaiveFQE -----
    try:
        naive = NaiveFQE(
            action_list=(-1, +1),
            gamma=args.gamma,
            krr_kwargs=dict(
                lam_grid=lam_grid_abs,
                folds=5,
                device=args.device,
            ),
            device=args.device,
        ).fit(ds, TargetPolicy())
        v_naive = float(naive.value(S1, O0))
        print(f"  ✓ NaiveFQE done, value={v_naive:.4f}")
    except Exception as e:
        print(f"  ✗ NaiveFQE failed: {e}")
        v_naive = np.nan

    if not np.isnan(v_naive) and not np.isnan(v_true):
        b = v_naive - v_true
        ae = abs(b)
        rel = ae / max(abs(v_true), 1e-8)
        mse = b * b
    else:
        b = ae = rel = mse = np.nan

    rows.append(
        dict(
            method="naive",
            n=n,
            T=T,
            seed=s,
            value=v_naive,
            true=v_true,
            bias=b,
            mae=ae,
            mae_rel=rel,
            mse=mse,
            missing=miss_rate,
        )
    )

    # ----- ProxFQE -----
    try:
        prox = ProxFQE(
            action_list=(-1, +1),
            gamma=args.gamma,
            bridge_cv_kwargs=dict(
                delta_scale=args.delta_scale,
                delta_exp=args.delta_exp,
                gamma_f="auto",
                gamma_hs="auto",
                n_gamma_hs=30,
                cv=5,
                device=args.device,
            ),
            krr_kwargs=dict(
                lam_grid=lam_grid_abs,
                folds=5,
                device=args.device,
            ),
            device=args.device,
        ).fit(ds, TargetPolicy())
        v_prox = float(prox.value(S1, O0))
        print(f"  ✓ ProxFQE done, value={v_prox:.4f}")
    except Exception as e:
        print(f"  ✗ ProxFQE failed: {e}")
        v_prox = np.nan

    if not np.isnan(v_prox) and not np.isnan(v_true):
        b = v_prox - v_true
        ae = abs(b)
        rel = ae / max(abs(v_true), 1e-8)
        mse = b * b
    else:
        b = ae = rel = mse = np.nan

    rows.append(
        dict(
            method="prox",
            n=n,
            T=T,
            seed=s,
            value=v_prox,
            true=v_true,
            bias=b,
            mae=ae,
            mae_rel=rel,
            mse=mse,
            missing=miss_rate,
        )
    )

    # ----- IPW-FQE (WeightedFQE) -----
    try:
        ipw = WeightedFQE(
            action_list=(-1, +1),
            gamma=args.gamma,
            bridge_cv_kwargs=dict(
                delta_scale=args.delta_scale,
                delta_exp=args.delta_exp,
                gamma_f="auto",
                gamma_hs="auto",
                n_gamma_hs=30,
                cv=5,
                device=args.device,
            ),
            krr_kwargs=dict(
                lam_grid=lam_grid_abs,
                folds=5,
                device=args.device,
            ),
            logit_l2=1e-3,
            logit_max_iter=200,
            pmin=1e-2,
            w_cap=50.0,
            device=args.device,
        ).fit(ds, TargetPolicy())
        v_ipw = float(ipw.value(S1, O0))
        print(f"  ✓ IPW-FQE done, value={v_ipw:.4f}")
    except Exception as e:
        print(f"  ✗ IPW-FQE failed: {e}")
        v_ipw = np.nan

    if not np.isnan(v_ipw) and not np.isnan(v_true):
        b = v_ipw - v_true
        ae = abs(b)
        rel = ae / max(abs(v_true), 1e-8)
        mse = b * b
    else:
        b = ae = rel = mse = np.nan

    rows.append(
        dict(
            method="ipw",
            n=n,
            T=T,
            seed=s,
            value=v_ipw,
            true=v_true,
            bias=b,
            mae=ae,
            mae_rel=rel,
            mse=mse,
            missing=miss_rate,
        )
    )

    dur = time.time() - start_job
    print(f"  ✓ Finished job {job_idx}/{total_jobs} in {dur:.1f}s")

    return rows




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="size", choices=["size", "horizon"],
                    help="size: vary n (fix T); horizon: vary T (fix n)")
    ap.add_argument("--ns", type=str, default="200,500",
                    help="comma list or a:b; for mode=size this is varied; for mode=horizon this is fixed (one value)")
    ap.add_argument("--Ts", type=str, default="5",
                    help="comma list or a:b; for mode=horizon this is varied; for mode=size this is fixed (one value)")
    ap.add_argument("--seeds", type=str, default="1:3")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--delta-scale", type=float, default=5.0)
    ap.add_argument("--delta-exp", type=float, default=0.4)
    ap.add_argument("--outdir", type=str, default="results/tables")
    ap.add_argument("--n_workers", type=int, default=1,
                    help="Number of worker processes for parallel evaluation (1 = no parallel).")

    args = ap.parse_args()

    ns = parse_int_list(args.ns)
    Ts = parse_int_list(args.Ts)
    seeds = parse_int_list(args.seeds)

    # enforce one-dimension sweep
    if args.mode == "size" and len(Ts) != 1:
        raise ValueError("mode=size expects exactly one T (use --Ts 10 for example).")
    if args.mode == "horizon" and len(ns) != 1:
        raise ValueError("mode=horizon expects exactly one n (use --ns 500 for example).")

    os.makedirs(args.outdir, exist_ok=True)

    suffix = "size" if args.mode == "size" else "horizon"
    raw_csv = os.path.join(args.outdir, f"ope_runs_{suffix}.csv")
    summary_csv = os.path.join(args.outdir, f"summary_{suffix}.csv")

    rows = []
    start_all = time.time()

    total_jobs = len(ns) * len(Ts) * len(seeds)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] >>> Starting evaluation grid ({total_jobs} total jobs) <<<\n")

    # build job list
    jobs = []
    job_idx = 0
    args_dict = vars(args).copy()

    for T in Ts:
        for n in ns:
            for s in seeds:
                job_idx += 1
                jobs.append((job_idx, total_jobs, n, T, s, args_dict))
    # sequential vs parallel
    if args.n_workers == 1:
        for job in jobs:
            job_rows = _run_single_job(job)
            rows.extend(job_rows)
    else:
        max_workers = min(args.n_workers, len(jobs))
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Using {max_workers} worker processes.\n")
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            for job_rows in ex.map(_run_single_job, jobs):
                rows.extend(job_rows)

    # save raw
    df = pd.DataFrame(rows)
    df.to_csv(raw_csv, index=False)

    # summary
    summary = (df.groupby(["method","n","T"])
                 .agg(mean_value=("value","mean"),
                      true_mean=("true","mean"),
                      mean_bias=("bias","mean"),
                      mean_mae=("mae","mean"),
                      mean_mae_rel=("mae_rel","mean"),
                      mean_mse=("mse","mean"),
                      sd=("value","std"),
                      n_rep=("value","size"),
                      missing=("missing","mean"))
                 .reset_index())
    summary.to_csv(summary_csv, index=False)

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] >>> All jobs finished in {time.time()-start_all:.1f}s <<<")
    print(f"Saved raw results to {raw_csv}")
    print(f"Saved summary to {summary_csv}\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()