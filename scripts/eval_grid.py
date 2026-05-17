from __future__ import annotations
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import argparse, time
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from scripts.simulation import compute_true_value_via_target_rollout
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Tuple


from src.OPE.fqe import ProxFQE, NaiveFQE, WeightedFQE, ImputeFQE, SCOPE
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

    mnar_c0 = getattr(args, 'mnar_c0', 1.0)
    reward_type = getattr(args, 'reward_type', 'sigmoid')

    try:
        cfg = EnvConfig(horizon=T, seed=s, gamma=args.gamma,
                        mnar_c0=mnar_c0, reward_type=reward_type)
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
    v_true = compute_true_value_via_target_rollout(
        T, args.gamma, s, n_eval=5000,
        mnar_c0=mnar_c0, reward_type=reward_type)
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

    # ----- ImputeFQE -----
    try:
        impute = ImputeFQE(
            action_list=(-1, +1),
            gamma=args.gamma,
            krr_kwargs=dict(
                lam_grid=lam_grid_abs,
                folds=5,
                device=args.device,
            ),
            device=args.device,
        ).fit(ds, TargetPolicy())
        v_impute = float(impute.value(S1, O0))
        print(f"  ✓ ImputeFQE done, value={v_impute:.4f}")
    except Exception as e:
        print(f"  ✗ ImputeFQE failed: {e}")
        v_impute = np.nan

    if not np.isnan(v_impute) and not np.isnan(v_true):
        b = v_impute - v_true
        ae = abs(b)
        rel = ae / max(abs(v_true), 1e-8)
        mse = b * b
    else:
        b = ae = rel = mse = np.nan

    rows.append(
        dict(
            method="impute",
            n=n,
            T=T,
            seed=s,
            value=v_impute,
            true=v_true,
            bias=b,
            mae=ae,
            mae_rel=rel,
            mse=mse,
            missing=miss_rate,
        )
    )

    # ----- SCOPE -----
    try:
        scope = SCOPE(
            gamma=args.gamma,
            frac_shape=0.3,
            krr_kwargs=dict(
                lam_grid=lam_grid_abs,
                folds=5,
                device=args.device,
            ),
            w_cap=50.0,
            device=args.device,
        ).fit(ds, TargetPolicy(), BehaviorPolicy(seed=s + 11))
        v_scope = float(scope.value())
        print(f"  ✓ SCOPE done, value={v_scope:.4f}")
    except Exception as e:
        print(f"  ✗ SCOPE failed: {e}")
        v_scope = np.nan

    if not np.isnan(v_scope) and not np.isnan(v_true):
        b = v_scope - v_true
        ae = abs(b)
        rel = ae / max(abs(v_true), 1e-8)
        mse = b * b
    else:
        b = ae = rel = mse = np.nan

    rows.append(
        dict(
            method="scope",
            n=n,
            T=T,
            seed=s,
            value=v_scope,
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




def _parse_float_list(spec: str):
    '''Parse comma-separated floats.'''
    return [float(x.strip()) for x in spec.split(",") if x.strip()]


def _make_summary(df, group_cols):
    '''Aggregate results over seeds.'''
    return (df.groupby(group_cols)
              .agg(mean_value=("value", "mean"),
                   true_mean=("true", "mean"),
                   mean_bias=("bias", "mean"),
                   mean_mae=("mae", "mean"),
                   mean_mae_rel=("mae_rel", "mean"),
                   mean_mse=("mse", "mean"),
                   sd=("value", "std"),
                   n_rep=("value", "size"),
                   missing=("missing", "mean"))
              .reset_index())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="size",
                    choices=["size", "horizon", "missrate", "reward_type",
                             "size_x_missrate", "horizon_x_missrate", "reward_x_missrate"])
    ap.add_argument("--ns", type=str, default="200,500",
                    help="comma list or a:b")
    ap.add_argument("--Ts", type=str, default="5",
                    help="comma list or a:b")
    ap.add_argument("--seeds", type=str, default="1:3")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--delta-scale", type=float, default=5.0)
    ap.add_argument("--delta-exp", type=float, default=0.4)
    ap.add_argument("--outdir", type=str, default="results/tables")
    ap.add_argument("--n_workers", type=int, default=1,
                    help="Number of worker processes for parallel evaluation (1 = no parallel).")
    # new args for missrate / reward_type sweeps
    ap.add_argument("--mnar-c0s", type=str, default="0.3,-0.7,-1.5,-2.8",
                    help="Comma-separated MNAR intercepts (mode=missrate). "
                         "Defaults give ~20%%,40%%,60%%,80%% missing.")
    ap.add_argument("--reward-types", type=str, default="sigmoid,linear",
                    help="Comma-separated reward types (mode=reward_type).")
    ap.add_argument("--mnar-c0", type=float, default=1.0,
                    help="Fixed MNAR intercept for modes other than missrate.")
    ap.add_argument("--reward-type", type=str, default="sigmoid",
                    help="Fixed reward type for modes other than reward_type.")

    args = ap.parse_args()

    ns = parse_int_list(args.ns)
    Ts = parse_int_list(args.Ts)
    seeds = parse_int_list(args.seeds)

    # enforce one-dimension sweep (not for cross-product modes)
    if args.mode == "size" and len(Ts) != 1:
        raise ValueError("mode=size expects exactly one T (use --Ts 10 for example).")
    if args.mode == "horizon" and len(ns) != 1:
        raise ValueError("mode=horizon expects exactly one n (use --ns 500 for example).")

    os.makedirs(args.outdir, exist_ok=True)

    suffix = args.mode
    raw_csv = os.path.join(args.outdir, f"ope_runs_{suffix}.csv")
    summary_csv = os.path.join(args.outdir, f"summary_{suffix}.csv")

    rows = []
    start_all = time.time()

    # ---- build job list ----
    jobs = []
    job_idx = 0

    if args.mode in ("size", "horizon"):
        args_dict = vars(args).copy()
        args_dict["mnar_c0"] = args.mnar_c0
        args_dict["reward_type"] = args.reward_type
        for T in Ts:
            for n in ns:
                for s in seeds:
                    job_idx += 1
                    jobs.append((job_idx, 0, n, T, s, args_dict))

    elif args.mode == "missrate":
        c0s = _parse_float_list(args.mnar_c0s)
        if len(ns) != 1 or len(Ts) != 1:
            raise ValueError("mode=missrate expects exactly one n and one T.")
        n_fix, T_fix = ns[0], Ts[0]
        for c0 in c0s:
            for s in seeds:
                job_idx += 1
                ad = vars(args).copy()
                ad["mnar_c0"] = c0
                ad["reward_type"] = args.reward_type
                jobs.append((job_idx, 0, n_fix, T_fix, s, ad))

    elif args.mode == "reward_type":
        rtypes = [x.strip() for x in args.reward_types.split(",") if x.strip()]
        if len(ns) != 1 or len(Ts) != 1:
            raise ValueError("mode=reward_type expects exactly one n and one T.")
        n_fix, T_fix = ns[0], Ts[0]
        for rt in rtypes:
            for s in seeds:
                job_idx += 1
                ad = vars(args).copy()
                ad["mnar_c0"] = args.mnar_c0
                ad["reward_type"] = rt
                jobs.append((job_idx, 0, n_fix, T_fix, s, ad))

    elif args.mode == "size_x_missrate":
        c0s = _parse_float_list(args.mnar_c0s)
        if len(Ts) != 1:
            raise ValueError("mode=size_x_missrate expects exactly one T.")
        T_fix = Ts[0]
        for c0 in c0s:
            for n in ns:
                for s in seeds:
                    job_idx += 1
                    ad = vars(args).copy()
                    ad["mnar_c0"] = c0
                    ad["reward_type"] = args.reward_type
                    jobs.append((job_idx, 0, n, T_fix, s, ad))

    elif args.mode == "horizon_x_missrate":
        c0s = _parse_float_list(args.mnar_c0s)
        if len(ns) != 1:
            raise ValueError("mode=horizon_x_missrate expects exactly one n.")
        n_fix = ns[0]
        for c0 in c0s:
            for T in Ts:
                for s in seeds:
                    job_idx += 1
                    ad = vars(args).copy()
                    ad["mnar_c0"] = c0
                    ad["reward_type"] = args.reward_type
                    jobs.append((job_idx, 0, n_fix, T, s, ad))

    elif args.mode == "reward_x_missrate":
        c0s = _parse_float_list(args.mnar_c0s)
        rtypes = [x.strip() for x in args.reward_types.split(",") if x.strip()]
        if len(ns) != 1 or len(Ts) != 1:
            raise ValueError("mode=reward_x_missrate expects exactly one n and one T.")
        n_fix, T_fix = ns[0], Ts[0]
        for c0 in c0s:
            for rt in rtypes:
                for s in seeds:
                    job_idx += 1
                    ad = vars(args).copy()
                    ad["mnar_c0"] = c0
                    ad["reward_type"] = rt
                    jobs.append((job_idx, 0, n_fix, T_fix, s, ad))

    total_jobs = len(jobs)
    # patch total_jobs into each tuple
    jobs = [(j[0], total_jobs, *j[2:]) for j in jobs]

    print(f"[{datetime.now().strftime('%H:%M:%S')}] >>> Starting evaluation grid "
          f"(mode={args.mode}, {total_jobs} total jobs) <<<\n")

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

    # add mnar_c0 and reward_type columns from job args
    # reconstruct from job order: each job produces 5 method rows
    n_methods = 5
    if args.mode == "missrate":
        c0s = _parse_float_list(args.mnar_c0s)
        c0_col, rt_col = [], []
        for c0 in c0s:
            for _s in seeds:
                for _m in range(n_methods):
                    c0_col.append(c0)
                    rt_col.append(args.reward_type)
        if len(c0_col) == len(df):
            df["mnar_c0"] = c0_col
            df["reward_type"] = rt_col
    elif args.mode == "reward_type":
        rtypes = [x.strip() for x in args.reward_types.split(",") if x.strip()]
        c0_col, rt_col = [], []
        for rt in rtypes:
            for _s in seeds:
                for _m in range(n_methods):
                    c0_col.append(args.mnar_c0)
                    rt_col.append(rt)
        if len(rt_col) == len(df):
            df["mnar_c0"] = c0_col
            df["reward_type"] = rt_col
    elif args.mode == "size_x_missrate":
        c0s = _parse_float_list(args.mnar_c0s)
        c0_col = []
        for c0 in c0s:
            for n in ns:
                for _s in seeds:
                    for _m in range(n_methods):
                        c0_col.append(c0)
        if len(c0_col) == len(df):
            df["mnar_c0"] = c0_col
    elif args.mode == "horizon_x_missrate":
        c0s = _parse_float_list(args.mnar_c0s)
        c0_col = []
        for c0 in c0s:
            for T in Ts:
                for _s in seeds:
                    for _m in range(n_methods):
                        c0_col.append(c0)
        if len(c0_col) == len(df):
            df["mnar_c0"] = c0_col
    elif args.mode == "reward_x_missrate":
        c0s = _parse_float_list(args.mnar_c0s)
        rtypes = [x.strip() for x in args.reward_types.split(",") if x.strip()]
        c0_col, rt_col = [], []
        for c0 in c0s:
            for rt in rtypes:
                for _s in seeds:
                    for _m in range(n_methods):
                        c0_col.append(c0)
                        rt_col.append(rt)
        if len(c0_col) == len(df):
            df["mnar_c0"] = c0_col
            df["reward_type"] = rt_col

    df.to_csv(raw_csv, index=False)

    # summary
    if args.mode in ("missrate", "size_x_missrate", "horizon_x_missrate"):
        group = ["method", "n", "T", "mnar_c0"] if "mnar_c0" in df.columns else ["method", "n", "T"]
    elif args.mode in ("reward_type", "reward_x_missrate"):
        group = ["method", "n", "T", "mnar_c0", "reward_type"] if "mnar_c0" in df.columns else ["method", "n", "T", "reward_type"]
    else:
        group = ["method", "n", "T"]

    summary = _make_summary(df, group)
    summary.to_csv(summary_csv, index=False)

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] >>> All jobs finished in {time.time()-start_all:.1f}s <<<")
    print(f"Saved raw results to {raw_csv}")
    print(f"Saved summary to {summary_csv}\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()