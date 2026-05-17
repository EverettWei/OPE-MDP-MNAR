"""
Evaluate OPE methods on sepsis MNAR data.

Takes the OPE data (original actions + DQN target actions, MNAR rewards)
and splits 60/40 by patient ID (seeded) for FQE fitting / evaluation.

Compares:
  1. OracleFQE   — uses r_true (ground truth baseline)
  2. NaiveFQE    — uses only O_t=1 data (biased under MNAR)
  3. ImputeFQE   — NN regression imputation + FQE (biased under MNAR)
  4. IPW-FQE     — inverse propensity weighted FQE
  5. SCOPE       — reward shaping + per-step IS (Parbhoo et al. 2020)
  6. ProxFQE     — our method: NN bridge imputation + FQE
"""

import argparse
import numpy as np
import pandas as pd
import torch
import os

from nn_fqe import (NNProxFQE, NNNaiveFQE, NNOracleFQE,
                     NNImputeFQE, NNIPWFQE, NNSCOPE)


def load_and_split(csv_path, seed=42, train_frac=0.6):
    """Load OPE CSV, split by patient ID (seeded), return fit/test data dicts."""
    df = pd.read_csv(csv_path)

    # Deterministic patient-level split
    ids = np.sort(df['icustayid'].unique())
    rng = np.random.RandomState(seed)
    rng.shuffle(ids)
    n_fit = int(len(ids) * train_frac)
    fit_ids = set(ids[:n_fit])
    test_ids = set(ids[n_fit:])

    df_fit = df[df['icustayid'].isin(fit_ids)].copy()
    df_test = df[df['icustayid'].isin(test_ids)].copy()

    return _df_to_data(df_fit), _df_to_data(df_test), n_fit, len(ids) - n_fit


def _df_to_data(df):
    """Convert dataframe to data dict for FQE."""
    # State: paper's 48 features (everything except identifiers, actions, reward, MNAR cols)
    drop = ['icustayid', 'vaso_input', 'iv_input', 'reward',
            'o_t', 'r_obs', 'r_true', 'o_prev', 'vaso_target', 'iv_target']
    state_cols = [c for c in df.columns if c not in drop]

    states = df[state_cols].values.astype(np.float32)

    T = 10
    next_states = np.zeros_like(states)
    dones = np.zeros(len(df), dtype=np.float32)
    bloc = df['bloc'].values
    icuids = df['icustayid'].values

    for i in range(len(df) - 1):
        if bloc[i] < T and icuids[i] == icuids[i + 1]:
            next_states[i] = states[i + 1]
        else:
            next_states[i] = states[i]
            dones[i] = 1.0
    next_states[-1] = states[-1]
    dones[-1] = 1.0

    a_joint = (df['vaso_input'].values * 5 + df['iv_input'].values).astype(np.int64)
    at_joint = (df['vaso_target'].values * 5 + df['iv_target'].values).astype(np.int64)

    # at_joint_next: target action at t+1 (for Bellman backup)
    at_joint_next = np.zeros_like(at_joint)
    for i in range(len(df) - 1):
        if bloc[i] < T and icuids[i] == icuids[i + 1]:
            at_joint_next[i] = at_joint[i + 1]
        else:
            at_joint_next[i] = at_joint[i]
    at_joint_next[-1] = at_joint[-1]

    return {
        'states': states,
        'next_states': next_states,
        'a_joint': a_joint,
        'at_joint': at_joint,
        'at_joint_next': at_joint_next,
        'r_true': df['r_true'].values.astype(np.float32),
        'r_obs': df['r_obs'].values.astype(np.float32),
        'o_t': df['o_t'].values.astype(np.int8),
        'o_prev': df['o_prev'].values.astype(np.int8),
        'dones': dones,
        'bloc': bloc.astype(np.int32),
        'state_cols': state_cols,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='realdata/sepsis_T10_mnar60_ope.csv')
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-frac', type=float, default=0.6)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--outdir', type=str, default='realdata/results')
    parser.add_argument('--bridge_steps', type=int, default=2000)
    parser.add_argument('--q_steps', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--q_hidden', type=str, default='128,128',
                        help='Comma-separated hidden layer sizes for Q-networks')
    parser.add_argument('--bridge_hidden', type=str, default='128,128',
                        help='Comma-separated hidden layer sizes for bridge networks')
    parser.add_argument('--aux_hidden', type=str, default='128,128',
                        help='Comma-separated hidden layer sizes for auxiliary NNs (regressor, classifier, policy)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    fit_data, test_data, n_fit, n_test = load_and_split(
        args.input, seed=args.seed, train_frac=args.train_frac)
    state_dim = fit_data['states'].shape[1]
    print(f"Fit: {n_fit} patients, Test: {n_test} patients (seed={args.seed})")
    print(f"State dim: {state_dim}, Actions: 25")
    print(f"Missing rate: {1 - fit_data['o_t'].mean():.3f}")

    results = {}

    q_hidden = tuple(int(x) for x in args.q_hidden.split(','))
    bridge_hidden = tuple(int(x) for x in args.bridge_hidden.split(','))
    aux_hidden = tuple(int(x) for x in args.aux_hidden.split(','))

    bridge_kw = dict(n_steps=args.bridge_steps, batch_size=args.batch_size,
                     device=args.device,
                     h_hidden=bridge_hidden, f_hidden=bridge_hidden)
    q_kw = dict(n_steps=args.q_steps, batch_size=args.batch_size,
                device=args.device, state_dim=state_dim, n_actions=25,
                hidden=q_hidden)

    # 1. Oracle
    print("\n[1/6] OracleFQE (r_true)...")
    oracle = NNOracleFQE(state_dim=state_dim, gamma=args.gamma,
                         q_kwargs=q_kw, device=args.device)
    oracle.fit(fit_data)
    v_oracle, se_oracle = oracle.value(test_data)
    results['OracleFQE'] = (v_oracle, se_oracle)
    print(f"  V = {v_oracle:.4f} +/- {se_oracle:.4f}")

    # 2. Naive
    print("\n[2/6] NaiveFQE (O=1 only)...")
    naive = NNNaiveFQE(state_dim=state_dim, gamma=args.gamma,
                       q_kwargs=q_kw, device=args.device)
    naive.fit(fit_data)
    v_naive, se_naive = naive.value(test_data)
    results['NaiveFQE'] = (v_naive, se_naive)
    print(f"  V = {v_naive:.4f} +/- {se_naive:.4f}")

    # 3. ImputeFQE
    print("\n[3/6] ImputeFQE (NN regression)...")
    impute_kw = dict(hidden=aux_hidden, n_steps=args.bridge_steps,
                     lr=1e-3, batch_size=args.batch_size, device=args.device)
    impute = NNImputeFQE(state_dim=state_dim, gamma=args.gamma,
                         q_kwargs=q_kw, impute_kwargs=impute_kw,
                         device=args.device)
    impute.fit(fit_data)
    v_impute, se_impute = impute.value(test_data)
    results['ImputeFQE'] = (v_impute, se_impute)
    print(f"  V = {v_impute:.4f} +/- {se_impute:.4f}")

    # 4. IPW-FQE
    print("\n[4/6] IPW-FQE (inverse propensity)...")
    prop_kw = dict(hidden=aux_hidden, n_steps=args.bridge_steps,
                   lr=1e-3, batch_size=args.batch_size, device=args.device)
    ipw = NNIPWFQE(state_dim=state_dim, gamma=args.gamma,
                   bridge_kwargs=bridge_kw, q_kwargs=q_kw,
                   prop_kwargs=prop_kw, device=args.device)
    ipw.fit(fit_data)
    v_ipw, se_ipw = ipw.value(test_data)
    results['IPW-FQE'] = (v_ipw, se_ipw)
    print(f"  V = {v_ipw:.4f} +/- {se_ipw:.4f}")

    # 5. SCOPE
    print("\n[5/6] SCOPE (reward shaping + IS)...")
    phi_kw = dict(hidden=aux_hidden, n_steps=args.bridge_steps,
                  lr=1e-3, batch_size=args.batch_size, device=args.device)
    beh_kw = dict(n_actions=25, hidden=q_hidden, n_steps=args.q_steps,
                  lr=1e-3, batch_size=args.batch_size, device=args.device)
    scope = NNSCOPE(state_dim=state_dim, gamma=args.gamma,
                    phi_kwargs=phi_kw, behavior_kwargs=beh_kw,
                    device=args.device)
    scope.fit(fit_data)
    v_scope, se_scope = scope.value(fit_data)
    results['SCOPE'] = (v_scope, se_scope)
    print(f"  V = {v_scope:.4f} +/- {se_scope:.4f}")

    # 6. ProxFQE (our method)
    print("\n[6/6] ProxFQE (NN bridge)...")
    prox = NNProxFQE(state_dim=state_dim, gamma=args.gamma,
                     bridge_kwargs=bridge_kw, q_kwargs=q_kw,
                     device=args.device)
    prox.fit(fit_data)
    v_prox, se_prox = prox.value(test_data)
    results['ProxFQE'] = (v_prox, se_prox)
    print(f"  V = {v_prox:.4f} +/- {se_prox:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print(f"{'Method':<15} {'V(pi)':<12} {'SE':<10} {'Bias vs Oracle'}")
    print("-" * 60)
    for name, (v, se) in results.items():
        bias = v - v_oracle if name != 'OracleFQE' else 0.0
        print(f"{name:<15} {v:<12.4f} {se:<10.4f} {bias:+.4f}")

    # Save
    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, 'ope_results.csv')
    pd.DataFrame([
        {'method': k, 'V_pi': v, 'SE': se} for k, (v, se) in results.items()
    ]).to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
