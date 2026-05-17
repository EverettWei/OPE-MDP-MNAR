"""
Clean the raw sepsis CSV following Raghu et al. (NeurIPS 2017):
  1. Keep trajectories with >= 11 steps, truncate to first 11
  2. Compute reward_t = -(SOFA_{t+1} - SOFA_t) for t=1..10
  3. Drop step 11 (no S_12 so no R_11) -> T=10
  4. Keep exactly the paper's 48 state features
  5. Actions: vaso_input (0-4), iv_input (0-4) -> 25 joint actions

State features (48):
  Demographics/Static (8): Shock_Index, elixhauser, SIRS, gender, re_admission, GCS, SOFA, age
  Lab Values (24): Albumin, Arterial_pH, Calcium, Glucose, Hb, Magnesium, PTT, Potassium,
      SGPT, Arterial_BE, BUN, Chloride, HCO3, INR, Sodium, Arterial_lactate, CO2_mEqL,
      Creatinine, Ionised_Ca, PT, Platelets_count, SGOT, Total_bili, WBC_count
  Vital Signs (12): DiaBP, SysBP, MeanBP, paCO2, paO2, FiO2_1, PaO2_FiO2, RR, Temp_C,
      Weight_kg, HR, SpO2
  Intake/Output (3): output_4hourly, output_total, mechvent
  Misc (1): bloc (timestep)
"""

import argparse
import os
import pandas as pd


# Columns NOT in the paper's 48 features (to drop along with metadata/outcomes)
DROP_COLS = [
    # metadata / time
    'charttime',
    'presumed_onset',
    # outcome columns
    'died_in_hosp',
    'died_within_48h_of_out_time',
    'mortality_90d',
    'delay_end_of_record_and_discharge_or_death',
    # not in paper's 48 features
    'input_total',
    'input_4hourly',
    'cumulated_balance',
    'median_dose_vaso',
    'max_dose_vaso',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,
                        default='realdata/sepsis_processed_state_action.csv')
    parser.add_argument('--outdir', type=str, default='realdata')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    n_before = df['icustayid'].nunique()

    # Step 1: Keep trajectories with >= 11 steps, truncate to first 11
    traj_len = df.groupby('icustayid')['bloc'].count()
    keep_ids = traj_len[traj_len >= 11].index
    df = df[df['icustayid'].isin(keep_ids)].copy()
    df = df.sort_values(['icustayid', 'bloc'])
    df = df.groupby('icustayid').head(11).reset_index(drop=True)
    n_after = df['icustayid'].nunique()
    print(f"Patients: {n_before} -> {n_after} (dropped {n_before - n_after} with < 11 steps)")

    # Step 2: Compute reward_t = -(SOFA_{t+1} - SOFA_t)
    df['reward'] = -(df.groupby('icustayid')['SOFA'].shift(-1) - df['SOFA'])

    # Step 3: Drop step 11 (reward is NaN there)
    df = df[df['reward'].notna()].copy()
    df['reward'] = df['reward'].astype(float)

    assert df.groupby('icustayid')['bloc'].count().unique().tolist() == [10]
    print(f"After dropping step 11: {len(df)} rows ({n_after} x 10)")

    # Step 4: Drop non-paper columns
    df = df.drop(columns=DROP_COLS)

    # Verify 48 state features
    non_state = ['icustayid', 'vaso_input', 'iv_input', 'reward']
    state_cols = [c for c in df.columns if c not in non_state]
    print(f"State features: {len(state_cols)}")
    assert len(state_cols) == 48, f"Expected 48 state features, got {len(state_cols)}: {state_cols}"
    print(f"Columns ({df.shape[1]}): {list(df.columns)}")

    # Save
    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, 'sepsis_T10.csv')
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


if __name__ == '__main__':
    main()
