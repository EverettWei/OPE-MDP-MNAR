'''
Minimal visualization for MNAR offline data.

Outputs (under results/figures/ or --outdir)
--------------------------------------------
- hist_counts_overall_observed_missing.png  : filled count hist of r_true for overall, observed (o==1), missing (o==0)
- scatter_s_by_action.png                   : (s1, s2) colored by action a ∈ {-1, +1}
- scatter_s_by_missing.png                  : (s1, s2) colored by missingness o ∈ {0, 1}
- missing_rate.txt                          : overall missing rate

Notes
-----
- Input can be .npz (canonical) or .csv from src/generate_data.py.
'''

from __future__ import annotations

import argparse
import os
from typing import Dict

import numpy as np
import matplotlib

try:
    import pandas as pd  # optional for CSV input
except Exception:
    pd = None


# ---------------------------- I/O helpers ----------------------------

def load_data(path: str) -> Dict[str, np.ndarray]:
    '''
    Load dataset produced by src/generate_data.py into a dict of ndarrays.
    Required keys: obs, a, o, r_obs, r_true, obs_n, ep, t
    '''
    p = path.lower()
    if p.endswith('.npz'):
        with np.load(path) as z:
            data = {k: z[k] for k in z.files}
    elif p.endswith('.csv'):
        if pd is None:
            raise RuntimeError('CSV input requires pandas. Install pandas or use a .npz file.')
        df = pd.read_csv(path)
        data = {
            'obs':    df[['s1','s2','o_prev']].to_numpy(np.float32),
            'a':      df['a'].to_numpy(np.int8),
            'o':      df['o_t'].to_numpy(np.int8),
            'r_obs':  df['r_obs'].to_numpy(np.float32),
            'r_true': df['r_true'].to_numpy(np.float32),
            'obs_n':  df[['s1_next','s2_next','o_t_next']].to_numpy(np.float32),
            'ep':     df['ep'].to_numpy(np.int32),
            't':      df['t'].to_numpy(np.int16),
        }
    else:
        raise ValueError(f'Unsupported input format: {path}')

    for k in ('obs','a','o','r_obs','r_true','obs_n','ep','t'):
        if k not in data:
            raise KeyError(f'Missing field: {k}')
    return data


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


# ---------------------------- plots ----------------------------

def plot_reward_counts_triple(ds: Dict[str, np.ndarray], out: str, bins: int = 50) -> None:
    """
    One figure with THREE overlaid FILLED count histograms:
      - r_true (overall)
      - r_true | o==1  (observed)
      - r_true | o==0  (missing)

    Additionally, overlay a line for the missing rate vs reward (right y-axis).

    Shared bin edges from the overall distribution to make counts comparable.
    """
    rt = ds["r_true"]
    o = ds["o"].astype(bool)
    mask = np.isfinite(rt)
    rt = rt[mask]
    o = o[mask]

    n_all = rt.size
    n_obs = int(o.sum())
    n_mis = int((~o).sum())
    if n_all == 0:
        raise ValueError("No finite r_true found.")

    # shared bins from robust range
    q_lo, q_hi = np.percentile(rt, [0.5, 99.5])
    if (not np.isfinite(q_lo)) or (not np.isfinite(q_hi)) or q_hi <= q_lo:
        q_lo, q_hi = float(rt.min()), float(rt.max())
        if q_hi <= q_lo:
            q_lo, q_hi = q_lo - 1.0, q_lo + 1.0
    edges = np.linspace(q_lo, q_hi, bins + 1)
    bin_width = edges[1] - edges[0]
    centers = 0.5 * (edges[:-1] + edges[1:])

    # histogram counts
    counts_all, _ = np.histogram(rt, bins=edges)
    counts_obs, _ = np.histogram(rt[o], bins=edges)
    counts_mis, _ = np.histogram(rt[~o], bins=edges)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(7.6, 4.8))
    ax = plt.gca()

    # filled bars with transparency
    ax.bar(centers, counts_all, width=bin_width, alpha=0.45, color="C0",
           edgecolor="#1a1a1a", linewidth=0.6, label=f"overall (n={n_all})")
    if n_obs > 0:
        ax.bar(centers, counts_obs, width=bin_width, alpha=0.45, color="C1",
               edgecolor="#1a1a1a", linewidth=0.6, label=f"observed o=1 (n={n_obs})")
    if n_mis > 0:
        ax.bar(centers, counts_mis, width=bin_width, alpha=0.45, color="C3",
               edgecolor="#1a1a1a", linewidth=0.6, label=f"missing  o=0 (n={n_mis})")

    ax.set_xlabel("reward (r_true)")
    ax.set_ylabel("count")
    ax.set_title("Counts: overall vs observed vs missing")
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    # missing rate per bin (right y-axis)
    counts_all_safe = np.maximum(counts_all, 1)  # avoid divide-by-zero
    miss_rate = counts_mis / counts_all_safe

    ax2 = ax.twinx()
    ax2.plot(centers, miss_rate, color="black", marker="o", linewidth=1.5,
             label="missing rate")
    ax2.set_ylabel("missing rate")

    # combine legends
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper right")

    _ensure_dir(out)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()



def scatter_s_colored(
    ds: Dict[str, np.ndarray],
    by: str,
    out: str,
    title: str,
    *,
    max_points: int = 20000,
) -> None:
    '''
    Scatter of (s1, s2) colored by 'a' (action) or 'o' (missingness).
    '''
    s = ds['obs'][:, :2].astype(np.float32)
    if by == 'a':
        c = ds['a'].astype(int); cmap = 'coolwarm'; ticks = [-1, 1]
    elif by == 'o':
        c = ds['o'].astype(int); cmap = 'viridis'; ticks = [0, 1]
    else:
        raise ValueError("`by` must be 'a' or 'o'.")

    n = s.shape[0]
    if n > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=max_points, replace=False)
        s = s[idx]; c = c[idx]

    import matplotlib.pyplot as plt
    plt.figure(figsize=(5.8, 4.8))
    sc = plt.scatter(s[:,0], s[:,1], c=c, cmap=cmap, s=10,
                     alpha=0.65, edgecolors='none')
    plt.xlabel('s1'); plt.ylabel('s2'); plt.title(title)
    cb = plt.colorbar(sc, ticks=ticks); cb.set_label(by)
    plt.grid(alpha=0.25, linestyle='--')

    _ensure_dir(out)
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()


def write_missing_rate(ds: dict[str, np.ndarray], csv_path: str) -> None:
    '''
    Write overall missing rate to a CSV file.

    Output columns:
      n_all, n_observed, n_missing, observed_rate, missing_rate
    '''
    o = ds['o'].astype(int)
    n_all = int(o.size)
    n_obs = int(o.sum())
    n_mis = n_all - n_obs
    obs_rate = (n_obs / n_all) if n_all > 0 else 0.0
    miss_rate = 1.0 - obs_rate

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w') as f:
        f.write('n_all,n_observed,n_missing,observed_rate,missing_rate\n')
        f.write(f'{n_all},{n_obs},{n_mis},{obs_rate:.6f},{miss_rate:.6f}\n')


def stitch_panel_horizontal(outdir: str,
                            panel_name: str = "data_overview_panel.png",
                            titles = ("Reward distribution","(s1,s2) by action","(s1,s2) by missing"),
                            paths: list[str] | None = None) -> None:
    import os
    import matplotlib.pyplot as plt
    if paths is None:
        files = ["hist_counts_overall_observed_missing.png",
                 "scatter_s_by_action.png",
                 "scatter_s_by_missing.png"]
        paths = [os.path.join(outdir, f) for f in files]
    for p in paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(p)

    imgs = [plt.imread(p) for p in paths]
    plt.rcParams.update({"figure.dpi":120,"savefig.dpi":300,"font.size":12})
    fig, axes = plt.subplots(1, 3, figsize=(14,4.2), constrained_layout=True)
    for ax, img, ttl in zip(axes, imgs, titles):
        ax.imshow(img); ax.set_title(ttl); ax.axis("off")
    out_path = os.path.join(outdir, panel_name)
    fig.savefig(out_path, dpi=300); plt.close(fig)
    print(f"[viz] stitched panel saved to {out_path}")

# ---------------------------- CLI ----------------------------

def parse_args():
    ap = argparse.ArgumentParser(description='Plot reward counts and state scatter diagnostics.')
    ap.add_argument('--in', dest='inp', required=True, help='Input .npz or .csv file')
    ap.add_argument('--outdir', default='results/figures', help='Output directory')
    ap.add_argument('--tablesdir', default='results/tables', help='Directory to write CSV tables.')
    ap.add_argument('--bins', type=int, default=50, help='Number of bins for histograms')
    ap.add_argument('--max-points', type=int, default=20000, help='Max points for scatter subsampling')
    ap.add_argument('--gui', action='store_true', help='Use interactive backend instead of Agg')
    ap.add_argument('--panel-name', default='data_overview_panel.png',
                help='Output filename for the stitched panel (saved to --outdir)')
    
    return ap.parse_args()


def main():
    args = parse_args()
    if not args.gui:
        matplotlib.use('Agg')  # headless (no windows)

    ds = load_data(args.inp)
    outdir = args.outdir

    # 1) reward counts (overall vs observed vs missing)
    plot_reward_counts_triple(ds, os.path.join(outdir, 'hist_counts_overall_observed_missing.png'), bins=args.bins)

    # 2) state scatters
    scatter_s_colored(ds, 'a', os.path.join(outdir, 'scatter_s_by_action.png'),
                      '(s1,s2) colored by action', max_points=args.max_points)
    scatter_s_colored(ds, 'o', os.path.join(outdir, 'scatter_s_by_missing.png'),
                      '(s1,s2) colored by missingness', max_points=args.max_points)

    # 3) missing rate
    tablesdir = 'results/tables'
    os.makedirs(tablesdir, exist_ok=True)
    write_missing_rate(ds, os.path.join(tablesdir, 'missing_rate.csv'))
    print(f'[viz] saved figures to {outdir}  |  tables to {tablesdir}')
    stitch_panel_horizontal(args.outdir, panel_name=args.panel_name)

if __name__ == '__main__':
    main()