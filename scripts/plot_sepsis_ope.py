"""Plot sepsis OPE results across missing rates."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULT_DIR = 'realdata/results'
OUT_DIR = 'realdata/results/figures'

# Match simulation color scheme from plot_ope_results.py
METHODS = ['OracleFQE', 'ProxFQE', 'IPW-FQE', 'NaiveFQE', 'ImputeFQE']
COLORS = {
    'ProxFQE':   '#1b9e77',   # green (ours)
    'NaiveFQE':  '#d95f02',   # orange
    'IPW-FQE':   '#7570b3',   # purple
    'ImputeFQE': '#e7298a',   # pink
    'OracleFQE': '#666666',   # gray (reference)
}
MARKERS = {
    'ProxFQE':   'D',
    'NaiveFQE':  'o',
    'IPW-FQE':   '^',
    'ImputeFQE': 'v',
    'OracleFQE': 's',
}
LABELS = {
    'ProxFQE':   'prox',
    'NaiveFQE':  'naive',
    'IPW-FQE':   'ipw',
    'ImputeFQE': 'impute',
    'OracleFQE': 'oracle',
}
MISS_RATES = [20, 40, 60, 80]


def _nice_style():
    plt.rcParams.update({
        'figure.dpi': 150, 'savefig.dpi': 250,
        'font.size': 16,
        'axes.labelsize': 17,
        'axes.titlesize': 18,
        'legend.fontsize': 14,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linestyle': '--',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'lines.linewidth': 2.5,
        'lines.markersize': 9,
    })


def load_results():
    rows = []
    for mr in MISS_RATES:
        path = os.path.join(RESULT_DIR, f'mnar{mr}', 'ope_results.csv')
        df = pd.read_csv(path)
        df['miss_rate'] = mr / 100.0
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def plot_panel(df):
    """Side-by-side: (a) Absolute Bias vs Oracle, (b) Grouped bar V(pi)."""
    _nice_style()
    os.makedirs(OUT_DIR, exist_ok=True)

    # Exclude SCOPE (extreme values)
    methods_bias = [m for m in METHODS if m not in ('OracleFQE',)]
    methods_bar = [m for m in METHODS]

    # Compute bias
    oracle = df[df['method'] == 'OracleFQE'][['miss_rate', 'V_pi']].rename(
        columns={'V_pi': 'V_oracle'})
    merged = df.merge(oracle, on='miss_rate')
    merged['bias'] = (merged['V_pi'] - merged['V_oracle']).abs()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- (a) Grouped bar chart ---
    n_methods = len(methods_bar)
    n_groups = len(MISS_RATES)
    width = 0.14
    x = np.arange(n_groups)

    for i, method in enumerate(methods_bar):
        sub = df[df['method'] == method].sort_values('miss_rate')
        offset = (i - n_methods / 2 + 0.5) * width
        ax1.bar(x + offset, sub['V_pi'].values, width,
                yerr=sub['SE'].values, capsize=4,
                error_kw=dict(lw=1.5, capthick=1.5, color='black'),
                label=LABELS[method], color=COLORS[method], alpha=0.88,
                edgecolor='white', linewidth=0.5)

    ax1.set_xlabel('MNAR Missing Rate')
    ax1.set_ylabel(r'$\hat{V}(\pi)$')
    ax1.set_title(r'(a) Estimated Policy Value $\hat{V}(\pi)$')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['20%', '40%', '60%', '80%'])
    ax1.axhline(y=0, color='gray', linewidth=0.5)

    # --- (b) Absolute Bias ---
    for method in methods_bias:
        sub = merged[merged['method'] == method].sort_values('miss_rate')
        ax2.plot(sub['miss_rate'], sub['bias'],
                 label=LABELS[method], color=COLORS[method],
                 marker=MARKERS[method], markersize=10)

    ax2.set_xlabel('MNAR Missing Rate')
    ax2.set_ylabel(r'$|\hat{V}(\pi) - V_{\mathrm{Oracle}}|$')
    ax2.set_title('(b) Absolute Bias vs Oracle')
    ax2.set_xticks([0.2, 0.4, 0.6, 0.8])
    ax2.set_xticklabels(['20%', '40%', '60%', '80%'])

    # Shared legend at top
    h1, l1 = ax1.get_legend_handles_labels()
    all_handles = list(h1)
    all_labels = list(l1)

    fig.subplots_adjust(left=0.06, right=0.97, top=0.92, bottom=0.18, wspace=0.25)

    fig.legend(all_handles, all_labels, loc='lower center',
               ncol=5, frameon=True, fontsize=14,
               bbox_to_anchor=(0.5, 0.01))

    out_path = os.path.join(OUT_DIR, 'sepsis_ope_panel.pdf')
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_bias_only(df):
    """Single plot: Absolute Bias vs Oracle."""
    _nice_style()
    os.makedirs(OUT_DIR, exist_ok=True)

    methods_bias = [m for m in METHODS if m not in ('OracleFQE',)]

    oracle = df[df['method'] == 'OracleFQE'][['miss_rate', 'V_pi']].rename(
        columns={'V_pi': 'V_oracle'})
    merged = df.merge(oracle, on='miss_rate')
    merged['bias'] = (merged['V_pi'] - merged['V_oracle']).abs()

    fig, ax = plt.subplots(figsize=(8, 6))

    for method in methods_bias:
        sub = merged[merged['method'] == method].sort_values('miss_rate')
        ax.plot(sub['miss_rate'], sub['bias'],
                label=LABELS[method], color=COLORS[method],
                marker=MARKERS[method], markersize=10)

    ax.set_xlabel('MNAR Missing Rate')
    ax.set_ylabel(r'$|\hat{V}(\pi) - V_{\mathrm{Oracle}}|$')
    ax.set_title('Absolute Bias vs Oracle')
    ax.set_xticks([0.2, 0.4, 0.6, 0.8])
    ax.set_xticklabels(['20%', '40%', '60%', '80%'])
    ax.legend(fontsize=14, frameon=True)

    fig.tight_layout()

    out_path = os.path.join(OUT_DIR, 'sepsis_ope_bias.pdf')
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == '__main__':
    df = load_results()
    print(df[df['method'] != 'SCOPE'].to_string(index=False))
    print()
    plot_panel(df)
    plot_bias_only(df)
