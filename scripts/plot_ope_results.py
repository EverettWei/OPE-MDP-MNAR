# scripts/plot_ope_results.py
from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

def _ensure_dir(d: str):
    if d: os.makedirs(d, exist_ok=True)

def _load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.copy()
    if "mae" not in df.columns: df["mae"] = df["bias"].abs()
    if "mae_rel" not in df.columns: df["mae_rel"] = df["mae"] / df["true"].abs().clip(lower=1e-8)
    return df


def _agg_with_se(df: pd.DataFrame, by: list[str], metric: str) -> pd.DataFrame:
    g = df.groupby(by, as_index=False).agg(
        mean=(metric, "mean"),
        sd=(metric, "std"),
        n_rep=(metric, "size"),
        true_mean=("TRUE", "mean") if "TRUE" in df.columns else (metric, "size"),
        missing=("missing", "mean") if "missing" in df.columns else (metric, "size"),
    )
    g["se"] = g["sd"].fillna(0.0) / g["n_rep"].clip(lower=1).pow(0.5)
    return g


def _nice_style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "font.size": 13,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2.2,
        "lines.markersize": 6.5,
    })

COLOR_MAP = {
    "naive":  "#d95f02",
    "prox":   "#1b9e77",
    "ipw":    "#7570b3",
    "impute": "#e7298a",
    "scope":  "#66a61e",
}

C0_TO_MISS_PCT = {
    0.3: "~20%",
    -0.7: "~40%",
    -1.5: "~60%",
    -2.8: "~80%",
}

CROSS_C0S = [0.3, -0.7, -2.8]  # ~20%, ~40%, ~80%

def _draw_manual_errbars(ax, xs, lo, hi, color, cap_frac=0.03, lw=1.6, alpha=0.9):
    ax.vlines(xs, lo, hi, colors=color, linewidth=lw, alpha=alpha)
    cap = np.maximum(xs * cap_frac, 1e-12)
    ax.hlines(lo, xs - cap, xs + cap, colors=color, linewidth=lw, alpha=alpha)
    ax.hlines(hi, xs - cap, xs + cap, colors=color, linewidth=lw, alpha=alpha)


def _plot_line_ci(ax, df_raw: pd.DataFrame, x: str, metric: str,
                  ylabel: str, title: str, logy: bool=False, logx: bool=False,
                  show_se: bool=True, se_mult: float=1.0):
    methods = list(dict.fromkeys(df_raw["method"].tolist()))
    eps = 1e-12

    for m in methods:
        d = df_raw[df_raw["method"] == m]
        if d.empty:
            continue

        g = _agg_with_se(d, by=[x], metric=metric).sort_values(by=x)
        xs = g[x].values
        color = COLOR_MAP.get(m, None)

        mean = g["mean"].values
        se = g["se"].values * float(se_mult)

        mean_plot = np.maximum(mean, eps) if logy else mean
        ax.plot(xs, mean_plot, marker="o", lw=2.0, label=m, color=color)

        if show_se:
            lo = mean - se
            hi = mean + se
            if logy:
                ok = lo > eps
                if np.any(ok):
                    _draw_manual_errbars(ax, xs[ok], lo[ok], hi[ok], color=color)
            else:
                _draw_manual_errbars(ax, xs, lo, hi, color=color)

    ax.set_xlabel(x)
    ax.set_ylabel(ylabel + (" (log2 scale)" if logy else ""))
    ax.set_title(title)
    if logy:
        ax.set_yscale("log", base=2)
    if logx:
        ax.set_xscale("log", base=2)
    ax.legend(frameon=True, loc="best", title="method")


# ============================================================
# Cross-product plots: 1x3, one subplot per missingness %
# ============================================================

def plot_size_x_missrate(tables: str, outdir: str):
    """1x3 panel: MSE vs n, one subplot per missingness percentage."""
    _nice_style()
    _ensure_dir(outdir)

    p = os.path.join(tables, "ope_runs_size_x_missrate.csv")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"{p} not found. Run eval_grid --mode size_x_missrate first.")

    df = _load_raw(p)
    if "mnar_c0" not in df.columns:
        raise KeyError("mnar_c0 column not found in CSV.")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.0))

    for idx, c0 in enumerate(CROSS_C0S):
        ax = axes[idx]
        sub = df[df["mnar_c0"] == c0]
        miss_label = C0_TO_MISS_PCT.get(c0, f"c0={c0}")
        _plot_line_ci(ax, sub, x="n", metric="mse",
                      ylabel="MSE", title=f"MSE vs n  (missingness {miss_label})",
                      logy=True, logx=True)

    for ax in axes.flat:
        leg = ax.get_legend()
        if leg is not None: leg.remove()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=len(labels), frameon=True, title="method",
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.08, 1, 1.0])
    out_path = os.path.join(outdir, "mse_vs_n_by_missingness.pdf")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.4)
    plt.close()
    print(f"[plot] size_x_missrate saved to {out_path}")


def plot_horizon_x_missrate(tables: str, outdir: str):
    """1x3 panel: MSE vs T, one subplot per missingness percentage."""
    _nice_style()
    _ensure_dir(outdir)

    p = os.path.join(tables, "ope_runs_horizon_x_missrate.csv")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"{p} not found. Run eval_grid --mode horizon_x_missrate first.")

    df = _load_raw(p)
    if "mnar_c0" not in df.columns:
        raise KeyError("mnar_c0 column not found in CSV.")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.0))

    for idx, c0 in enumerate(CROSS_C0S):
        ax = axes[idx]
        sub = df[df["mnar_c0"] == c0]
        miss_label = C0_TO_MISS_PCT.get(c0, f"c0={c0}")
        _plot_line_ci(ax, sub, x="T", metric="mse",
                      ylabel="MSE", title=f"MSE vs T  (missingness {miss_label})",
                      logy=True, logx=True)

    for ax in axes.flat:
        leg = ax.get_legend()
        if leg is not None: leg.remove()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=len(labels), frameon=True, title="method",
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.08, 1, 1.0])
    out_path = os.path.join(outdir, "mse_vs_T_by_missingness.pdf")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.4)
    plt.close()
    print(f"[plot] horizon_x_missrate saved to {out_path}")


def plot_reward_x_missrate(tables: str, outdir: str):
    """1x3 panel: grouped bars of MSE by reward type, one subplot per missingness percentage."""
    _nice_style()
    _ensure_dir(outdir)

    p = os.path.join(tables, "ope_runs_reward_x_missrate.csv")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"{p} not found. Run eval_grid --mode reward_x_missrate first.")

    df = _load_raw(p)
    if "mnar_c0" not in df.columns or "reward_type" not in df.columns:
        raise KeyError("mnar_c0 and reward_type columns required in CSV.")

    rtypes = list(dict.fromkeys(df["reward_type"].tolist()))
    methods = list(dict.fromkeys(df["method"].tolist()))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.0))

    for idx, c0 in enumerate(CROSS_C0S):
        ax = axes[idx]
        sub = df[df["mnar_c0"] == c0]
        miss_label = C0_TO_MISS_PCT.get(c0, f"c0={c0}")

        x_pos = np.arange(len(rtypes))
        width = 0.8 / max(len(methods), 1)

        for i, m in enumerate(methods):
            means, ses = [], []
            for rt in rtypes:
                s = sub[(sub["method"] == m) & (sub["reward_type"] == rt)]
                means.append(s["mse"].mean() if len(s) else 0)
                se = s["mse"].std() / max(len(s), 1) ** 0.5 if len(s) > 1 else 0
                ses.append(se)

            offset = (i - len(methods) / 2 + 0.5) * width
            color = COLOR_MAP.get(m, None)
            ax.bar(x_pos + offset, means, width, yerr=ses,
                   label=m, color=color, capsize=3, alpha=0.85)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(rtypes)
        ax.set_xlabel("Reward Type")
        ax.set_ylabel("MSE")
        ax.set_title(f"MSE by Reward Type  (missingness {miss_label})")
        ax.set_yscale("log", base=2)

    for ax in axes.flat:
        leg = ax.get_legend()
        if leg is not None: leg.remove()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=len(labels), frameon=True, title="method",
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.08, 1, 1.0])
    out_path = os.path.join(outdir, "mse_reward_by_missingness.pdf")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.4)
    plt.close()
    print(f"[plot] reward_x_missrate saved to {out_path}")


# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser("Plot OPE simulation results.")
    ap.add_argument("--tables", type=str, default="results/tables",
                    help="directory containing ope_runs_*.csv")
    ap.add_argument("--which", type=str, default="all",
                choices=["size_x_missrate", "horizon_x_missrate", "reward_x_missrate", "all"],
                help="which figure(s) to make")
    ap.add_argument("--outdir", type=str, default="results/figures",
                    help="where to save figures")
    ap.add_argument("--gui", action="store_true", help="interactive backend")
    args = ap.parse_args()
    if not args.gui:
        matplotlib.use("Agg")

    _ensure_dir(args.outdir)

    if args.which == "all":
        plot_size_x_missrate(args.tables, args.outdir)
        plot_horizon_x_missrate(args.tables, args.outdir)
        plot_reward_x_missrate(args.tables, args.outdir)
    elif args.which == "size_x_missrate":
        plot_size_x_missrate(args.tables, args.outdir)
    elif args.which == "horizon_x_missrate":
        plot_horizon_x_missrate(args.tables, args.outdir)
    elif args.which == "reward_x_missrate":
        plot_reward_x_missrate(args.tables, args.outdir)

    print(f"[plot] figures saved to {args.outdir}")

if __name__ == "__main__":
    main()
