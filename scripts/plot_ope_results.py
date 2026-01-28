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
    # ensure required metrics
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
    "naive": "#d95f02",
    "prox":  "#1b9e77",
    "ipw":   "#7570b3",
}

def _draw_manual_errbars(ax, xs, lo, hi, color, cap_frac=0.03, lw=1.6, alpha=0.9):
    # vertical segments
    ax.vlines(xs, lo, hi, colors=color, linewidth=lw, alpha=alpha)

    # caps: width proportional to x (works for log-scaled x)
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

        # mean line
        mean_plot = np.maximum(mean, eps) if logy else mean
        ax.plot(xs, mean_plot, marker="o", lw=2.0, label=m, color=color)

        # manual +/- se bars
        if show_se:
            lo = mean - se
            hi = mean + se

            if logy:
                # only draw bars where lower bound is valid on log scale
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








# ---------------- plot MSE ----------------
def plot_size_mse(df_size: pd.DataFrame, outdir: str):
    _nice_style()
    _ensure_dir(outdir)

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    _plot_line_ci(
        ax, df_size, x="n", metric="mse",
        ylabel="MSE", title="MSE vs n",
        logy=True, logx=True
    )
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "mse_vs_n_size.png"))
    plt.close()


def plot_horizon_mse(df_h: pd.DataFrame, outdir: str):
    _nice_style()
    _ensure_dir(outdir)

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    _plot_line_ci(
        ax, df_h, x="T", metric="mse",
        ylabel="MSE", title="MSE vs T",
        logy=True, logx=True
    )
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "mse_vs_T_horizon.png"))
    plt.close()



# ---------------- MSE panel (1x2) ----------------

def plot_panel(tables: str, outdir: str):
    _nice_style()
    _ensure_dir(outdir)

    p_size = os.path.join(tables, "ope_runs_size.csv")
    p_hor  = os.path.join(tables, "ope_runs_horizon.csv")
    if not os.path.isfile(p_size):
        raise FileNotFoundError(f"{p_size} not found. Run eval_grid --mode size first.")
    if not os.path.isfile(p_hor):
        raise FileNotFoundError(f"{p_hor} not found. Run eval_grid --mode horizon first.")

    df_size = _load_raw(p_size)
    df_h    = _load_raw(p_hor)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ax0, ax1 = axes

    # left: MSE vs n
    _plot_line_ci(ax0, df_size, x="n", metric="mse",
                  ylabel="MSE", title="MSE vs n", logy=True, logx=True)

    # right: MSE vs T
    _plot_line_ci(ax1, df_h, x="T", metric="mse",
                  ylabel="MSE", title="MSE vs T", logy=True, logx=True)

    # remove per-axis legends, use one common legend
    leg0 = ax0.get_legend()
    if leg0 is not None:
        leg0.remove()
    leg1 = ax1.get_legend()
    if leg1 is not None:
        leg1.remove()

    handles, labels = ax0.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               ncol=max(3, len(labels)), frameon=True, title="method")

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    out_path = os.path.join(outdir, "panel_ope_mse.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[plot] panel saved to {out_path}")


# ---------------- MAE panel (1x2) ----------------

def plot_panel_mae(tables: str, outdir: str):
    _nice_style()
    _ensure_dir(outdir)

    p_size = os.path.join(tables, "ope_runs_size.csv")
    p_hor  = os.path.join(tables, "ope_runs_horizon.csv")
    if not os.path.isfile(p_size):
        raise FileNotFoundError(f"{p_size} not found. Run eval_grid --mode size first.")
    if not os.path.isfile(p_hor):
        raise FileNotFoundError(f"{p_hor} not found. Run eval_grid --mode horizon first.")

    df_size = _load_raw(p_size)
    df_h    = _load_raw(p_hor)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ax0, ax1 = axes

    _plot_line_ci(ax0, df_size, x="n", metric="mae",
              ylabel="MAE", title="MAE vs n", logy=True, logx=True)


    _plot_line_ci(ax1, df_h, x="T", metric="mae",
                  ylabel="MAE", title="MAE vs T", logy=True, logx=True)

    leg0 = ax0.get_legend()
    if leg0 is not None:
        leg0.remove()
    leg1 = ax1.get_legend()
    if leg1 is not None:
        leg1.remove()

    handles, labels = ax0.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               ncol=max(3, len(labels)), frameon=True, title="method")

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    out_path = os.path.join(outdir, "panel_ope_mae.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[plot] MAE panel saved to {out_path}")


# ---------------- MSE+MAE panel (2x2) ----------------

def plot_grid_2x2(tables: str, outdir: str):
    _nice_style()
    _ensure_dir(outdir)

    p_size = os.path.join(tables, "ope_runs_size.csv")
    p_hor  = os.path.join(tables, "ope_runs_horizon.csv")
    if not os.path.isfile(p_size):
        raise FileNotFoundError(f"{p_size} not found. Run eval_grid --mode size first.")
    if not os.path.isfile(p_hor):
        raise FileNotFoundError(f"{p_hor} not found. Run eval_grid --mode horizon first.")

    df_size = _load_raw(p_size)
    df_h    = _load_raw(p_hor)

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.0))
    ax00, ax01 = axes[0, 0], axes[0, 1]
    ax10, ax11 = axes[1, 0], axes[1, 1]

    _plot_line_ci(ax00, df_size, x="n", metric="mse",
                  ylabel="MSE", title="MSE vs n", logy=True, logx=True)
    _plot_line_ci(ax01, df_h, x="T", metric="mse",
                  ylabel="MSE", title="MSE vs T", logy=True, logx=True)

    _plot_line_ci(ax10, df_size, x="n", metric="mae",
                  ylabel="MAE", title="MAE vs n", logy=True, logx=True)
    _plot_line_ci(ax11, df_h, x="T", metric="mae",
                  ylabel="MAE", title="MAE vs T", logy=True, logx=True)

    for ax in [ax00, ax01, ax10, ax11]:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    handles, labels = ax00.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               ncol=max(3, len(labels)), frameon=True, title="method")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = os.path.join(outdir, "grid_ope_mse_mae_2x2.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[plot] 2x2 grid saved to {out_path}")


# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser("Plot OPE results (size/horizon/panel).")
    ap.add_argument("--tables", type=str, default="results/tables",
                    help="directory containing ope_runs_{size,horizon}.csv")
    ap.add_argument("--which", type=str, default="size",
                choices=["size", "horizon", "panel", "panel_mae", "grid2x2"],
                help="which figure(s) to make")

    ap.add_argument("--outdir", type=str, default="results/figures",
                    help="where to save figures")
    ap.add_argument("--gui", action="store_true", help="interactive backend")
    args = ap.parse_args()
    if not args.gui:
        matplotlib.use("Agg")

    _ensure_dir(args.outdir)

    if args.which == "size":
        df = _load_raw(os.path.join(args.tables, "ope_runs_size.csv"))
        plot_size_mse(df, args.outdir)
    elif args.which == "horizon":
        df = _load_raw(os.path.join(args.tables, "ope_runs_horizon.csv"))
        plot_horizon_mse(df, args.outdir)
    elif args.which == "panel_mae":
        plot_panel_mae(args.tables, args.outdir)
    elif args.which == "grid2x2":
        plot_grid_2x2(args.tables, args.outdir)
    else:
        plot_panel(args.tables, args.outdir)

    print(f"[plot] figures saved to {args.outdir}")

if __name__ == "__main__":
    main()