import argparse
import glob
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def collect_csv_paths(folder: str, pattern: str) -> List[str]:
    search_pattern = os.path.join(folder, pattern)
    paths = sorted(glob.glob(search_pattern))
    if not paths:
        raise FileNotFoundError(f"No CSV files found under: {search_pattern}")
    return paths


def objective_trajectory(df: pd.DataFrame) -> np.ndarray:
    if "best_feasible_objective" in df.columns:
        traj = pd.to_numeric(df["best_feasible_objective"], errors="coerce").to_numpy(dtype=float)
    else:
        obj = pd.to_numeric(df["objective_min"], errors="coerce")
        feasible = df.get("feasible", pd.Series([True] * len(df))).astype(bool)
        obj = obj.where(feasible, np.nan)
        traj = obj.ffill().to_numpy(dtype=float)
    return traj


def build_trajectory_matrix(paths: List[str]) -> np.ndarray:
    trajectories = []
    max_len = 0
    for path in paths:
        df = pd.read_csv(path)
        traj = objective_trajectory(df)
        trajectories.append(traj)
        max_len = max(max_len, len(traj))

    matrix = np.full((len(trajectories), max_len), np.nan, dtype=float)
    for idx, traj in enumerate(trajectories):
        matrix[idx, : len(traj)] = traj
    return matrix


def summarize_matrix(matrix: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "mean": np.nanmean(matrix, axis=0),
        "median": np.nanmedian(matrix, axis=0),
        "p05": np.nanpercentile(matrix, 5, axis=0),
        "p95": np.nanpercentile(matrix, 95, axis=0),
    }


def counts_per_bin(df: pd.DataFrame, bin_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "source" not in df.columns:
        raise ValueError("CSV is missing required 'source' column.")

    n = len(df)
    bin_starts = np.arange(1, n + 1, bin_size)
    bin_ids = np.arange(len(bin_starts), dtype=int)

    predicted_counts = np.zeros(len(bin_starts), dtype=float)
    simulated_counts = np.zeros(len(bin_starts), dtype=float)

    src = df["source"].astype(str).str.lower().to_numpy()
    for i, start in enumerate(bin_starts):
        end = min(start + bin_size - 1, n)
        chunk = src[start - 1 : end]
        predicted_counts[i] = np.sum(chunk == "predicted")
        simulated_counts[i] = np.sum(chunk != "predicted")

    return bin_ids, predicted_counts, simulated_counts


def aggregate_bin_counts(paths: List[str], bin_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_pred = []
    all_sim = []
    max_bins = 0

    for path in paths:
        df = pd.read_csv(path)
        _, pred, sim = counts_per_bin(df, bin_size)
        all_pred.append(pred)
        all_sim.append(sim)
        max_bins = max(max_bins, len(pred))

    pred_matrix = np.full((len(paths), max_bins), np.nan, dtype=float)
    sim_matrix = np.full((len(paths), max_bins), np.nan, dtype=float)
    for i, (pred, sim) in enumerate(zip(all_pred, all_sim)):
        pred_matrix[i, : len(pred)] = pred
        sim_matrix[i, : len(sim)] = sim

    return np.arange(max_bins), pred_matrix, sim_matrix


def metric_per_bin(df: pd.DataFrame, col_name: str, bin_size: int) -> np.ndarray:
    if col_name not in df.columns:
        raise ValueError(f"CSV is missing required '{col_name}' column.")

    n = len(df)
    values = pd.to_numeric(df[col_name], errors="coerce").to_numpy(dtype=float)
    bin_starts = np.arange(1, n + 1, bin_size)
    binned_means = np.full(len(bin_starts), np.nan, dtype=float)
    for i, start in enumerate(bin_starts):
        end = min(start + bin_size - 1, n)
        binned_means[i] = np.nanmean(values[start - 1 : end])
    return binned_means


def metric_per_bin_after_training(
    df: pd.DataFrame,
    col_name: str,
    bin_size: int,
    min_eval_to_plot: int,
    source_scope: str,
) -> Tuple[int, np.ndarray]:
    if "source" not in df.columns:
        raise ValueError("CSV is missing required 'source' column.")
    if col_name not in df.columns:
        raise ValueError(f"CSV is missing required '{col_name}' column.")

    src = df["source"].astype(str).str.lower().to_numpy()
    predicted_indices = np.where(src == "predicted")[0]
    if predicted_indices.size == 0:
        raise ValueError("No predicted samples found; cannot determine training start.")

    model_start_eval = int(predicted_indices[0] + 1)
    start_eval = max(model_start_eval, int(min_eval_to_plot))
    start_idx = start_eval - 1
    if start_idx >= len(df):
        raise ValueError("Requested plot start evaluation is beyond run length.")

    values = pd.to_numeric(df[col_name], errors="coerce").to_numpy(dtype=float)
    values = values[start_idx:]
    src = src[start_idx:]

    n = len(values)
    bin_starts = np.arange(1, n + 1, bin_size)
    binned_means = np.full(len(bin_starts), np.nan, dtype=float)
    for i, start in enumerate(bin_starts):
        end = min(start + bin_size - 1, n)
        chunk_values = values[start - 1 : end]
        chunk_src = src[start - 1 : end]
        if source_scope == "predicted":
            mask = chunk_src == "predicted"
            chunk_values = chunk_values[mask]
        elif source_scope == "simulated":
            mask = chunk_src != "predicted"
            chunk_values = chunk_values[mask]
        binned_means[i] = np.nanmean(chunk_values) if chunk_values.size else np.nan
    return start_eval, binned_means


def aggregate_metric_bins(
    paths: List[str],
    col_name: str,
    bin_size: int,
    min_eval_to_plot: int,
    source_scope: str,
) -> Tuple[np.ndarray, np.ndarray, int]:
    all_runs = []
    max_bins = 0
    start_evals = []
    for path in paths:
        df = pd.read_csv(path)
        start_eval, per_bin = metric_per_bin_after_training(
            df,
            col_name,
            bin_size,
            min_eval_to_plot,
            source_scope,
        )
        start_evals.append(start_eval)
        all_runs.append(per_bin)
        max_bins = max(max_bins, len(per_bin))

    matrix = np.full((len(paths), max_bins), np.nan, dtype=float)
    for i, per_bin in enumerate(all_runs):
        matrix[i, : len(per_bin)] = per_bin
    start_eval_ref = int(round(float(np.nanmedian(np.array(start_evals, dtype=float)))))
    return np.arange(max_bins), matrix, start_eval_ref


def plot_trajectory(ax, stats: Dict[str, np.ndarray], label_prefix: str, color: str):
    x = np.arange(1, len(stats["mean"]) + 1)
    ax.plot(x, stats["mean"], color=color, linewidth=2.0, label=f"{label_prefix} mean")
    ax.plot(x, stats["median"], color=color, linewidth=1.6, linestyle="--", label=f"{label_prefix} median")
    ax.fill_between(
        x,
        stats["p05"],
        stats["p95"],
        color=color,
        alpha=0.18,
        label=f"{label_prefix} 5-95 percentile",
    )


def plot_bin_variability(ax, x: np.ndarray, stats: Dict[str, np.ndarray], label_prefix: str, color: str):
    ax.plot(x, stats["mean"], color=color, linewidth=2.0, label=f"{label_prefix} mean")
    ax.plot(x, stats["median"], color=color, linewidth=1.6, linestyle="--", label=f"{label_prefix} median")
    ax.fill_between(
        x,
        stats["p05"],
        stats["p95"],
        color=color,
        alpha=0.18,
        label=f"{label_prefix} 5-95 percentile",
    )


def draw_boxplot_group(
    ax,
    x_base: np.ndarray,
    values: np.ndarray,
    offset: float,
    color: str,
    label: str,
    width: float,
):
    positions = x_base + offset
    series = [values[:, i][~np.isnan(values[:, i])] for i in range(values.shape[1])]
    bp = ax.boxplot(
        series,
        positions=positions,
        widths=width,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    for whisker in bp["whiskers"]:
        whisker.set_color(color)
    for cap in bp["caps"]:
        cap.set_color(color)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.0)
    # Create a proxy artist entry in legend.
    ax.plot([], [], color=color, linewidth=8, alpha=0.5, label=label)


def main():
    parser = argparse.ArgumentParser(description="Compare objective trajectories and predicted/simulated variability across wing_design runs.")
    pattern = "wing_eval_log_validation_*.csv"
    parser.add_argument(
        "--ens-dir",
        default=r"C:\apps\code\Py_Dev\WS-AIMDO\simple_airfoil\CFD\airfoil\post\Wing-design_run_Ens_400",
        help="Folder with Bayesian-ensemble run CSVs.",
    )
    parser.add_argument(
        "--base-dir",
        default=r"C:\apps\code\Py_Dev\WS-AIMDO\simple_airfoil\CFD\airfoil\post\Wing-design_run_400_mads",
        help="Folder with baseline run CSVs.",
    )
    parser.add_argument("--pattern", default=pattern, help="CSV glob pattern.")
    parser.add_argument("--bin-size", type=int, default=50, help="Bin size for ensemble probability trajectories.")
    parser.add_argument(
        "--source-scope",
        choices=["all", "predicted", "simulated"],
        default="all",
        help="Samples included in lower-panel probabilities after model training starts.",
    )
    parser.add_argument(
        "--no-plot-until",
        type=int,
        default=100,
        help="Keep x-axis visible up to this evaluation with no lower-panel probability curves.",
    )
    parser.add_argument(
        "--out-file",
        default=r"C:\apps\code\Py_Dev\WS-AIMDO\simple_airfoil\CFD\airfoil\post\Wing_design_validation_stats.png",
        help="Output figure path.",
    )
    args = parser.parse_args()

    ens_paths = collect_csv_paths(args.ens_dir, args.pattern)
    base_paths = collect_csv_paths(args.base_dir, args.pattern)

    ens_matrix = build_trajectory_matrix(ens_paths)
    base_matrix = build_trajectory_matrix(base_paths)
    ens_stats = summarize_matrix(ens_matrix)
    base_stats = summarize_matrix(base_matrix)

    bin_ids_ens, pred_ens, _ = aggregate_bin_counts(ens_paths, args.bin_size)
    bin_ids_base, pred_base, _ = aggregate_bin_counts(base_paths, args.bin_size)
    max_bins = max(len(bin_ids_ens), len(bin_ids_base))
    x_base = np.arange(max_bins, dtype=float) * args.bin_size + args.bin_size * 0.5

    ens_total_pred_per_run = np.nansum(pred_ens, axis=1)
    base_total_pred_per_run = np.nansum(pred_base, axis=1)
    ens_avg_total_pred = float(np.nanmean(ens_total_pred_per_run))
    base_avg_total_pred = float(np.nanmean(base_total_pred_per_run))

    fig, axes = plt.subplots(2, 1, figsize=(13, 10), constrained_layout=True)
    max_eval = int(max(ens_matrix.shape[1], base_matrix.shape[1]))
    common_xticks = np.arange(0, max_eval + 1, args.bin_size)

    ax0 = axes[0]
    plot_trajectory(ax0, ens_stats, "Bayesian Ensemble", "tab:blue")
    plot_trajectory(ax0, base_stats, "Without Ensemble", "tab:orange")
    ax0.set_title("Objective Trajectory Statistics Across Runs")
    ax0.set_xlabel("Evaluation")
    ax0.set_ylabel("Objective (best feasible Cd)")
    ax0.set_xlim(0, max_eval)
    ax0.set_xticks(common_xticks)
    ax0.grid(True, alpha=0.25)
    summary_text = (
        f"Avg #predictions/run (Ensemble NNs): {ens_avg_total_pred:.1f}\n"
        f"Avg eval. reductions/run (Ensemble NNs): ~{100*ens_avg_total_pred/400:.1f}%\n"
    )
    ax0.text(
        0.01,
        0.97,
        summary_text,
        transform=ax0.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.75, "edgecolor": "0.7"},
    )
    ax0.legend(loc="best")

    ax1 = axes[1]
    box_width = args.bin_size * 0.2
    draw_boxplot_group(
        ax1,
        x_base,
        pred_base,
        -args.bin_size * 0.12,
        "sandybrown",
        "Without Ensemble: # predicted per bin",
        box_width,
    )
    draw_boxplot_group(
        ax1,
        x_base,
        pred_ens,
        +args.bin_size * 0.12,
        "skyblue",
        "Ensemble: # predicted per bin",
        box_width,
    )

    ax1.set_xlim(0, max_eval)
    ax1.set_xticks(common_xticks)
    ax1.set_title(f"Predicted Design Counts per {args.bin_size} Evaluations")
    ax1.set_xlabel("Evaluation")
    ax1.set_ylabel("# predicted designs in bin")
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.legend(loc="upper right", fontsize=9)

    out_dir = os.path.dirname(args.out_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.out_file, dpi=220)

    print(f"Saved figure: {args.out_file}")
    print(f"Ensemble runs: {len(ens_paths)} | Baseline runs: {len(base_paths)}")


if __name__ == "__main__":
    main()