import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import cm
import numpy as np
import pandas as pd

from .model import Coefs

def visualize_runoff(
        dates,
        runoff_target,
        total_runoff,
        precip=None,
        show=True,
        save_path=None
    ):
    start, end = dates[-1], dates[0]

    runoff_max = max(runoff_target.max(), np.array(total_runoff).max())
    fig, ax1 = plt.subplots(figsize=(20, 8))

    ax1.plot(
        dates,
        total_runoff,
        label="Simulated Runoff",
        color="black"
    )
    ax1.plot(
        dates,
        runoff_target,
        label="Observed Runoff",
        color="tab:orange"
    )
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Runoff [m³/s]")
    ax1.set_ylim(0, runoff_max)
    ax1.legend(loc="upper left")
    ax1.set_title(f"Best Runoff Simulation ({start} to {end})")

    if end - start > 365:
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    else:
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.tick_params(axis='x', rotation=45)

    if precip is not None:
        precip_max = max(precip) * 2.5

        # 강수량 바 차트
        ax2 = ax1.twinx()
        ax2.bar(dates, precip, label="Precipitation", color='tab:blue', width=1.0)
        ax2.set_ylabel("Precipitation [mm]")
        ax2.set_ylim(precip_max, 0)
        ax2.legend(loc='upper right')

    plt.tight_layout()

    if show:
        plt.show()

    if save_path:
        plt.savefig({save_path})

def visualize_tuning(params_history: list[(Coefs, float)], show=True, save_path=None):
    """
    Given a history of (params_dict, score) tuples, draw a parallel‐coordinates
    plot in which each line represents one trial, the x‐axes are the hyperparameter
    names (plus “score” at the far right), and the color of each line encodes that trial’s score.

    params_history: List of (params_dict, score).  For example:
        [
            ({"batch_size": 32, "lr": 0.01,   "n_layers": 2}, 0.72),
            ({"batch_size": 16, "lr": 0.001,  "n_layers": 3}, 0.81),
            ({"batch_size": 64, "lr": 0.005,  "n_layers": 2}, 0.75),
            ...
        ]
    """
    # 1) Build a DataFrame: each row has all hyperparameters + a “score” column
    records = []
    for params, score in params_history:
        row = params.get_parameters().copy()
        row["score"] = score
        records.append(row)

    df = pd.DataFrame(records)

    all_cols = list(df.columns)
    if "score" not in all_cols:
        raise ValueError("DataFrame must have a 'score' column.")

    # Extract names by prefix
    infil_cols = sorted([c for c in all_cols if c.startswith("infil")])
    run_cols = sorted([c for c in all_cols if c.startswith("run")])
    side_cols = sorted([c for c in all_cols if c.startswith("side")])
    storage_cols = sorted([c for c in all_cols if c.startswith("storage")])
    plot_cols = infil_cols + run_cols + side_cols + storage_cols + ["score"]

    df = df[plot_cols]

    # 3) Normalize each column to [0,1] for plotting
    normalized = df.copy().astype(float)
    for col in plot_cols:
        mn = normalized[col].min()
        mx = normalized[col].max()
        if mx > mn:
            normalized[col] = (normalized[col] - mn) / (mx - mn)
        else:
            normalized[col] = 0.5  # if all values are identical

    # 4) Prepare a colormap for “score.”  Map actual score values → [0,1].
    score_min = df["score"].min()
    score_max = df["score"].max()
    norm = plt.Normalize(vmin=score_min, vmax=score_max)
    cmap = cm.get_cmap("viridis")

    # 5) Create the parallel‐coordinates plot
    fig, ax = plt.subplots(figsize=(14, 7))
    x_positions = np.arange(len(plot_cols))  # e.g. [0,1,2,...,n_params]

    for idx in range(len(normalized)):
        y = normalized.iloc[idx].values
        score_val = df["score"].iloc[idx]
        color = cmap(norm(score_val))
        ax.plot(
            x_positions,
            y,
            color=color,
            alpha=0.4,
            linewidth=1.0
        )

    # 6) Label the x‐axis ticks with the hyperparameter names
    ax.set_xticks(x_positions)
    ax.set_xticklabels(plot_cols, rotation=45, ha="right", fontsize=9)

    # 7) For each column, add small “min” and “max” annotations at y=0 and y=1:
    for i, col in enumerate(plot_cols):
        mn = df[col].min()
        mx = df[col].max()

        # Plot a small tick at y=0 and y=1:
        ax.plot([i, i], [0.0, 1.0], marker="", color="gray", alpha=0.15, linewidth=0.5)

        # Print the numeric min just below y=0 on that axis
        ax.text(
            i - 0.02,  # slight left offset
            0.0 - 0.04,
            f"{mn:.3g}",
            ha="right",
            va="top",
            fontsize=7,
            color="black",
        )
        # Print the numeric max just above y=1 on that axis
        ax.text(
            i - 0.02,
            1.0 + 0.04,
            f"{mx:.3g}",
            ha="right",
            va="bottom",
            fontsize=7,
            color="black",
        )

    # 8) Draw a colorbar for the “score” dimension
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", pad=0.02)
    cbar.set_label("Score", rotation=270, labelpad=15)

    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Normalized parameter value", fontsize=10)
    ax.set_title("Hyperparameter Tuning\n(Parallel Coordinates, colored by score)", fontsize=12)

    plt.tight_layout()
    plt.show()
    