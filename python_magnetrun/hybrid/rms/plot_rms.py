"""
Quick RMS Data Plotter
======================

A simple utility to quickly plot variables from RMS files.
"""

import argparse
import sys
from pathlib import Path
from rms_reader import RMSFileReader
import matplotlib.pyplot as plt
import numpy as np


def plot_variables(filepath, variable_names, output_file=None, same_plot=False):
    """
    Plot specific variables from an RMS file.

    Parameters:
    -----------
    filepath : str
        Path to RMS file
    variable_names : list
        List of variable names to plot
    output_file : str, optional
        Output filename for saving plot
    same_plot : bool
        If True, plot all variables on same axes; if False, create subplots
    """
    reader = RMSFileReader(filepath)
    df = reader.read()

    # Filter valid variables
    valid_vars = [v for v in variable_names if v in df.columns]

    if not valid_vars:
        print("Error: None of the requested variables found in file")
        print(f"Requested: {variable_names}")
        print(f"Available variables: {df.columns.tolist()}")
        return

    if same_plot:
        # Single plot with multiple y-axes if needed
        fig, ax1 = plt.subplots(figsize=(12, 6))

        colors = plt.cm.tab10(np.linspace(0, 1, len(valid_vars)))

        for i, var in enumerate(valid_vars):
            if i == 0:
                ax = ax1
                color = colors[i]
            else:
                ax = ax1.twinx()
                ax.spines["right"].set_position(("outward", 60 * (i - 1)))
                color = colors[i]

            ax.plot(df.index, df[var], color=color, label=var, linewidth=1.5)
            ax.set_ylabel(var, color=color)
            ax.tick_params(axis="y", labelcolor=color)

        ax1.set_xlabel("Time")
        ax1.grid(True, alpha=0.3)
        fig.suptitle(f'Variables: {", ".join(valid_vars)}', fontsize=12)

    else:
        # Separate subplots
        n_plots = len(valid_vars)
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 2.5 * n_plots))
        if n_plots == 1:
            axes = [axes]

        fig.suptitle(f"RMS File: {Path(filepath).name}", fontsize=14, fontweight="bold")

        for ax, var in zip(axes, valid_vars):
            ax.plot(df.index, df[var], linewidth=1.5)
            ax.set_ylabel(var)
            ax.set_xlabel("Time")
            ax.grid(True, alpha=0.3)

            # Add statistics
            mean_val = df[var].mean()
            std_val = df[var].std()
            min_val = df[var].min()
            max_val = df[var].max()

            stats_text = f"μ={mean_val:.3f}, σ={std_val:.3f}, min={min_val:.3f}, max={max_val:.3f}"
            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                verticalalignment="top",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()

    return fig


def main():
    """Main entry point for the RMS plotter CLI."""
    parser = argparse.ArgumentParser(
        prog="plot_rms",
        description="Quick RMS Data Plotter - Plot variables from RMS files",
    )
    parser.add_argument(
        "file",
        type=str,
        help="Path to the RMS file",
    )
    parser.add_argument(
        "variables",
        type=str,
        nargs="+",
        help="Variable names to plot",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output filename for saving the plot (e.g., output.png)",
    )
    parser.add_argument(
        "--same-plot",
        action="store_true",
        help="Plot all variables on the same axes with multiple y-axes",
    )

    args = parser.parse_args()

    try:
        plot_variables(args.file, args.variables, args.output, args.same_plot)
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
