"""
VProcess Data Plotting Utility
===============================

Plotting utilities for VProcess data visualization.
Follows patterns from hybrid.plotting and plot_fepc_data.py modules.

Usage:
    python plot_vprocess.py data.vprocess --vars TT115A TT508A
    python plot_vprocess.py data.vprocess --overview
    python plot_vprocess.py data.vprocess --compare TT115A TT508A
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from vprocess_reader import VProcessFileReader, read_vprocess_file

# Setup logger
logger = logging.getLogger(__name__)


def plot_variables(
    filepath: str,
    variables: List[str],
    start_time: Optional[pd.Timestamp] = None,
    end_time: Optional[pd.Timestamp] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    layout: str = "subplots",
) -> Tuple[Figure, Axes]:
    """
    Plot selected variables from VProcess file.
    
    Parameters
    ----------
    filepath : str
        Path to VProcess file
    variables : list of str
        Variable names to plot
    start_time : pd.Timestamp, optional
        Start time for filtering
    end_time : pd.Timestamp, optional
        End time for filtering
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display the plot
    layout : str
        'subplots' or 'overlay'
    
    Returns
    -------
    tuple
        (figure, axes)
    """
    # Read data
    df = read_vprocess_file(filepath)

    # Filter by time
    if start_time:
        df = df[df.index >= start_time]
    if end_time:
        df = df[df.index <= end_time]

    # Check available variables
    available_vars = [v for v in variables if v in df.columns]
    if not available_vars:
        logger.error(f"None of the requested variables found in file")
        logger.info(f"Available variables: {list(df.columns[:10])}...")
        return None, None

    missing_vars = set(variables) - set(available_vars)
    if missing_vars:
        logger.warning(f"Variables not found: {', '.join(missing_vars)}")

    df_plot = df[available_vars]

    # Create figure
    if layout == "overlay":
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        axes = [ax]

        for var in available_vars:
            ax.plot(df_plot.index, df_plot[var], label=var, linewidth=0.8, alpha=0.8)

        ax.set_ylabel("Value", fontsize=10, fontweight="bold")
        ax.set_xlabel("Time", fontsize=10, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    else:  # subplots
        n_vars = len(available_vars)
        fig, axes = plt.subplots(n_vars, 1, figsize=(12, 3 * n_vars), sharex=True)

        if n_vars == 1:
            axes = [axes]

        for i, var in enumerate(available_vars):
            ax = axes[i]
            ax.plot(df_plot.index, df_plot[var], linewidth=0.8)
            ax.set_ylabel(var, fontsize=10, fontweight="bold")
            ax.grid(True, alpha=0.3)

            # Add statistics
            mean_val = df_plot[var].mean()
            std_val = df_plot[var].std()
            min_val = df_plot[var].min()
            max_val = df_plot[var].max()

            stats_text = (
                f"μ={mean_val:.2f}, σ={std_val:.2f}\n"
                f"min={min_val:.2f}, max={max_val:.2f}"
            )
            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                verticalalignment="top",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

    # Format x-axis
    axes[-1].set_xlabel("Time", fontsize=10, fontweight="bold")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Set title
    if title is None:
        title = f"VProcess Data: {Path(filepath).name}"
    fig.suptitle(title, fontsize=12, fontweight="bold")

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Figure saved to: {save_path}")

    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()

    return fig, axes


def plot_overview(
    filepath: str,
    max_vars: int = 10,
    save_path: Optional[str] = None,
    show: bool = True,
) -> Tuple[Figure, Axes]:
    """
    Create overview plot with first N analog variables.
    
    Parameters
    ----------
    filepath : str
        Path to VProcess file
    max_vars : int
        Maximum number of variables to plot
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display the plot
    
    Returns
    -------
    tuple
        (figure, axes)
    """
    # Get variable names
    reader = VProcessFileReader(filepath)
    reader.parse_header()

    # Get first max_vars analog variables
    analog_vars = [v.name for v in reader.variables if v.is_analog][:max_vars]

    if not analog_vars:
        logger.error("No analog variables found")
        return None, None

    return plot_variables(
        filepath,
        analog_vars,
        title=f"VProcess Overview: {Path(filepath).name}",
        save_path=save_path,
        show=show,
    )


def plot_comparison(
    filepath: str,
    var1: str,
    var2: str,
    save_path: Optional[str] = None,
    show: bool = True,
) -> Tuple[Figure, Axes]:
    """
    Create comparison plot of two variables.
    
    Parameters
    ----------
    filepath : str
        Path to VProcess file
    var1 : str
        First variable name
    var2 : str
        Second variable name
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display the plot
    
    Returns
    -------
    tuple
        (figure, axes)
    """
    # Read data
    df = read_vprocess_file(filepath)

    if var1 not in df.columns or var2 not in df.columns:
        logger.error(f"Variables not found in file")
        return None, None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Time series plots
    axes[0, 0].plot(df.index, df[var1], linewidth=0.8)
    axes[0, 0].set_ylabel(var1, fontweight="bold")
    axes[0, 0].set_title("Time Series", fontweight="bold")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45, ha="right")

    axes[1, 0].plot(df.index, df[var2], linewidth=0.8, color="orange")
    axes[1, 0].set_ylabel(var2, fontweight="bold")
    axes[1, 0].set_xlabel("Time", fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Scatter plot
    axes[0, 1].scatter(df[var1], df[var2], alpha=0.5, s=1)
    axes[0, 1].set_xlabel(var1, fontweight="bold")
    axes[0, 1].set_ylabel(var2, fontweight="bold")
    axes[0, 1].set_title("Correlation", fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3)

    # Correlation coefficient
    corr = df[[var1, var2]].corr().iloc[0, 1]
    axes[0, 1].text(
        0.05,
        0.95,
        f"r = {corr:.3f}",
        transform=axes[0, 1].transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # Histograms
    axes[1, 1].hist(df[var1], bins=50, alpha=0.7, label=var1)
    axes[1, 1].hist(df[var2], bins=50, alpha=0.7, label=var2)
    axes[1, 1].set_xlabel("Value", fontweight="bold")
    axes[1, 1].set_ylabel("Frequency", fontweight="bold")
    axes[1, 1].set_title("Distributions", fontweight="bold")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f"Comparison: {var1} vs {var2}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Figure saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig, axes


def plot_heatmap(
    filepath: str,
    variables: Optional[List[str]] = None,
    max_vars: int = 20,
    save_path: Optional[str] = None,
    show: bool = True,
) -> Tuple[Figure, Axes]:
    """
    Create correlation heatmap for variables.
    
    Parameters
    ----------
    filepath : str
        Path to VProcess file
    variables : list of str, optional
        Variable names (default: first max_vars analog variables)
    max_vars : int
        Maximum number of variables if not specified
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display the plot
    
    Returns
    -------
    tuple
        (figure, axes)
    """
    df = read_vprocess_file(filepath)

    # Select variables
    if variables is None:
        reader = VProcessFileReader(filepath)
        reader.parse_header()
        analog_vars = [v.name for v in reader.variables if v.is_analog][:max_vars]
        variables = analog_vars

    available_vars = [v for v in variables if v in df.columns]
    if not available_vars:
        logger.error("No variables available")
        return None, None

    df_subset = df[available_vars]

    # Calculate correlation
    corr_matrix = df_subset.corr()

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Correlation", rotation=270, labelpad=20)

    # Labels
    ax.set_xticks(np.arange(len(available_vars)))
    ax.set_yticks(np.arange(len(available_vars)))
    ax.set_xticklabels(available_vars, rotation=45, ha="right")
    ax.set_yticklabels(available_vars)

    # Title
    plt.title("Variable Correlation Heatmap", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Figure saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig, ax


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Plot VProcess data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot specific variables
  python plot_vprocess.py data.vprocess --vars TT115A TT508A
  
  # Plot overview of first 10 variables
  python plot_vprocess.py data.vprocess --overview
  
  # Compare two variables
  python plot_vprocess.py data.vprocess --compare TT115A TT508A
  
  # Create correlation heatmap
  python plot_vprocess.py data.vprocess --heatmap
  
  # Save plot without displaying
  python plot_vprocess.py data.vprocess --vars TT115A --save plot.png --no-show
        """,
    )

    parser.add_argument("filepath", help="Path to VProcess file")
    parser.add_argument("--vars", nargs="+", help="Variable names to plot")
    parser.add_argument(
        "--overview", action="store_true", help="Plot overview of first N variables"
    )
    parser.add_argument(
        "--max-vars",
        type=int,
        default=10,
        help="Maximum variables for overview (default: 10)",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("VAR1", "VAR2"),
        help="Compare two variables",
    )
    parser.add_argument(
        "--heatmap",
        action="store_true",
        help="Create correlation heatmap",
    )
    parser.add_argument(
        "--layout",
        choices=["subplots", "overlay"],
        default="subplots",
        help="Plot layout (default: subplots)",
    )
    parser.add_argument("--save", "-s", help="Save figure to file")
    parser.add_argument(
        "--no-show", action="store_true", help="Do not display plot (useful with --save)"
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Choose plot type
    if args.heatmap:
        plot_heatmap(
            args.filepath,
            variables=args.vars,
            max_vars=args.max_vars,
            save_path=args.save,
            show=not args.no_show,
        )

    elif args.compare:
        plot_comparison(
            args.filepath,
            args.compare[0],
            args.compare[1],
            save_path=args.save,
            show=not args.no_show,
        )

    elif args.overview:
        plot_overview(
            args.filepath,
            max_vars=args.max_vars,
            save_path=args.save,
            show=not args.no_show,
        )

    elif args.vars:
        plot_variables(
            args.filepath,
            args.vars,
            save_path=args.save,
            show=not args.no_show,
            layout=args.layout,
        )

    else:
        print("Please specify --vars, --overview, --compare, or --heatmap")
        print("Use --help for more information")


if __name__ == "__main__":
    main()
