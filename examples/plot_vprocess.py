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
import logging
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from python_magnetrun.cli_args import create_hybrid_parser, validate_file_extension
from python_magnetrun.hybrid.vprocess.vprocess_reader import (
    VProcessFileReader,
    parse_vprocess_filename,
    read_vprocess_file,
)
from python_magnetrun.log_utils import get_logger, setup_logging
from python_magnetrun.plotting import PlotStyle, get_backend, plot_overlay, plot_subplots

# Setup logger
logger = get_logger()


def plot_variables(
    filepath: str,
    variables: list[str],
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
    title: str | None = None,
    save_path: str | None = None,
    show: bool = True,
    layout: str = "subplots",
) -> tuple[Figure | None, Any]:
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
        logger.error("None of the requested variables found in file")
        logger.info(f"Available variables: {list(df.columns[:10])}...")
        return None, None

    missing_vars = set(variables) - set(available_vars)
    if missing_vars:
        logger.warning(f"Variables not found: {', '.join(missing_vars)}")

    # Reset DatetimeIndex into a plain column so plot_subplots/plot_overlay can use it
    df_plot = df[available_vars].reset_index(names="t")

    b = get_backend("matplotlib")
    style = PlotStyle(figsize=(12, 3))

    if title is None:
        title = f"VProcess Data: {Path(filepath).name}"

    if layout == "overlay":
        fig = plot_overlay(df_plot, available_vars, t_col="t", backend=b, style=style, title=title)
        axes = list(fig._magnetrun_axes)
        ax = axes[0]
        ax.set_ylabel("Value", fontsize=10, fontweight="bold")
        ax.set_xlabel("Time", fontsize=10, fontweight="bold")
        ax.legend(loc="best")
    else:  # subplots
        fig = plot_subplots(df_plot, available_vars, t_col="t", backend=b, style=style, title=title)
        axes = list(fig._magnetrun_axes)

        for i, var in enumerate(available_vars):
            ax = axes[i]
            ax.set_ylabel(var, fontsize=10, fontweight="bold")

            # Add per-variable statistics box
            col_data = df_plot[var]
            stats_text = (
                f"μ={col_data.mean():.2f}, σ={col_data.std():.2f}\n"
                f"min={col_data.min():.2f}, max={col_data.max():.2f}"
            )
            ax.text(
                0.02, 0.98, stats_text,
                transform=ax.transAxes, verticalalignment="top", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

    # Format x-axis with date labels
    axes[-1].set_xlabel("Time", fontsize=10, fontweight="bold")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    for tick in axes[-1].xaxis.get_majorticklabels():
        tick.set_rotation(45)
        tick.set_horizontalalignment("right")

    fig.tight_layout()

    if save_path:
        b.save(fig, Path(save_path), dpi=150)
        logger.info(f"Figure saved to: {save_path}")

    if show:
        b.show(fig)
    else:
        plt.close(fig)

    return fig, axes


def plot_overview(
    filepath: str,
    max_vars: int = 10,
    save_path: str | None = None,
    show: bool = True,
) -> tuple[Figure | None, Any]:
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
    save_path: str | None = None,
    show: bool = True,
) -> tuple[Figure | None, Any]:
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
        logger.error("Variables not found in file")
        return None, None

    # 2×2 grid: time series, scatter, histograms — matplotlib-specific layout
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Time series plots
    axes[0, 0].plot(df.index, df[var1], linewidth=0.8)
    axes[0, 0].set_ylabel(var1, fontweight="bold")
    axes[0, 0].set_title("Time Series", fontweight="bold")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    for tick in axes[0, 0].xaxis.get_majorticklabels():
        tick.set_rotation(45)
        tick.set_horizontalalignment("right")

    axes[1, 0].plot(df.index, df[var2], linewidth=0.8, color="orange")
    axes[1, 0].set_ylabel(var2, fontweight="bold")
    axes[1, 0].set_xlabel("Time", fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    for tick in axes[1, 0].xaxis.get_majorticklabels():
        tick.set_rotation(45)
        tick.set_horizontalalignment("right")

    # Scatter plot
    axes[0, 1].scatter(df[var1], df[var2], alpha=0.5, s=1)
    axes[0, 1].set_xlabel(var1, fontweight="bold")
    axes[0, 1].set_ylabel(var2, fontweight="bold")
    axes[0, 1].set_title("Correlation", fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3)

    corr = df[[var1, var2]].corr().iloc[0, 1]
    axes[0, 1].text(
        0.05, 0.95, f"r = {corr:.3f}",
        transform=axes[0, 1].transAxes, verticalalignment="top",
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

    fig.suptitle(f"Comparison: {var1} vs {var2}", fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Figure saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes


def plot_heatmap(
    filepath: str,
    variables: list[str] | None = None,
    max_vars: int = 20,
    save_path: str | None = None,
    show: bool = True,
) -> tuple[Figure | None, Any]:
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

    # imshow + colorbar: matplotlib-specific layout
    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Correlation", rotation=270, labelpad=20)

    ax.set_xticks(np.arange(len(available_vars)))
    ax.set_yticks(np.arange(len(available_vars)))
    ax.set_xticklabels(available_vars, rotation=45, ha="right")
    ax.set_yticklabels(available_vars)
    ax.set_title("Variable Correlation Heatmap", fontsize=14, fontweight="bold", pad=20)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Figure saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def resolve_vprocess_filepaths(
    input_file: str | None,
    hybrid_datadir: str | None,
    hybrid_date: str | None,
) -> list[str]:
    """Resolve VProcess file paths from a direct path or hybrid directory.

    Resolution order:
    1. If *input_file* is given and exists on disk — use it directly.
    2. If *input_file* is given but not found — infer date from the filename
       using :func:`parse_vprocess_filename`, then search
       ``hybrid_datadir/vprocess/<date>/``.
    3. If no *input_file* — list all ``.vprocess`` files in
       ``hybrid_datadir/vprocess/<hybrid_date>/``.

    Parameters
    ----------
    input_file:
        Optional path (or bare filename) of a VProcess file.
    hybrid_datadir:
        Base directory for hybrid data (contains ``vprocess/`` subtree).
    hybrid_date:
        Date string ``YYYY-MM-DD`` used when searching the hybrid tree.

    Returns
    -------
    list[str]
        Resolved absolute file paths, never empty.

    Raises
    ------
    FileNotFoundError
        When the file cannot be found or the directory yields no matches.
    ValueError
        When insufficient information is provided to locate the files.
    """
    if input_file:
        candidate = Path(input_file)
        if candidate.exists():
            return [str(candidate)]

        if not hybrid_datadir:
            raise FileNotFoundError(f"File not found: {input_file}")

        # Infer date from filename using the existing parser
        inferred_date: str | None = None
        time_range = parse_vprocess_filename(candidate.name)
        if time_range:
            inferred_date = time_range[0].strftime("%Y-%m-%d")

        resolved_date = hybrid_date or inferred_date
        logger.info(f"Inferred from filename: date='{resolved_date}'")

        if not resolved_date:
            raise ValueError(
                f"'{input_file}' not found; could not infer date from filename. "
                "Provide --hybrid_date."
            )

        vprocess_dir = Path(hybrid_datadir) / "vprocess" / resolved_date
        matches = sorted(vprocess_dir.glob(f"*{candidate.name}*"))
        if not matches:
            matches = sorted(vprocess_dir.glob("*.vprocess"))
        if not matches:
            raise FileNotFoundError(
                f"'{input_file}' not found and no .vprocess files in {vprocess_dir}"
            )
        resolved = str(matches[0])
        logger.info(f"Resolved '{input_file}' → {resolved}")
        return [resolved]

    # No input_file — collect all files from the hybrid tree
    if not hybrid_date:
        raise ValueError("Provide either input_file or --hybrid_date")
    if not hybrid_datadir:
        raise ValueError("--hybrid_datadir is required when no input_file is given")

    vprocess_dir = Path(hybrid_datadir) / "vprocess" / hybrid_date
    vprocess_files = sorted(vprocess_dir.glob("*.vprocess"))
    if not vprocess_files:
        raise FileNotFoundError(f"No .vprocess files found in {vprocess_dir}")
    logger.info(f"Found {len(vprocess_files)} VProcess file(s) in {vprocess_dir}")
    return [str(f) for f in vprocess_files]


def main() -> None:
    """Main function for command-line usage."""
    hybrid_parser = create_hybrid_parser()
    parser = argparse.ArgumentParser(
        description="Plot VProcess data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[hybrid_parser],
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

    parser.add_argument(
        "input_file",
        nargs="?",
        type=validate_file_extension([".vprocess"]),
        help="VProcess file to process (optional if --hybrid_date is given)",
    )
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
        "--show",
        action="store_true",
        help="Display plot interactively (default when --save is not given)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="set logging level",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="path to log file (if not specified, logs to console)",
    )

    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper())
    setup_logging(level=log_level, log_file=args.log_file)
    logger.debug(f"Parsed arguments: {args}")

    try:
        filepaths = resolve_vprocess_filepaths(
            args.input_file, args.hybrid_datadir, args.hybrid_date
        )
    except (FileNotFoundError, ValueError) as e:
        parser.error(str(e))

    filepath = filepaths[0]
    logger.info(f"Using file: {filepath}")

    show = args.show or args.save is None  # default: show when not saving

    # Choose plot type
    if args.heatmap:
        plot_heatmap(
            filepath,
            variables=args.vars,
            max_vars=args.max_vars,
            save_path=args.save,
            show=show,
        )

    elif args.compare:
        plot_comparison(
            filepath,
            args.compare[0],
            args.compare[1],
            save_path=args.save,
            show=show,
        )

    elif args.overview:
        plot_overview(
            filepath,
            max_vars=args.max_vars,
            save_path=args.save,
            show=show,
        )

    elif args.vars:
        plot_variables(
            filepath,
            args.vars,
            save_path=args.save,
            show=show,
            layout=args.layout,
        )

    else:
        logger.warning("Please specify --vars, --overview, --compare, or --heatmap")
        logger.warning("Use --help for more information")


if __name__ == "__main__":
    main()
