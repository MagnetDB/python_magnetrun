"""
Plot trigger data from FEPC trigger files

This script reads and plots specific variables from trigger binary files.
"""

import argparse
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt

from python_magnetrun.cli_args import create_hybrid_parser
from python_magnetrun.hybrid.trigger.trigger_reader import (
    apply_calibration,
    create_time_array,
    find_trigger_directories,
    load_trigger_config,
    parse_trigger_directory,
    read_trigger_data,
)
from python_magnetrun.log_utils import get_logger, setup_logging

logger = get_logger()


def plot_trigger_variable(
    trigger_dir: Path,
    system: str,
    variable: str,
    endian: str = "big",
    show_plot: bool = True,
    save_path: Path | None = None,
    apply_calib: bool = True,
    cnv_dir: Path | None = None,
    fig_size: tuple = (12, 6),
    title: str | None = None,
):
    """
    Plot a trigger variable

    Parameters:
    -----------
    trigger_dir : Path
        Path to trigger directory
    system : str
        'FEPC-LNCMI' or 'FEPC-AUX-LNCMI'
    variable : str
        Variable name to plot
    endian : str
        'big' or 'little'
    show_plot : bool
        Show interactive plot
    save_path : Path, optional
        Save plot to file
    apply_calib : bool
        Apply calibration
    cnv_dir : Path, optional
        Directory containing CNV files
    fig_size : tuple
        Figure size (width, height)
    title : str, optional
        Plot title
    """
    # Parse trigger info
    trigger_info = parse_trigger_directory(trigger_dir)

    # Read data
    logger.info(f"Reading {variable} from {trigger_dir.name}...")
    data, timestamp, config = read_trigger_data(
        trigger_dir, system, variable_name=variable, endian=endian
    )

    # Find card and channel info
    card = None
    channel = None
    for c in config.cards:
        if variable in c.variable_names:
            card = c
            channel = c.variable_names.index(variable)
            break

    if card is None:
        raise ValueError(f"Variable {variable} not found in configuration")

    assert (
        channel is not None
    ), f"Variable {variable} found in card but channel index is None"

    # Apply calibration if requested
    if apply_calib and card.calibrations and channel < len(card.calibrations):
        calib = card.calibrations[channel]

        if cnv_dir is None:
            # Try to find CNV directory in trigger system directory
            cnv_dir = trigger_dir / system

        logger.info(f"Applying calibration: a={calib.a}, b={calib.b}")
        data = apply_calibration(data, calib, cnv_dir)
        y_label = f"{variable} ({calib.unit})" if calib.unit else variable
    else:
        y_label = f"{variable} (raw ADC)"

    # Create time array
    time = create_time_array(len(data))

    # Convert to time relative to trigger (trigger at PRE seconds)
    pre_time = trigger_info.pre_samples / 10000.0  # 20s
    time_rel = time - pre_time

    # Create plot
    fig, ax = plt.subplots(figsize=fig_size)

    # Plot data
    ax.plot(time_rel, data, linewidth=0.5, color="blue", alpha=0.7)

    # Mark trigger point
    ax.axvline(0, color="red", linestyle="--", linewidth=2, label="Trigger", alpha=0.7)

    # Labels
    ax.set_xlabel("Time relative to trigger (s)", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)

    if title is None:
        title = f"{variable} - {trigger_info.trigger_name}"
    ax.set_title(title, fontsize=14, fontweight="bold")

    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add trigger info text
    info_text = (
        f"Trigger: {trigger_info.timestamp.strftime('%Y-%m-%d %H:%M')}\n"
        f"Sample: {trigger_info.sample_idx}\n"
        f"PRE: {trigger_info.pre_samples / 10000:.1f}s, POST: {trigger_info.post_samples / 10000:.1f}s"
    )
    ax.text(
        0.02,
        0.98,
        info_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        fontsize=9,
        family="monospace",
    )

    plt.tight_layout()

    # Save if requested
    if save_path:
        logger.info(f"Saving plot to {save_path}")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    # Show if requested
    if show_plot:
        plt.show()

    return fig, ax


def plot_trigger_variables_subplots(
    trigger_dir: Path,
    system: str,
    variables: list[str],
    endian: str = "big",
    show_plot: bool = True,
    save_path: Path | None = None,
    apply_calib: bool = True,
    cnv_dir: Path | None = None,
    fig_size: tuple = (12, 10),
    title: str | None = None,
):
    """Plot multiple variables from a single trigger as subplots with shared x-axis.

    Parameters:
    -----------
    trigger_dir : Path
        Path to trigger directory
    system : str
        'FEPC-LNCMI' or 'FEPC-AUX-LNCMI'
    variables : list[str]
        Variable names to plot (one subplot each)
    endian : str
        'big' or 'little'
    show_plot : bool
        Show interactive plot
    save_path : Path, optional
        Save plot to file
    apply_calib : bool
        Apply calibration
    cnv_dir : Path, optional
        Directory containing CNV files
    fig_size : tuple
        Figure size (width, height)
    title : str, optional
        Overall figure title
    """
    trigger_info = parse_trigger_directory(trigger_dir)
    n_vars = len(variables)

    fig, axes = plt.subplots(n_vars, 1, figsize=fig_size, sharex=True)
    if n_vars == 1:
        axes = [axes]

    pre_time = trigger_info.pre_samples / 10000.0

    for ax, variable in zip(axes, variables, strict=False):
        try:
            data, _timestamp, config = read_trigger_data(
                trigger_dir, system, variable_name=variable, endian=endian
            )

            card = None
            channel = None
            for c in config.cards:
                if variable in c.variable_names:
                    card = c
                    channel = c.variable_names.index(variable)
                    break

            if apply_calib and card and card.calibrations and channel is not None and channel < len(card.calibrations):
                calib = card.calibrations[channel]
                _cnv_dir = cnv_dir if cnv_dir is not None else trigger_dir / system
                data = apply_calibration(data, calib, _cnv_dir)
                y_label = f"{variable} ({calib.unit})" if calib.unit else variable
            else:
                y_label = f"{variable} (raw ADC)"

            time = create_time_array(len(data))
            time_rel = time - pre_time

            ax.plot(time_rel, data, linewidth=0.5, color="blue", alpha=0.7)
            ax.axvline(0, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="Trigger")
            ax.set_ylabel(y_label, fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        except (OSError, ValueError, RuntimeError) as e:
            logger.error(f"Error reading {variable}: {e}")
            ax.text(0.5, 0.5, f"Error: {e}", transform=ax.transAxes, ha="center", va="center")
            ax.set_ylabel(variable, fontsize=10)

    axes[-1].set_xlabel("Time relative to trigger (s)", fontsize=12)

    if title is None:
        title = f"{trigger_info.trigger_name}"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Add trigger info text on first subplot
    info_text = (
        f"Trigger: {trigger_info.timestamp.strftime('%Y-%m-%d %H:%M')}\n"
        f"PRE: {trigger_info.pre_samples / 10000:.1f}s, POST: {trigger_info.post_samples / 10000:.1f}s"
    )
    axes[0].text(
        0.02, 0.98, info_text,
        transform=axes[0].transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        fontsize=9,
        family="monospace",
    )

    plt.tight_layout()

    if save_path:
        logger.info(f"Saving plot to {save_path}")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show_plot:
        plt.show()

    return fig, axes


def plot_multiple_triggers(
    base_dir: Path,
    date: str,
    system: str,
    variable: str,
    endian: str = "big",
    show_plot: bool = True,
    save_path: Path | None = None,
    apply_calib: bool = True,
    cnv_dir: Path | None = None,
    fig_size: tuple = (14, 8),
):
    """
    Plot same variable from multiple triggers on the same day

    Parameters:
    -----------
    base_dir : Path
        Base directory containing trigger subdirectory
    date : str
        Date in YYYY-MM-DD format
    system : str
        'FEPC-LNCMI' or 'FEPC-AUX-LNCMI'
    variable : str
        Variable name to plot
    endian : str
        'big' or 'little'
    show_plot : bool
        Show interactive plot
    save_path : Path, optional
        Save plot to file
    apply_calib : bool
        Apply calibration
    cnv_dir : Path, optional
        Directory containing CNV files
    fig_size : tuple
        Figure size (width, height)
    """
    # Find all triggers for this date
    triggers = find_trigger_directories(base_dir, date)

    if not triggers:
        raise ValueError(f"No triggers found for date {date}")

    logger.info(f"Found {len(triggers)} trigger(s) for {date}")

    # Create subplots
    n_triggers = len(triggers)
    fig, axes = plt.subplots(n_triggers, 1, figsize=fig_size, sharex=True)

    if n_triggers == 1:
        axes = [axes]

    # Plot each trigger
    for _i, (trigger_dir, ax) in enumerate(zip(triggers, axes, strict=False)):
        logger.info(f"Processing {trigger_dir.name}...")

        try:
            # Parse trigger info
            trigger_info = parse_trigger_directory(trigger_dir)

            # Read data
            data, timestamp, config = read_trigger_data(
                trigger_dir, system, variable_name=variable, endian=endian
            )

            # Find card and channel info
            card = None
            channel = None
            for c in config.cards:
                if variable in c.variable_names:
                    card = c
                    channel = c.variable_names.index(variable)
                    break

            # Apply calibration
            if (
                apply_calib
                and card
                and card.calibrations
                and channel is not None
                and channel < len(card.calibrations)
            ):
                calib = card.calibrations[channel]

                if cnv_dir is None:
                    cnv_dir = trigger_dir / system

                data = apply_calibration(data, calib, cnv_dir)
                y_label = f"{variable} ({calib.unit})" if calib.unit else variable
            else:
                y_label = f"{variable} (raw ADC)"

            # Create time array
            time = create_time_array(len(data))
            pre_time = trigger_info.pre_samples / 10000.0
            time_rel = time - pre_time

            # Plot
            ax.plot(time_rel, data, linewidth=0.5, color="blue", alpha=0.7)
            ax.axvline(0, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
            ax.set_ylabel(y_label, fontsize=10)
            ax.grid(True, alpha=0.3)

            # Add trigger time as title
            ax.set_title(
                trigger_info.timestamp.strftime("%H:%M:%S"), fontsize=10, loc="left"
            )

        except (OSError, ValueError, RuntimeError) as e:
            logger.error(f"Error processing {trigger_dir.name}: {e}")
            ax.text(
                0.5,
                0.5,
                f"Error: {str(e)}",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )

    # Common labels
    axes[-1].set_xlabel("Time relative to trigger (s)", fontsize=12)
    fig.suptitle(f"{variable} - {date} - All Triggers", fontsize=14, fontweight="bold")

    plt.tight_layout()

    # Save if requested
    if save_path:
        logger.info(f"Saving plot to {save_path}")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    # Show if requested
    if show_plot:
        plt.show()

    return fig, axes


def resolve_trigger_dirs(
    input_dir: str | None,
    hybrid_datadir: str | None,
    hybrid_date: str | None,
    all_for_date: bool = False,
) -> list[Path]:
    """Resolve trigger directories from a direct path or hybrid directory.

    Resolution order:
    1. If *input_dir* is given and exists on disk — use it directly
       (or all dirs for that date if *all_for_date* is True).
    2. If *input_dir* is given but not found — infer date from the directory
       name (``TRIGGER__YYYY-MM-DD__HH-MM``), then search
       ``hybrid_datadir/trigger/`` (all dirs if *all_for_date* is True).
    3. If no *input_dir* — list all trigger directories for *hybrid_date* in
       ``hybrid_datadir/trigger/``.

    Parameters
    ----------
    input_dir:
        Optional path (or bare name) of a trigger directory.
    hybrid_datadir:
        Base directory for hybrid data (``find_trigger_directories`` appends
        ``/trigger`` internally).
    hybrid_date:
        Date string ``YYYY-MM-DD`` used when searching the hybrid tree.
    all_for_date:
        When True and *input_dir* is given, return all trigger directories for
        the inferred date instead of just the matched one.

    Returns
    -------
    list[Path]
        Resolved trigger directory paths, never empty.

    Raises
    ------
    FileNotFoundError
        When the directory cannot be found or the search yields no matches.
    ValueError
        When insufficient information is provided to locate the directories.
    """
    if input_dir:
        candidate = Path(input_dir)
        if candidate.exists() and candidate.is_dir() and not all_for_date:
            return [candidate]
        # If all_for_date, fall through to resolve all dirs for the inferred date

        if not hybrid_datadir:
            raise FileNotFoundError(f"Trigger directory not found: {input_dir}")

        # Infer date from directory name TRIGGER__YYYY-MM-DD__HH-MM
        inferred_date: str | None = None
        m = re.match(r"TRIGGER_+(\d{4}-\d{2}-\d{2})_+\d{2}-\d{2}", candidate.name)
        if m:
            inferred_date = m.group(1)

        resolved_date = hybrid_date or inferred_date
        logger.info(f"Inferred from directory name: date='{resolved_date}'")

        if not resolved_date:
            raise ValueError(
                f"'{input_dir}' not found; could not infer date from directory name. "
                "Provide --hybrid_date."
            )

        matches = find_trigger_directories(Path(hybrid_datadir), resolved_date)
        if all_for_date:
            result = matches
        else:
            # Try exact name match first, fall back to all dirs for that date
            exact = [d for d in matches if d.name == candidate.name]
            result = exact or matches
        if not result:
            raise FileNotFoundError(
                f"'{input_dir}' not found and no trigger dirs for "
                f"{resolved_date} in {hybrid_datadir}/trigger"
            )
        if all_for_date:
            logger.info(f"Resolved '{input_dir}' → {len(result)} dir(s) for {resolved_date}")
            return result
        logger.info(f"Resolved '{input_dir}' → {result[0]}")
        return result[:1]

    # No input_dir — collect all trigger dirs from the hybrid tree
    if not hybrid_date:
        raise ValueError("Provide either input_dir or --hybrid_date")
    if not hybrid_datadir:
        raise ValueError("--hybrid_datadir is required when no input_dir is given")

    trigger_dirs = find_trigger_directories(Path(hybrid_datadir), hybrid_date)
    if not trigger_dirs:
        raise FileNotFoundError(
            f"No trigger directories found for {hybrid_date} in {hybrid_datadir}/trigger"
        )
    logger.info(f"Found {len(trigger_dirs)} trigger dir(s) for {hybrid_date}")
    return trigger_dirs


def main():
    """Command-line interface"""
    hybrid_parser = create_hybrid_parser()
    parser = argparse.ArgumentParser(
        description="Plot FEPC trigger data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[hybrid_parser],
        epilog="""
Examples:
  # Plot single trigger by directory name
  python plot_trigger_data.py TRIGGER__2025-11-05__08-16 --variable I_H1
  python plot_trigger_data.py TRIGGER__2025-11-05__08-16 --variable I_H1 I_BOB

  # Plot single trigger resolved from hybrid dir
  python plot_trigger_data.py TRIGGER__2025-11-05__08-16 --hybrid_datadir /data/hybrid --variable I_H1

  # Plot all triggers for a date
  python plot_trigger_data.py --hybrid_date 2025-11-05 --variable I_H1 --all

  # Save plot without displaying
  python plot_trigger_data.py TRIGGER__2025-11-05__08-16 --variable I_H1 --save trigger_plot.png

  # Skip calibration
  python plot_trigger_data.py TRIGGER__2025-11-05__08-16 --variable I_H1 --no-calib
        """,
    )

    parser.add_argument(
        "input_dir",
        nargs="?",
        help="Trigger directory to process (optional if --hybrid_date is given)",
    )
    parser.add_argument(
        "--variable",
        type=str,
        nargs="+",
        default=None,
        metavar="VARIABLE",
        help="Variable name(s) to plot (one figure per variable)",
    )
    parser.add_argument(
        "--list-variables",
        action="store_true",
        help="List all variables available in the trigger config and exit",
    )
    parser.add_argument(
        "--all", action="store_true", help="Plot all triggers for the resolved date"
    )
    parser.add_argument(
        "--endian",
        type=str,
        choices=["big", "little"],
        default="big",
        help="Endianness of binary data",
    )
    parser.add_argument("--save", type=Path, help="Save plot to file")
    parser.add_argument(
        "--no-show", action="store_true", help="Do not show interactive plot"
    )
    parser.add_argument("--no-calib", action="store_true", help="Skip calibration")
    parser.add_argument(
        "--cnv-dir", type=Path, help="Directory containing CNV calibration files"
    )
    parser.add_argument(
        "--fig-size",
        type=str,
        default="12,6",
        help="Figure size as width,height (default: 12,6)",
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

    # Parse figure size
    try:
        fig_size = tuple(map(float, args.fig_size.split(",")))
    except ValueError:
        logger.warning("Invalid fig-size format, using default (12,6)")
        fig_size = (12, 6)

    try:
        trigger_dirs = resolve_trigger_dirs(
            args.input_dir, args.hybrid_datadir, args.hybrid_date, all_for_date=args.all
        )
    except (FileNotFoundError, ValueError) as e:
        parser.error(str(e))

    if args.list_variables:
        config = load_trigger_config(trigger_dirs[0], args.fepc_system)
        if config is None:
            parser.error(f"Could not load config from {trigger_dirs[0]}")
        for card in config.cards:
            print(f"Card {card.slot} ({card.card_type}): {', '.join(card.variable_names)}")
        return

    if not args.variable:
        parser.error("--variable is required — check available variables with --list-variables")

    try:
        if args.all and len(trigger_dirs) > 1:
            # Multiple triggers: one figure per variable, subplots per trigger
            for var in args.variable:
                if args.save and len(args.variable) > 1:
                    save_path = args.save.with_stem(f"{args.save.stem}_{var}")
                else:
                    save_path = args.save
                plot_multiple_triggers(
                    trigger_dirs[0].parent.parent,  # hybrid_datadir
                    trigger_dirs[0].parent.name,    # YYYY-MM-DD from trigger dir parent
                    args.fepc_system,
                    var,
                    args.endian,
                    not args.no_show,
                    save_path,
                    not args.no_calib,
                    args.cnv_dir,
                    fig_size,
                )
        elif len(args.variable) > 1:
            # Single trigger, multiple variables: one figure with subplots
            plot_trigger_variables_subplots(
                trigger_dirs[0],
                args.fepc_system,
                args.variable,
                args.endian,
                not args.no_show,
                args.save,
                not args.no_calib,
                args.cnv_dir,
                fig_size,
            )
        else:
            plot_trigger_variable(
                trigger_dirs[0],
                args.fepc_system,
                args.variable[0],
                args.endian,
                not args.no_show,
                args.save,
                not args.no_calib,
                args.cnv_dir,
                fig_size,
            )

    except (OSError, ValueError, RuntimeError) as e:
        logger.error(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
