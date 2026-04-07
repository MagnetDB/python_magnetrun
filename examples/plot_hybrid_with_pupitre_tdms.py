#!/usr/bin/env python3
"""
Example script to load and plot hybrid_run data with corresponding pupitre and tdms data.

This script demonstrates how to:
1. Load kHz data from FEPC-AUX-LNCMI or FEPC-LNCMI
2. Find and load corresponding pupitre (txt) and tdms (pigbrother) data based on date
3. Map field names between hybrid_run and pupitre/tdms using dictionaries
4. Plot all data sources on the same graph for comparison

The script uses the data directories as defined in python_magnetrun analysis configuration.
"""

import argparse
import glob
import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from python_magnetrun.cli_args import create_datadir_parser
from python_magnetrun.data_dirs import HYBRID_DATA_DIR
from python_magnetrun.hybrid.hybrid_run import HybridRun
from python_magnetrun.hybrid.utils import (
    binarize_signal,
    format_exception_location,
    log_exception,
    normalize_signal,
)
from python_magnetrun.log_utils import setup_logging
from python_magnetrun.MagnetRun import MagnetRun

logger = logging.getLogger(__name__)

# =============================================================================
# Field name mapping dictionaries
# =============================================================================

# setup for M8

# Map hybrid_run kHz field names to pupitre field names
HYBRID_TO_PUPITRE_MAP = {
    # FEPC-AUX-LNCMI channels
    "kHz/FEPC-AUX-LNCMI/ALIM1_J1": "Idcct1",  # Example: Helix current
    "kHz/FEPC-AUX-LNCMI/ALIM1_J2": "Idcct2",  # Example: Bottom coil current
    "kHz/FEPC-AUX-LNCMI/ALIM2_J1": "Idcct3",  # Example mapping
    "kHz/FEPC-AUX-LNCMI/ALIM2_J2": "Idcct4",  # Example mapping
}

# Map hybrid_run kHz field names to TDMS (pigbrother) field names
HYBRID_TO_TDMS_MAP = {
    # Map to Référence channels in Overview TDMS files
    "kHz/FEPC-AUX-LNCMI/ALIM1_J1": "Courants_Alimentations/Courant_A1",
    "kHz/FEPC-AUX-LNCMI/ALIM1_J2": "Courants_Alimentations/Courant_A2",
    "kHz/FEPC-AUX-LNCMI/ALIM2_J1": "Courants_Alimentations/Courant_A3",
    "kHz/FEPC-AUX-LNCMI/ALIM2_J2": "Courants_Alimentations/Courant_A4",
}


# =============================================================================
# Helper functions
# =============================================================================


def parse_date_from_filename(filename: str | Path) -> datetime:
    """
    Parse date from hybrid data directory or Overview filename.

    Parameters
    ----------
    filename : str or Path
        Filename or directory name to parse the date from.

    Returns
    -------
    datetime
        Parsed date.

    Raises
    ------
    ValueError
        If the date cannot be parsed from the filename.

    Examples
    --------
    - "2025-01-06" -> returns datetime(2025, 1, 6)
    - "M9_Overview_250127-1605.tdms" -> returns datetime(2025, 1, 27)
    """
    from datetime import datetime

    if isinstance(filename, Path):
        filename = str(filename)

    # Try YYYY-MM-DD format first
    try:
        return datetime.strptime(filename, "%Y-%m-%d")
    except:  # noqa: E722
        pass

    # Try to extract from Overview filename: YYMMDD-HHMM
    import re

    match = re.search(r"(\d{6})-(\d{4})", filename)
    if match:
        date_str = match.group(1)  # YYMMDD
        # Convert YY to YYYY (assuming 2000s)
        year = 2000 + int(date_str[0:2])
        month = int(date_str[2:4])
        day = int(date_str[4:6])
        return datetime(year, month, day)

    raise ValueError(f"Cannot parse date from: {filename}")


def find_pupitre_files(
    date: datetime,
    site: str,
    pupitre_datadir: str | Path,
    hours: range | list[int] | None = None,
) -> list[str]:
    """
    Find pupitre files for a given date and site.

    Parameters
    ----------
    date : datetime
        Date to search for
    site : str
        Site name (M9, M10, etc.)
    pupitre_datadir : Path
        Base directory for pupitre data
    hours : range, list of int, or None
        If provided, only return files whose start hour is included.
        Accepts a list (e.g. [10, 11, 12]) or a range (e.g. range(10, 13)).

    Returns
    -------
    list
        List of matching pupitre file paths
    """
    pupitre_datadir = Path(pupitre_datadir)
    site_dir = pupitre_datadir / site

    # Format: 2025.01.27 - 15:39:29.txt
    date_pattern = f"{date.year}.{date.month:02d}.{date.day:02d}*.txt"
    pattern = str(site_dir / date_pattern)

    files = sorted(glob.glob(pattern))
    if hours is None:
        return files

    filtered = []
    for f in files:
        t0 = t0_from_filename(f)
        if int(t0 // 3600) in hours:
            logger.debug("Included pupitre file: %s (t0=%s seconds)", f, t0)
            filtered.append(f)
    return filtered


def find_tdms_overview_files(
    date: datetime,
    site: str,
    pigbrother_datadir: str | Path,
    hours: range | list[int] | None = None,
) -> list[str]:
    """
    Find Overview TDMS files for a given date and site.

    Parameters
    ----------
    date : datetime
        Date to search for
    site : str
        Site name (M9, M10, etc.)
    pigbrother_datadir : Path
        Base directory for pigbrother data
    hours : range, list of int, or None
        If provided, only return files whose start hour is included.
        Accepts a list (e.g. [10, 11, 12]) or a range (e.g. range(10, 13)).

    Returns
    -------
    list
        List of matching Overview TDMS file paths
    """
    pigbrother_datadir = Path(pigbrother_datadir)
    overview_dir = pigbrother_datadir / site / "Overview"

    # Format: M9_Overview_250127-*.tdms (YY = year - 2000)
    year_yy = date.year % 100
    date_pattern = f"{site}_Overview_{year_yy:02d}{date.month:02d}{date.day:02d}-*.tdms"
    pattern = str(overview_dir / date_pattern)

    files = sorted(glob.glob(pattern))
    if hours is None:
        return files

    filtered = []
    for f in files:
        t0 = t0_from_tdms_filename(f)
        if int(t0 // 3600) in hours:
            logger.debug("Included TDMS file: %s (t0=%s seconds)", f, t0)
            filtered.append(f)
    return filtered


def t0_from_filename(filename: str) -> float:
    """
    Extract start time as seconds from midnight from a pupitre filename.

    Expected stem format: 'YYYY.MM.DD - HH:MM:SS'
    Example: '2025.11.05 - 05:53:00.txt'

    Returns
    -------
    float
        Seconds elapsed since midnight of the recording day.
    """
    stem = Path(filename).stem
    logger.debug("Parsing t0 from filename: %s, stem: %s", filename, stem)
    try:
        dt = datetime.strptime(stem, "%Y.%m.%d - %H:%M:%S")
        logger.debug(
            "hours: %d, minutes: %d, seconds: %d", dt.hour, dt.minute, dt.second
        )
        return dt.hour * 3600 + dt.minute * 60 + dt.second
    except ValueError:
        logger.warning(
            "t0_from_filename: could not parse t0 from filename '%s', using 0", stem
        )
        return 0.0


def t0_from_tdms_filename(filename: str) -> float:
    """
    Extract start time as seconds from midnight from a TDMS overview filename.

    Expected stem format: '<site>_Overview_YYMMDD-HHMM'
    Example: 'M8_Overview_251105-0949.tdms'

    Returns
    -------
    float
        Seconds elapsed since midnight of the recording day.
    """
    stem = Path(filename).stem
    logger.debug(
        "t0_from_tdms_filename: Parsing t0 from TDMS filename: %s, stem: %s",
        filename,
        stem,
    )
    try:
        date_part = stem.rsplit("_", 1)[-1]  # '251105-0949'
        dt = datetime.strptime(date_part, "%y%m%d-%H%M")
        return dt.hour * 3600 + dt.minute * 60
    except ValueError:
        logger.warning(
            "t0_from_tdms_filename: could not parse t0 from TDMS filename '%s', using 0",
            stem,
        )
        return 0.0


def load_pupitre_data(
    pupitre_file: str | Path, site: str, insert: str = "Unknown"
) -> MagnetRun:
    """
    Load pupitre data from a text file.

    Parameters
    ----------
    pupitre_file : str or Path
        Path to pupitre txt file
    site : str
        Site name (M9, M10, etc.)
    insert : str
        Insert name

    Returns
    -------
    MagnetRun
        MagnetRun object containing the pupitre data
    """
    logger.info(f"Loading pupitre data from: {pupitre_file}")
    return MagnetRun.fromtxt(site, insert, str(pupitre_file))


def load_tdms_data(
    tdms_file: str | Path, site: str, insert: str = "Unknown"
) -> MagnetRun:
    """
    Load TDMS data from a pigbrother Overview file.

    Parameters
    ----------
    tdms_file : str or Path
        Path to TDMS overview file
    site : str
        Site name (M9, M10, etc.)
    insert : str
        Insert name

    Returns
    -------
    MagnetRun
        MagnetRun object containing the TDMS data
    """
    logger.info(f"Loading TDMS data from: {tdms_file}")
    return MagnetRun.fromtdms(site, insert, str(tdms_file))



def plot_comparison(
    hybrid_data: HybridRun,
    pupitre_data: list[MagnetRun],
    tdms_data: list[MagnetRun],
    hybrid_key: str,
    site: str,
    hours: range | list[int] | None = None,
    normalize: bool = False,
) -> tuple[Figure, Axes]:
    """
    Plot hybrid, pupitre, and TDMS data on the same graph.

    Parameters
    ----------
    hybrid_data : HybridRun
        Hybrid run data.
    pupitre_data : list of MagnetRun
        Pupitre data list (can be empty).
    tdms_data : list of MagnetRun
        TDMS data list (can be empty).
    hybrid_key : str
        Key for hybrid data (e.g., 'kHz/FEPC-AUX-LNCMI/ALIM1_J1').
    site : str
        Site name for field mapping.
    hours : range, list of int, or None, optional
        Hours to restrict the plot to.
    normalize : bool, optional
        If True, normalize each signal by its maximum absolute value before plotting.

    Returns
    -------
    tuple[Figure, Axes]
        The matplotlib Figure and Axes objects.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Get mapped field names
    pupitre_field = HYBRID_TO_PUPITRE_MAP.get(hybrid_key)
    tdms_field = HYBRID_TO_TDMS_MAP.get(hybrid_key)

    # Get t start from hours if provided
    t0 = 0.0
    if hours is not None and len(hours) > 0:
        t0 = hours[0] * 3600  # Convert first hour to seconds
        logger.debug(f"Plotting data starting from hour {hours[0]} (t0={t0} seconds)")

    # Plot hybrid kHz data
    print(f"Loading hybrid data for key: {hybrid_key}")
    try:
        res = hybrid_key.split("/")
        logger.debug(f"hybrid_key parts: {res}")
        hybrid_type = res[0]
        hybrid_system = None
        if res and len(res) > 1:
            hybrid_system = res[1]
        logger.debug(f"Hybrid data keys:\n{hybrid_data.getKeys()}")
        # print(
        #     f"Hybrid data keys: system={hybrid_system}, data_type={hybrid_type}\n",
        #     json.dumps(
        #         hybrid_data.getKeys(),
        #         indent=2,
        #     ),
        #     flush=True,
        # )
        data, time = hybrid_data.getData(hybrid_key, downsample=10000, hours=hours)
        logger.debug(
            f"Hybrid data loaded: {len(data)} points, time range: {time[0]} to {time[-1]} seconds"
        )
        # Convert time to relative seconds if it's datetime
        if len(time) > 0:
            if hasattr(time[0], "timestamp"):
                time_seconds = np.array([(t - time[0]).total_seconds() for t in time])
            else:
                time_seconds = time
        else:
            time_seconds = time

        max_abs = np.max(np.abs(data))
        label = f"Hybrid kHz ({hybrid_key})"
        if normalize:
            label += f" [max={max_abs:.4g}]"
        ax.plot(
            time_seconds,
            normalize_signal(data) if normalize else data,
            "b-",
            alpha=0.7,
            linewidth=0.5,
            label=label,
        )

        # Add V for Bitters (BITTER_V1, BITTER_V2), V for Helices (from PH_V8 to PH_V14) if available
        if "Alim" in hybrid_key:
            for vkey in ["BITTER_V2", "PH_V8"]:
                data, time = hybrid_data.getData(
                    f"kHz/FEPC-AUX-LNCMI/{vkey}", downsample=10000, hours=hours
                )
                logger.debug(
                    "Hybrid data loaded: %d points, time range: %s to %s seconds",
                    len(data),
                    time[0],
                    time[-1],
                )
            # Convert time to relative seconds if it's datetime
            if len(time) > 0:
                if hasattr(time[0], "timestamp"):
                    time_seconds = np.array(
                        [(t - time[0]).total_seconds() for t in time]
                    )
                else:
                    time_seconds = time
            else:
                time_seconds = time

            max_abs = np.max(np.abs(data))
            label = f"Hybrid kHz (kHz/FEPC-AUX-LNCMI/{vkey})"
            if normalize:
                label += f" [max={max_abs:.4g}]"
            ax.plot(
                time_seconds,
                binarize_signal(data) if normalize else data,
                "y-",
                alpha=0.7,
                linewidth=0.5,
                label=label,
            )

    except (OSError, ValueError, RuntimeError, KeyError) as e:
        log_exception(
            "Warning: Could not load hybrid data",
            e,
            logger_instance=logger,
            include_traceback=False,
        )
        logger.debug("  Error at %s: %s", format_exception_location(), e)

    # Plot pupitre data if available
    if pupitre_data and pupitre_field:
        for i, pdata in enumerate(pupitre_data):
            try:
                mdata = pdata.getMData()
                if pupitre_field in mdata.getKeys():
                    pupitre_values = mdata.getData(pupitre_field)
                    pupitre_t0 = t0_from_filename(mdata.FileName)
                    logger.debug("Pupitre t0 from filename: %s seconds", pupitre_t0)
                    pupitre_time = mdata.getData("t") + (
                        pupitre_t0 - t0
                    )  # Shift to absolute seconds from midnight

                    prefix = "Pupitre" if i == 0 else f"Pupitre {i+1}"
                    label = f"{prefix} ({pupitre_field})"
                    if normalize:
                        label += f" [max={np.max(np.abs(pupitre_values)):.4g}]"
                    ax.plot(
                        pupitre_time,
                        (
                            normalize_signal(pupitre_values)
                            if normalize
                            else pupitre_values
                        ),
                        "r-",
                        alpha=0.8,
                        linewidth=1.5,
                        label=label,
                    )
                else:
                    logger.warning("Pupitre field '%s' not found", pupitre_field)
            except (OSError, ValueError, RuntimeError, KeyError) as e:
                log_exception(
                    "Warning: Could not plot pupitre data",
                    e,
                    logger_instance=logger,
                    include_traceback=False,
                )
                logger.debug("  Error at %s: %s", format_exception_location(), e)

    # Plot TDMS data if available
    if tdms_data and tdms_field:
        for i, tdata in enumerate(tdms_data):
            try:
                mdata = tdata.getMData()
                # Get the appropriate group (usually 'Courants_Alimentations')
                tdms_keys = mdata.getKeys()
                if tdms_field in tdms_keys:
                    tdms_values = mdata.getData(tdms_field)

                    # TDMS typically uses 't' for time
                    tdms_t0 = t0_from_tdms_filename(mdata.FileName)
                    logger.debug("TDMS t0 from filename: %s seconds", tdms_t0)
                    logger.debug("TDMS keys: %s", mdata.getKeys())
                    mdata.addTdmsTime(tdms_field.split("/")[0])
                    logger.debug("TDMS keys after addTdmsTime: %s", mdata.getKeys())
                    group = tdms_field.split("/")[0]
                    tdms_time = mdata.getData(f"{group}/t") + (tdms_t0 - t0)
                    prefix = "TDMS" if i == 0 else f"TDMS {i+1}"
                    label = f"{prefix} ({tdms_field})"
                    if normalize:
                        label += f" [max={np.max(np.abs(tdms_values)):.4g}]"
                    ax.plot(
                        tdms_time,
                        normalize_signal(tdms_values) if normalize else tdms_values,
                        "g-",
                        alpha=0.8,
                        linewidth=1.5,
                        label=label,
                    )
                else:
                    logger.warning(
                        "TDMS field '%s' not found. Available: %s",
                        tdms_field,
                        tdms_keys[:10],
                    )
            except (OSError, ValueError, RuntimeError, KeyError) as e:
                log_exception(
                    "Warning: Could not plot TDMS data",
                    e,
                    logger_instance=logger,
                    include_traceback=False,
                )
                logger.debug("  Error at %s: %s", format_exception_location(), e)

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Normalized value (a.u.)" if normalize else "Value")
    ax.set_title(f"Comparison: {hybrid_key} - Site {site}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax


# =============================================================================
# Main script
# =============================================================================


def main() -> int:
    """Entry point: parse arguments, load data, and generate the comparison plot."""
    datadir_parser = create_datadir_parser()
    parser = argparse.ArgumentParser(
        description="Plot hybrid kHz data with corresponding pupitre and TDMS data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[datadir_parser],
        epilog="""
Examples:
  # Plot FEPC-AUX-LNCMI data for a specific date
  python %(prog)s -d 2025-01-27 -s FEPC-AUX-LNCMI -k ALIM1_J1 --site Hybrid

  # Specify custom data directories
  python %(prog)s -d 2025-01-27 -s FEPC-LNCMI -k I_H1 --site Hybrid \\
      --hybrid-dir /path/to/hybrid/data \\
      --pupitre_datadir /path/to/pupitre \\
      --pigbrother_datadir /path/to/pigbrother

  # Plot only specific hours (comma-separated list)
  python %(prog)s -d 2025-01-27 -s FEPC-AUX-LNCMI -k ALIM1_J1 --site Hybrid --hours 10,11,12

  # Plot a range of hours (colon notation: start:stop, stop excluded)
  python %(prog)s -d 2025-01-27 -s FEPC-AUX-LNCMI -k ALIM1_J1 --site Hybrid --hours 10:13
        """,
    )

    parser.add_argument(
        "-d",
        "--date",
        required=True,
        help="Date in YYYY-MM-DD format (e.g., 2025-01-27)",
    )

    parser.add_argument(
        "-s",
        "--fepc-system",
        required=True,
        choices=["FEPC-LNCMI", "FEPC-AUX-LNCMI"],
        help="FEPC system to use",
    )

    parser.add_argument(
        "-k",
        "--key",
        required=True,
        help="Variable name to plot (e.g., ALIM1_J1, I_H1)",
    )

    parser.add_argument(
        "--site", required=True, type=str, help="site -- aka assembly magnet name"
    )

    parser.add_argument(
        "--hybrid-dir",
        type=Path,
        default=Path(HYBRID_DATA_DIR),
        help=f"Base directory for hybrid data (overrides MAGNETRUN_HYBRID_DATA_DIR, default: {HYBRID_DATA_DIR})",
    )

    parser.add_argument(
        "--hours",
        type=str,
        help=(
            "Hours to select. Either a comma-separated list (e.g. '10,11,12') "
            "or a range using colon notation (e.g. '10:13' means hours 10, 11, 12)."
        ),
    )

    parser.add_argument(
        "--insert", default="Unknown", help="Insert name (default: Unknown)"
    )

    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize each signal by its maximum absolute value before plotting",
    )

    parser.add_argument("--show", action="store_true", help="Show plot interactively")

    parser.add_argument("--save", type=Path, help="Save plot to file")

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )
    parser.add_argument("--log-file", type=Path, help="Write logs to file")

    args = parser.parse_args()

    setup_logging(
        level=args.log_level,
        log_file=args.log_file if args.log_file else None,
    )

    logger.debug("args: %s", args)
    housing = "M8"

    # Parse date
    date = datetime.strptime(args.date, "%Y-%m-%d")
    print(f"Date: {date.strftime('%Y-%m-%d')}")
    print(f"Housing: {housing}")
    print(f"Site: {args.site}")
    print(f"FEPC System: {args.fepc_system}")

    # Parse hours if provided (supports '10,11,12' or '10:13' range notation)
    hours = None
    if args.hours:
        if ":" in args.hours:
            parts = args.hours.split(":")
            hours = range(int(parts[0]), int(parts[1]))
        else:
            hours = [int(h.strip()) for h in args.hours.split(",")]
        logger.debug("Hours: %s", list(hours))

    # Construct hybrid key
    hybrid_key = f"kHz/{args.fepc_system}/{args.key}"
    logger.debug(f"Hybrid key: {hybrid_key}")

    # Load hybrid data
    print(f"Loading hybrid data from: {args.hybrid_dir}")
    try:
        hrun = HybridRun.fromdir(
            base_dir=str(args.hybrid_dir),
            date_str=args.date,
            fepc_system=args.fepc_system,
            site=args.site,
        )
        logger.info("Hybrid data loaded successfully")
        logger.debug(f"Available keys: {hrun.getKeys()[:10]}...")
        logger.debug(f"hrun:\n{hrun}")
        logger.debug(f"hrun type: {type(hrun)}")
        logger.debug(f"hrun.HybridData:\n{hrun.getData()}")
        # print(hrun.HybridData.getInfo())
    except (OSError, ValueError, RuntimeError) as e:
        log_exception(
            "Error loading hybrid data",
            e,
            logger_instance=logger,
            include_traceback=True,
        )
        return 1

    # Find and load pupitre data
    pupitre_data = []
    pupitre_files = find_pupitre_files(date, housing, args.pupitre_datadir, hours=hours)
    if pupitre_files:
        print(f"Found {len(pupitre_files)} pupitre file(s)")
        for pupitre_file in pupitre_files:
            try:
                pdata = load_pupitre_data(pupitre_file, args.site, args.insert)
                logger.debug(f"Pupitre keys: {pdata.getMData().getKeys()[:10]}...")
                pupitre_data.append(pdata)
            except (OSError, ValueError, RuntimeError) as e:
                log_exception(
                    "Warning: Could not load pupitre data",
                    e,
                    logger_instance=logger,
                    include_traceback=False,
                )
                logger.debug(f"  Error at {format_exception_location()}: {e}")
    else:
        logger.info(f"No pupitre files found for {date.strftime('%Y-%m-%d')}")

    # Find and load TDMS data
    tdms_data = []
    tdms_files = find_tdms_overview_files(
        date, housing, args.pigbrother_datadir, hours=hours
    )
    if tdms_files:
        print(f"Found {len(tdms_files)} TDMS Overview file(s)")
        for tdms_file in tdms_files:
            try:
                tdata = load_tdms_data(tdms_file, args.site, args.insert)
                logger.debug(f"TDMS keys: {tdata.getMData().getKeys()[:10]}...")
                tdms_data.append(tdata)
            except (OSError, ValueError, RuntimeError) as e:
                log_exception(
                    "Warning: Could not load TDMS data",
                    e,
                    logger_instance=logger,
                    include_traceback=False,
                )
                logger.debug(f"  Error at {format_exception_location()}: {e}")
    else:
        logger.info(f"No TDMS Overview files found for {date.strftime('%Y-%m-%d')}")

    # Plot comparison
    print("Generating comparison plot...")
    fig, _ = plot_comparison(
        hrun,
        pupitre_data,
        tdms_data,
        hybrid_key,
        args.site,
        hours=hours,
        normalize=args.normalize,
    )

    # Save or show plot
    if args.save:
        logger.info("Saving plot to: %s", args.save)
        fig.savefig(args.save, dpi=150, bbox_inches="tight")

    if args.show:
        logger.debug("Displaying plot...")
        plt.show()

    if not args.save and not args.show:
        logger.warning(
            "No output specified. Use --show to display or --save to save the plot."
        )

    logger.info("Done!")
    return 0


if __name__ == "__main__":
    exit(main())
