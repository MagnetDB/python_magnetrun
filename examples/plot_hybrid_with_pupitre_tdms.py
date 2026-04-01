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
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from python_magnetrun.analysis.config import (
    DEFAULT_DATA_DIR,
    DEFAULT_HYBRID_DATA_DIR,
    DEFAULT_PIGBROTHER_DATA_DIR,
)

# Import from hybrid module
from python_magnetrun.hybrid.hybrid_run import HybridRun
from python_magnetrun.hybrid.utils import format_exception_location, log_exception

# Import from python_magnetrun
from python_magnetrun.MagnetRun import MagnetRun

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


def parse_date_from_filename(filename):
    """
    Parse date from hybrid data directory or Overview filename.

    Examples:
        - "2025-01-06" -> returns datetime
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
    date, site, pupitre_datadir, hours: range | list[int] | None = None
):
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
            print(f"Included pupitre file: {f} (t0={t0} seconds)")
            filtered.append(f)
    return filtered


def find_tdms_overview_files(
    date, site, pigbrother_datadir, hours: range | list[int] | None = None
):
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
            print(f"Included TDMS file: {f} (t0={t0} seconds)")
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
    print(f"Parsing t0 from filename: {filename}, stem: {stem}")
    try:
        dt = datetime.strptime(stem, "%Y.%m.%d - %H:%M:%S")
        print(f"hours: {dt.hour}, minutes: {dt.minute}, seconds: {dt.second}")
        return dt.hour * 3600 + dt.minute * 60 + dt.second
    except ValueError:
        print(
            f"t0_from_filename: Warning: could not parse t0 from filename '{stem}', using 0"
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
    print(
        f"t0_from_tdms_filename: Parsing t0 from TDMS filename: {filename}, stem: {stem}"
    )
    try:
        date_part = stem.rsplit("_", 1)[-1]  # '251105-0949'
        dt = datetime.strptime(date_part, "%y%m%d-%H%M")
        return dt.hour * 3600 + dt.minute * 60
    except ValueError:
        print(f"Warning: could not parse t0 from TDMS filename '{stem}', using 0")
        return 0.0


def load_pupitre_data(pupitre_file, site, insert="Unknown"):
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
    print(f"Loading pupitre data from: {pupitre_file}")
    return MagnetRun.fromtxt(site, insert, str(pupitre_file))


def load_tdms_data(tdms_file, site, insert="Unknown"):
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
    print(f"Loading TDMS data from: {tdms_file}")
    return MagnetRun.fromtdms(site, insert, str(tdms_file))


def normalize_signal(data):
    """Normalize a signal by its maximum absolute value. Returns data unchanged if max is 0."""
    max_abs = np.max(np.abs(data))
    if max_abs == 0:
        return data
    return data / max_abs


def plot_comparison(
    hybrid_data,
    pupitre_data,
    tdms_data,
    hybrid_key,
    site,
    hours=None,
    normalize=False,
):
    """
    Plot hybrid, pupitre, and TDMS data on the same graph.

    Parameters
    ----------
    hybrid_data : HybridRun
        Hybrid run data
    pupitre_data : list of MagnetRun
        Pupitre data list (can be empty)
    tdms_data : list of MagnetRun
        TDMS data list (can be empty)
    hybrid_key : str
        Key for hybrid data (e.g., 'kHz/FEPC-AUX-LNCMI/ALIM1_J1')
    site : str
        Site name for field mapping
    hours : list of int, optional
        Specific hours to plot
    normalize : bool, optional
        If True, normalize each signal by its maximum absolute value before plotting
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Get mapped field names
    pupitre_field = HYBRID_TO_PUPITRE_MAP.get(hybrid_key)
    tdms_field = HYBRID_TO_TDMS_MAP.get(hybrid_key)

    # Get t start from hours if provided
    t0 = 0.0
    if hours is not None and len(hours) > 0:
        t0 = hours[0] * 3600  # Convert first hour to seconds
        print(f"Plotting data starting from hour {hours[0]} (t0={t0} seconds)")

    # Plot hybrid kHz data
    print(f"Loading hybrid data for key: {hybrid_key}")
    try:
        data, time = hybrid_data.getData(hybrid_key, downsample=10000, hours=hours)
        print(
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
        for vkey in ["BITTER_V1", "BITTER_V2", "PH_V8"]:
            data, time = hybrid_data.getData(
                f"kHz/FEPC-AUX-LNCMI/{vkey}", downsample=10000, hours=hours
            )
            print(
                f"Hybrid data loaded: {len(data)} points, time range: {time[0]} to {time[-1]} seconds"
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
                normalize_signal(data) if normalize else data,
                "y-",
                alpha=0.7,
                linewidth=0.5,
                label=label,
            )

    except (OSError, ValueError, RuntimeError, KeyError) as e:
        log_exception(
            "Warning: Could not load hybrid data",
            e,
            use_print=True,
            include_traceback=False,
        )
        print(f"  Error at {format_exception_location()}: {e}")

    # Plot pupitre data if available
    if pupitre_data and pupitre_field:
        for i, pdata in enumerate(pupitre_data):
            try:
                mdata = pdata.getMData()
                if pupitre_field in mdata.getKeys():
                    pupitre_values = mdata.getData(pupitre_field)
                    pupitre_t0 = t0_from_filename(mdata.FileName)
                    print(f"Pupitre t0 from filename: {pupitre_t0} seconds")
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
                    print(f"Warning: Pupitre field '{pupitre_field}' not found")
            except (OSError, ValueError, RuntimeError, KeyError) as e:
                log_exception(
                    "Warning: Could not plot pupitre data",
                    e,
                    use_print=True,
                    include_traceback=False,
                )
                print(f"  Error at {format_exception_location()}: {e}")

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
                    print(f"TDMS t0 from filename: {tdms_t0} seconds")
                    print(mdata.getKeys())
                    mdata.addTdmsTime(tdms_field.split("/")[0])
                    print(mdata.getKeys())
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
                    print(
                        f"Warning: TDMS field '{tdms_field}' not found. Available: {tdms_keys[:10]}"
                    )
            except (OSError, ValueError, RuntimeError, KeyError) as e:
                log_exception(
                    "Warning: Could not plot TDMS data",
                    e,
                    use_print=True,
                    include_traceback=False,
                )
                print(f"  Error at {format_exception_location()}: {e}")

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


def main():
    parser = argparse.ArgumentParser(
        description="Plot hybrid kHz data with corresponding pupitre and TDMS data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot FEPC-AUX-LNCMI data for a specific date
  python %(prog)s -d 2025-01-27 -s FEPC-AUX-LNCMI -k ALIM1_J1 --site Hybrid

  # Specify custom data directories
  python %(prog)s -d 2025-01-27 -s FEPC-LNCMI -k I_H1 --site Hybrid \\
      --hybrid-dir /path/to/hybrid/data \\
      --pupitre-dir /path/to/pupitre \\
      --pigbrother-dir /path/to/pigbrother

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
        default=Path(DEFAULT_HYBRID_DATA_DIR),
        help=f"Base directory for hybrid data (default: {DEFAULT_HYBRID_DATA_DIR})",
    )

    parser.add_argument(
        "--pupitre-dir",
        type=Path,
        default=Path(DEFAULT_DATA_DIR),
        help=f"Base directory for pupitre data (default: {DEFAULT_DATA_DIR})",
    )

    parser.add_argument(
        "--pigbrother-dir",
        type=Path,
        default=Path(DEFAULT_PIGBROTHER_DATA_DIR),
        help=f"Base directory for pigbrother data (default: {DEFAULT_PIGBROTHER_DATA_DIR})",
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

    args = parser.parse_args()
    print(f"args: {args}")
    housing = "M8"

    # Parse date
    date = datetime.strptime(args.date, "%Y-%m-%d")
    print(f"Date: {date.strftime('%Y-%m-%d')}")
    print(f"housing: {housing}")
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
        print(f"Hours: {list(hours)}")

    # Construct hybrid key
    hybrid_key = f"kHz/{args.fepc_system}/{args.key}"
    print(f"Hybrid key: {hybrid_key}")

    # Load hybrid data
    print(f"\nLoading hybrid data from: {args.hybrid_dir}")
    try:
        hrun = HybridRun.fromdir(
            base_dir=str(args.hybrid_dir),
            date_str=args.date,
            fepc_system=args.fepc_system,
            site=args.site,
        )
        print("Hybrid data loaded successfully")
        print(f"Available keys: {hrun.getKeys()[:10]}...")  # Show first 10 keys
        print(f"hrun:\n {hrun}\n")  # Print hrun summary
        print(type(hrun))
        print(f"hrun.HybridData:\n {hrun.getData()}\n")  # Print hrun summary
        # hrun.print_summary
    except (OSError, ValueError, RuntimeError) as e:
        log_exception(
            "Error loading hybrid data", e, use_print=True, include_traceback=True
        )
        return 1

    # Find and load pupitre data
    pupitre_data = []
    pupitre_files = find_pupitre_files(date, housing, args.pupitre_dir, hours=hours)
    if pupitre_files:
        print(f"\nFound {len(pupitre_files)} pupitre file(s)")
        for pupitre_file in pupitre_files:
            try:
                pdata = load_pupitre_data(pupitre_file, args.site, args.insert)
                print(f"Pupitre keys: {pdata.getMData().getKeys()[:10]}...")
                pupitre_data.append(pdata)
            except (OSError, ValueError, RuntimeError) as e:
                log_exception(
                    "Warning: Could not load pupitre data",
                    e,
                    use_print=True,
                    include_traceback=False,
                )
                print(f"  Error at {format_exception_location()}: {e}")
    else:
        print(f"\nNo pupitre files found for {date.strftime('%Y-%m-%d')}")

    # Find and load TDMS data
    tdms_data = []
    tdms_files = find_tdms_overview_files(
        date, housing, args.pigbrother_dir, hours=hours
    )
    if tdms_files:
        print(f"\nFound {len(tdms_files)} TDMS Overview file(s)")
        for tdms_file in tdms_files:
            try:
                tdata = load_tdms_data(tdms_file, args.site, args.insert)
                print(f"TDMS keys: {tdata.getMData().getKeys()[:10]}...")
                tdms_data.append(tdata)
            except (OSError, ValueError, RuntimeError) as e:
                log_exception(
                    "Warning: Could not load TDMS data",
                    e,
                    use_print=True,
                    include_traceback=False,
                )
                print(f"  Error at {format_exception_location()}: {e}")
    else:
        print(f"\nNo TDMS Overview files found for {date.strftime('%Y-%m-%d')}")

    # Plot comparison
    print("\nGenerating comparison plot...")
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
        print(f"Saving plot to: {args.save}")
        fig.savefig(args.save, dpi=150, bbox_inches="tight")

    if args.show:
        print("Displaying plot...")
        plt.show()

    if not args.save and not args.show:
        print("No output specified. Use --show to display or --save to save the plot.")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
