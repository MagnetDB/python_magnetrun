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

# Map hybrid_run kHz field names to pupitre field names
HYBRID_TO_PUPITRE_MAP = {
    # FEPC-AUX-LNCMI channels
    "kHz/FEPC-AUX-LNCMI/ALIM1_J1": "IH",  # Example: Helix current
    "kHz/FEPC-AUX-LNCMI/ALIM1_J2": "IB",  # Example: Bottom coil current
    "kHz/FEPC-AUX-LNCMI/ALIM2_J1": "FlowH",  # Example mapping
    "kHz/FEPC-AUX-LNCMI/ALIM2_J2": "FlowB",  # Example mapping
    # FEPC-LNCMI channels (customize based on your setup)
    "kHz/FEPC-LNCMI/I_H1": "IH",
    "kHz/FEPC-LNCMI/I_H2": "IH",
    "kHz/FEPC-LNCMI/I_B1": "IB",
    "kHz/FEPC-LNCMI/I_B2": "IB",
}

# Map hybrid_run kHz field names to TDMS (pigbrother) field names
HYBRID_TO_TDMS_MAP = {
    # Map to Référence channels in Overview TDMS files
    "kHz/FEPC-AUX-LNCMI/ALIM1_J1": "Référence_GR1",
    "kHz/FEPC-AUX-LNCMI/ALIM1_J2": "Référence_GR2",
    "kHz/FEPC-LNCMI/I_H1": "Référence_GR1",
    "kHz/FEPC-LNCMI/I_H2": "Référence_GR1",
    "kHz/FEPC-LNCMI/I_B1": "Référence_GR2",
    "kHz/FEPC-LNCMI/I_B2": "Référence_GR2",
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


def find_pupitre_files(date, site, pupitre_datadir):
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

    Returns
    -------
    list
        List of matching pupitre file paths
    """
    pupitre_datadir = Path(pupitre_datadir)
    site_dir = pupitre_datadir / site

    # Format: M9/2025.01.27---15:39:29.txt
    date_pattern = f"{date.year}.{date.month:02d}.{date.day:02d}*.txt"
    pattern = str(site_dir / date_pattern)

    files = glob.glob(pattern)
    return sorted(files)


def find_tdms_overview_files(date, site, pigbrother_datadir):
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

    files = glob.glob(pattern)
    return sorted(files)


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


def plot_comparison(hybrid_data, pupitre_data, tdms_data, hybrid_key, site, hours=None):
    """
    Plot hybrid, pupitre, and TDMS data on the same graph.

    Parameters
    ----------
    hybrid_data : HybridRun
        Hybrid run data
    pupitre_data : MagnetRun or None
        Pupitre data (can be None)
    tdms_data : MagnetRun or None
        TDMS data (can be None)
    hybrid_key : str
        Key for hybrid data (e.g., 'kHz/FEPC-AUX-LNCMI/ALIM1_J1')
    site : str
        Site name for field mapping
    hours : list of int, optional
        Specific hours to plot
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Get mapped field names
    pupitre_field = HYBRID_TO_PUPITRE_MAP.get(hybrid_key)
    tdms_field = HYBRID_TO_TDMS_MAP.get(hybrid_key)

    # Plot hybrid kHz data
    print(f"Loading hybrid data for key: {hybrid_key}")
    try:
        data, time = hybrid_data.getData(hybrid_key, downsample=10000, hours=hours)

        # Convert time to relative seconds if it's datetime
        if len(time) > 0:
            if hasattr(time[0], "timestamp"):
                time_seconds = np.array([(t - time[0]).total_seconds() for t in time])
            else:
                time_seconds = time
        else:
            time_seconds = time

        ax.plot(
            time_seconds,
            data,
            "b-",
            alpha=0.7,
            linewidth=0.5,
            label=f"Hybrid kHz ({hybrid_key})",
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
        try:
            mdata = pupitre_data.getMData()
            if pupitre_field in mdata.getKeys():
                pupitre_values = mdata.getData(pupitre_field)
                pupitre_time = mdata.getData("t")  # Relative time in seconds
                ax.plot(
                    pupitre_time,
                    pupitre_values,
                    "r-",
                    alpha=0.8,
                    linewidth=1.5,
                    label=f"Pupitre ({pupitre_field})",
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
        try:
            mdata = tdms_data.getMData()
            # Get the appropriate group (usually 'Courants_Alimentations')
            tdms_keys = mdata.getKeys()
            if tdms_field in tdms_keys:
                tdms_values = mdata.getData(tdms_field)
                # TDMS typically uses 't' for time
                if "t" in tdms_keys:  # noqa: SIM108
                    tdms_time = mdata.getData("t")
                else:
                    tdms_time = np.arange(len(tdms_values))
                ax.plot(
                    tdms_time,
                    tdms_values,
                    "g-",
                    alpha=0.8,
                    linewidth=1.5,
                    label=f"TDMS ({tdms_field})",
                )
            else:
                print(f"Warning: TDMS field '{tdms_field}' not found. Available: {tdms_keys[:10]}")
        except (OSError, ValueError, RuntimeError, KeyError) as e:
            log_exception(
                "Warning: Could not plot TDMS data",
                e,
                use_print=True,
                include_traceback=False,
            )
            print(f"  Error at {format_exception_location()}: {e}")

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Value")
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
  python %(prog)s -d 2025-01-27 -s FEPC-AUX-LNCMI -k ALIM1_J1 --site M10

  # Specify custom data directories
  python %(prog)s -d 2025-01-27 -s FEPC-LNCMI -k I_H1 --site M9 \\
      --hybrid-dir /path/to/hybrid/data \\
      --pupitre-dir /path/to/pupitre \\
      --pigbrother-dir /path/to/pigbrother

  # Plot only specific hours
  python %(prog)s -d 2025-01-27 -s FEPC-AUX-LNCMI -k ALIM1_J1 --site M10 --hours 10,11,12
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
        "--site", required=True, choices=["M8", "M9", "M10"], help="Measurement site"
    )

    parser.add_argument(
        "--hybrid-dir",
        type=Path,
        default=Path(__file__).parent.parent / "hybrid_data",
        help="Base directory for hybrid data (default: ../hybrid_data)",
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
        help="Comma-separated list of hours to plot (e.g., 0,1,2 or 10,11,12)",
    )

    parser.add_argument("--insert", default="Unknown", help="Insert name (default: Unknown)")

    parser.add_argument("--show", action="store_true", help="Show plot interactively")

    parser.add_argument("--save", type=Path, help="Save plot to file")

    args = parser.parse_args()

    # Parse date
    date = datetime.strptime(args.date, "%Y-%m-%d")
    print(f"Date: {date.strftime('%Y-%m-%d')}")
    print(f"Site: {args.site}")
    print(f"FEPC System: {args.fepc_system}")

    # Parse hours if provided
    hours = None
    if args.hours:
        hours = [int(h.strip()) for h in args.hours.split(",")]
        print(f"Hours: {hours}")

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
    except (OSError, ValueError, RuntimeError) as e:
        log_exception("Error loading hybrid data", e, use_print=True, include_traceback=True)
        return 1

    # Find and load pupitre data
    pupitre_data = None
    pupitre_files = find_pupitre_files(date, args.site, args.pupitre_dir)
    if pupitre_files:
        print(f"\nFound {len(pupitre_files)} pupitre file(s)")
        # Load the first matching file
        try:
            pupitre_data = load_pupitre_data(pupitre_files[0], args.site, args.insert)
            print(f"Pupitre keys: {pupitre_data.getMData().getKeys()[:10]}...")
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
    tdms_data = None
    tdms_files = find_tdms_overview_files(date, args.site, args.pigbrother_dir)
    if tdms_files:
        print(f"\nFound {len(tdms_files)} TDMS Overview file(s)")
        # Load the first matching file
        try:
            tdms_data = load_tdms_data(tdms_files[0], args.site, args.insert)
            print(f"TDMS keys: {tdms_data.getMData().getKeys()[:10]}...")
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
    fig, ax = plot_comparison(hrun, pupitre_data, tdms_data, hybrid_key, args.site, hours=hours)

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
