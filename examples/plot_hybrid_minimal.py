#!/usr/bin/env python3
"""
Minimal example: Load and plot hybrid kHz data with pupitre and TDMS data.

This is a simplified version demonstrating the core concepts.
For a full-featured version, see plot_hybrid_with_pupitre_tdms.py
"""

from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import glob

# Import from python_magnetrun
from python_magnetrun.MagnetRun import MagnetRun

# Import from hybrid module
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "hybrid"))
from hybrid.hybrid_run import HybridRun
from hybrid.utils import log_exception, format_exception_location


# =============================================================================
# Configuration
# =============================================================================

# Data directories (adjust these to your setup)
HYBRID_BASE_DIR = "/path/to/hybrid_data"  # Base dir containing kHz/, rms/, trigger/
PUPITRE_DATADIR = "/home/LNCMI-G/christophe.trophime/LNCMIG-Data/srv-data-install"
PIGBROTHER_DATADIR = "/home/LNCMI-G/christophe.trophime/github/python_magnetrun/pigbrotherdata/Fichiers_Data"

# Field name mappings: hybrid kHz -> pupitre -> TDMS
# Customize based on your channel configuration
FIELD_MAPPING = {
    # Format: hybrid_key -> (pupitre_field, tdms_field)
    "kHz/FEPC-AUX-LNCMI/ALIM1_J1": ("IH", "Référence_GR1"),
    "kHz/FEPC-AUX-LNCMI/ALIM1_J2": ("IB", "Référence_GR2"),
    "kHz/FEPC-LNCMI/I_H1": ("IH", "Référence_GR1"),
    "kHz/FEPC-LNCMI/I_B1": ("IB", "Référence_GR2"),
}


# =============================================================================
# Example usage
# =============================================================================


def main():
    # =========================================================================
    # 1. Load hybrid kHz data
    # =========================================================================
    date_str = "2025-01-27"
    fepc_system = "FEPC-AUX-LNCMI"  # or "FEPC-LNCMI"
    site = "M10"

    print(f"Loading hybrid data for {date_str}, {fepc_system}, site {site}")

    hrun = HybridRun.fromdir(
        base_dir=HYBRID_BASE_DIR, date_str=date_str, fepc_system=fepc_system, site=site
    )

    # Get available keys
    print(f"Available hybrid keys (first 10): {hrun.getKeys()[:10]}")

    # =========================================================================
    # 2. Find and load pupitre data for the same date
    # =========================================================================
    date = datetime.strptime(date_str, "%Y-%m-%d")

    # Search pattern for pupitre files: M10/2025.01.27---*.txt
    pupitre_pattern = (
        f"{PUPITRE_DATADIR}/{site}/{date.year}.{date.month:02d}.{date.day:02d}*.txt"
    )
    pupitre_files = glob.glob(pupitre_pattern)

    pupitre_data = None
    if pupitre_files:
        print(f"Found pupitre file: {pupitre_files[0]}")
        pupitre_data = MagnetRun.fromtxt(site, "Unknown", pupitre_files[0])
        print(f"Pupitre keys: {pupitre_data.getMData().getKeys()[:10]}")
    else:
        print(f"No pupitre files found matching: {pupitre_pattern}")

    # =========================================================================
    # 3. Find and load TDMS data for the same date
    # =========================================================================

    # Search pattern for TDMS: M10/Overview/M10_Overview_250127-*.tdms
    year_yy = date.year % 100
    tdms_pattern = f"{PIGBROTHER_DATADIR}/{site}/Overview/{site}_Overview_{year_yy:02d}{date.month:02d}{date.day:02d}-*.tdms"
    tdms_files = glob.glob(tdms_pattern)

    tdms_data = None
    if tdms_files:
        print(f"Found TDMS file: {tdms_files[0]}")
        tdms_data = MagnetRun.fromtdms(site, "Unknown", tdms_files[0])
        print(f"TDMS keys: {tdms_data.getMData().getKeys()[:10]}")
    else:
        print(f"No TDMS files found matching: {tdms_pattern}")

    # =========================================================================
    # 4. Plot data from all three sources using field mapping
    # =========================================================================

    # Choose a channel to plot
    hybrid_key = "kHz/FEPC-AUX-LNCMI/ALIM1_J1"

    # Get corresponding fields from mapping dictionary
    if hybrid_key in FIELD_MAPPING:
        pupitre_field, tdms_field = FIELD_MAPPING[hybrid_key]
    else:
        print(f"Warning: No mapping defined for {hybrid_key}")
        return

    print(f"\nPlotting comparison:")
    print(f"  Hybrid: {hybrid_key}")
    print(f"  Pupitre: {pupitre_field}")
    print(f"  TDMS: {tdms_field}")

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot hybrid data (high frequency, downsampled for visualization)
    try:
        data, time = hrun.getData(hybrid_key, downsample=10000, hours=[10, 11, 12])
        # Convert to seconds if needed
        if len(time) > 0 and hasattr(time[0], "timestamp"):
            time_sec = np.array([(t - time[0]).total_seconds() for t in time])
        else:
            time_sec = time
        ax.plot(
            time_sec,
            data,
            "b-",
            alpha=0.7,
            linewidth=0.5,
            label=f"Hybrid kHz ({hybrid_key})",
        )
    except Exception as e:
        log_exception(
            "Could not plot hybrid data", e, use_print=True, include_traceback=True
        )

    # Plot pupitre data
    if pupitre_data and pupitre_field:
        try:
            mdata = pupitre_data.getMData()
            if pupitre_field in mdata.getKeys():
                values = mdata.getData(pupitre_field)
                time_p = mdata.getData("t")  # Time in seconds
                ax.plot(
                    time_p,
                    values,
                    "r-",
                    alpha=0.8,
                    linewidth=1.5,
                    label=f"Pupitre ({pupitre_field})",
                )
            else:
                print(f"Pupitre field '{pupitre_field}' not found")
        except Exception as e:
            log_exception(
                "Could not plot pupitre data", e, use_print=True, include_traceback=True
            )

    # Plot TDMS data
    if tdms_data and tdms_field:
        try:
            mdata = tdms_data.getMData()
            if tdms_field in mdata.getKeys():
                values = mdata.getData(tdms_field)
                time_t = (
                    mdata.getData("t")
                    if "t" in mdata.getKeys()
                    else np.arange(len(values))
                )
                ax.plot(
                    time_t,
                    values,
                    "g-",
                    alpha=0.8,
                    linewidth=1.5,
                    label=f"TDMS ({tdms_field})",
                )
            else:
                print(f"TDMS field '{tdms_field}' not found")
        except Exception as e:
            log_exception(
                "Could not plot TDMS data", e, use_print=True, include_traceback=True
            )

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Value")
    ax.set_title(f"Comparison: {hybrid_key} vs Pupitre vs TDMS - Site {site}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("hybrid_comparison_minimal.png", dpi=150)
    print("\nPlot saved to: hybrid_comparison_minimal.png")

    # Uncomment to show interactively:
    # plt.show()


if __name__ == "__main__":
    main()
