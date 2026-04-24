#!/usr/bin/env python3
"""
Minimal example: Load and plot hybrid kHz data with pupitre and TDMS data.

This is a simplified version demonstrating the core concepts.
For a full-featured version, see plot_hybrid_with_pupitre_tdms.py
"""

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np

# Import file-finder helpers from the full example
from plot_hybrid_with_pupitre_tdms import find_pupitre_files, find_tdms_overview_files

# Import from python_magnetrun
from python_magnetrun.data_dirs import (
    HYBRID_DATA_DIR,
    PIGBROTHER_DATA_DIR,
    PUPITRE_DATA_DIR,
)

# Import from hybrid module
from python_magnetrun.hybrid.hybrid_run import HybridRun
from python_magnetrun.hybrid.utils import binarize_signal, log_exception
from python_magnetrun.log_utils import get_logger, setup_logging
from python_magnetrun.MagnetRun import MagnetRun

TZ_FRANCE = ZoneInfo("Europe/Paris")
logger = get_logger()

# =============================================================================
# Configuration — override via MAGNETRUN_* env vars or edit here
# =============================================================================

HYBRID_BASE_DIR = HYBRID_DATA_DIR  # Base dir containing kHz/, rms/, trigger/
PUPITRE_DATADIR = PUPITRE_DATA_DIR
PIGBROTHER_DATADIR = PIGBROTHER_DATA_DIR

# Field name mappings: hybrid kHz -> pupitre -> TDMS
# Customize based on your channel configuration
FIELD_MAPPING = {
    # Format: hybrid_key -> (pupitre_field, tdms_field)
    "kHz/FEPC-AUX-LNCMI/ALIM1_J1": ("Idcct1", "Courants_Alimentations/Courant_A1"),
    "kHz/FEPC-AUX-LNCMI/ALIM1_J2": ("Idcct2", "Courants_Alimentations/Courant_A2"),
}

VOLTAGE_MAPPING = {
    # DEPRECATED: masking is now applied transparently inside HybridRun.getData()
    # when housing="M8" is passed to HybridRun.fromdir() and
    # hybrid_voltage_mask_map is defined in M8-housing-config.json.
    # This dict is kept here only for the explicit visualisation of mask signals
    # below (voltage overlay + binarized mask plots). Remove once those plots
    # are no longer needed.
    "kHz/FEPC-AUX-LNCMI/ALIM1_J1": "kHz/FEPC-AUX-LNCMI/BITTER_V2",
}
# NOTE (Hybrid data): When loading a current channel like ALIM1_J1, the raw
# signal includes noise from the power supply even when the magnet is off.
# To isolate the actual current, you MUST also load the corresponding
# Bitter voltage key (e.g. BITTER_V1 or BITTER_V2) and derive a binary mask
# from it (via binarize_signal). The current data is then element-wise
# multiplied by this mask so that off-state samples are zeroed out.
# See the VOLTAGE_MAPPING dict above and the masking block in main().
# =============================================================================
# Example usage
# =============================================================================


def main():
    setup_logging()  # logging.DEBUG)

    # =========================================================================
    # 1. Load hybrid kHz data
    # =========================================================================
    date_str = "2025-11-05"
    fepc_system = "FEPC-AUX-LNCMI"  # or "FEPC-LNCMI"
    housing = "M8"
    save = False
    hours = [13, 14, 15, 16, 17]  # Hours to load (optional)

    if hours is None:
        t0 = (
            datetime.strptime(date_str, "%Y-%m-%d")  # .replace(tzinfo=TZ_FRANCE)
            # .astimezone(timezone.utc)
        )
    else:
        t0 = (
            datetime.strptime(
                f"{date_str} {hours[0]:02d}:00", "%Y-%m-%d %H:%M"
            )  # .replace(tzinfo=TZ_FRANCE)
            # .astimezone(timezone.utc)
        )

    print(f"Loading hybrid data for {date_str}, {fepc_system}, housing {housing}")

    hrun = HybridRun.fromdir(
        base_dir=HYBRID_BASE_DIR,
        date_str=date_str,
        fepc_system=fepc_system,
        housing=housing,
    )

    # Get available keys
    print(f"Available hybrid keys (first 10): {hrun.getKeys()[:10]}")

    # =========================================================================
    # 2. Find and load pupitre data for the same date
    # =========================================================================
    date = datetime.strptime(date_str, "%Y-%m-%d")

    pupitre_files = find_pupitre_files(date, housing, PUPITRE_DATADIR, hours=hours)
    print(f"Found {len(pupitre_files)} pupitre file(s): {pupitre_files}")

    pupitre_data = None
    if pupitre_files:
        pupitre_data = MagnetRun.fromtxt(housing, "Unknown", pupitre_files[0])
        print(f"Pupitre keys: {pupitre_data.getMData().getKeys()[:10]}")
    else:
        print("No pupitre files found")

    # =========================================================================
    # 3. Find and load TDMS data for the same date
    # =========================================================================

    tdms_files = find_tdms_overview_files(
        date, housing, PIGBROTHER_DATADIR, hours=hours
    )
    print(f"Found {len(tdms_files)} TDMS Overview file(s): {tdms_files}")

    tdms_data = None
    if tdms_files:
        tdms_data = MagnetRun.fromtdms(housing, "Unknown", tdms_files[0])
        print(f"TDMS keys: {tdms_data.getMData().getKeys()[:10]}")
    else:
        print("No TDMS Overview files found")

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

    print("\nPlotting comparison:")
    print(f"  Hybrid: {hybrid_key}")
    print(f"  Pupitre: {pupitre_field}")
    print(f"  TDMS: {tdms_field}", flush=True)

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 6))

    from python_magnetrun.hybrid.utils import normalize_signal

    # Plot hybrid data (high frequency, downsampled for visualization)
    try:
        data, time = hrun.getData(hybrid_key, downsample=100000, hours=hours)
        if hybrid_key in VOLTAGE_MAPPING:
            voltage_key = VOLTAGE_MAPPING[hybrid_key]
            print(f"Loading voltage data for {voltage_key} to create mask")
            voltage_data, _ = hrun.getData(voltage_key, downsample=100000, hours=hours)
            ax.plot(
                time,
                normalize_signal(voltage_data),
                "c-",
                alpha=0.5,
                linewidth=0.5,
                label=f"Hybrid voltage mask ({voltage_key})",
            )

            # Apply mask to hybrid signal
            voltage_binary_mask = binarize_signal(voltage_data, method="fixed")
            ax.plot(
                time,
                voltage_binary_mask,
                "c-",
                alpha=0.5,
                linewidth=0.5,
                label=f"Hybrid voltage mask ({voltage_key}) -- fixed threshold",
            )
            data = data * voltage_binary_mask

            # Test other mask to hybrid signal
            voltage_binary_mask_otsu = binarize_signal(
                voltage_data, method="otsu", n_bins=2048
            )
            ax.plot(
                time,
                voltage_binary_mask_otsu,
                "y-",
                alpha=0.5,
                linewidth=0.5,
                label=f"Hybrid voltage mask ({voltage_key}) -- otsu",
            )

            voltage_binary_mask_noise = binarize_signal(voltage_data, method="noise")
            ax.plot(
                time,
                voltage_binary_mask_noise,
                "g-",
                alpha=0.5,
                linewidth=0.5,
                label=f"Hybrid voltage mask ({voltage_key}) -- noise",
            )

        ax.plot(
            time,
            normalize_signal(data),
            "b-",
            alpha=0.7,
            linewidth=0.5,
            label=f"Hybrid kHz ({hybrid_key})",
        )
    except (OSError, ValueError, RuntimeError, KeyError) as e:
        log_exception(
            "Could not plot hybrid data", e, use_print=True, include_traceback=True
        )

    # Plot pupitre data
    if pupitre_data and pupitre_field:
        try:
            mdata = pupitre_data.getMData()
            time_p = mdata.getStartDate()
            t0_p = mdata.start_timestamp  # is local not UTC
            t0_utc = t0.astimezone(UTC).replace(tzinfo=None)
            print(
                f"Pupitre start time: {time_p}, t0_p={t0_p} ({type(t0_p)}), t0={t0} ({type(t0)})",
                end="",
            )
            dt = (t0_utc - t0_p).total_seconds()
            print(f", dt={dt} seconds")

            if pupitre_field in mdata.getKeys():
                values = mdata.getData(pupitre_field)
                mdata.shiftTime(-dt)  # Shift to relative time in seconds
                time_p = mdata.getData("t")  # Time in seconds
                ax.plot(
                    time_p,
                    normalize_signal(np.asarray(values)),
                    "r-",
                    alpha=0.8,
                    linewidth=1.5,
                    label=f"Pupitre ({pupitre_field})",
                )
            else:
                print(f"Pupitre field '{pupitre_field}' not found")
        except (OSError, ValueError, RuntimeError, KeyError) as e:
            log_exception(
                "Could not plot pupitre data", e, use_print=True, include_traceback=True
            )

    # Plot TDMS data
    if tdms_data and tdms_field:
        try:
            mdata = tdms_data.getMData()
            time_p = mdata.getStartDate()
            t0_p = mdata.start_timestamp  # is UTC not local
            t0_utc = t0.astimezone(UTC).replace(tzinfo=None)
            dt = (t0_utc - t0_p).total_seconds()
            print(f"Pigbrother start time: {time_p}, t0_p={t0_p}, dt={dt} seconds")
            if tdms_field in mdata.getKeys():
                values = mdata.getData(tdms_field)
                group = tdms_field.split("/")[0]  # Extract group from field path
                time_t = mdata.getData(f"{group}/t")  # Time in seconds
                time_t = (
                    time_t - dt
                )  # Shift to relative time --shall be done using shiftTime that must be implemented in MagnetDataTDMS, but for simplicity we do it here

                ax.plot(
                    time_t,
                    normalize_signal(np.asarray(values)),
                    "g-",
                    alpha=0.8,
                    linewidth=1.5,
                    label=f"TDMS ({tdms_field})",
                )
            else:
                print(f"TDMS field '{tdms_field}' not found")
        except (OSError, ValueError, RuntimeError, KeyError) as e:
            log_exception(
                "Could not plot TDMS data", e, use_print=True, include_traceback=True
            )

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Normalized value (a.u.)")
    ax.set_title(f"Comparison: {hybrid_key} vs Pupitre vs TDMS")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save:
        fig.savefig("hybrid_comparison_minimal.png", dpi=150)
        print("\nPlot saved to: hybrid_comparison_minimal.png")
    else:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()
