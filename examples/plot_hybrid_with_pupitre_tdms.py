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
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from python_magnetrun.analysis.args import args_to_downsample_config
from python_magnetrun.cli_args import create_base_parser, create_downsampling_parser
from python_magnetrun.data_dirs import HYBRID_DATA_DIR
from python_magnetrun.hybrid.hybrid_run import BinarizeConfig, HybridRun, LoadOptions
from python_magnetrun.hybrid.utils import (
    binarize_signal,
    local_hour_to_utc,
    normalize_signal,
)
from python_magnetrun.log_utils import (
    format_exception_location,
    get_logger,
    log_exception,
    setup_logging,
)
from python_magnetrun.magnetdata_tdms import TdmsMagnetData
from python_magnetrun.MagnetRun import MagnetRun, load_mrun
from python_magnetrun.utils.downsampling import DownsampleConfig
from python_magnetrun.utils.files import find_files, select_files
from python_magnetrun.utils.timestamps import align_to_common_time

logger = get_logger(__name__)

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


def load_pupitre_data(
    pupitre_file: str | Path, housing: str, insert: str = "Unknown"
) -> MagnetRun:
    """Load pupitre data from a text file."""
    logger.info(f"Loading pupitre data from: {pupitre_file}")
    return load_mrun(str(pupitre_file), housing=housing, site=insert)


def load_tdms_data(
    tdms_file: str | Path, housing: str, insert: str = "Unknown"
) -> MagnetRun:
    """Load TDMS data from a pigbrother Overview file."""
    logger.info(f"Loading TDMS data from: {tdms_file}")
    return load_mrun(str(tdms_file), housing=housing, site=insert)


def plot_comparison(
    hybrid_data: HybridRun,
    pupitre_data: list[MagnetRun],
    tdms_data: list[MagnetRun],
    hybrid_key: str,
    housing: str,
    hours: range | list[int] | None = None,
    normalize: bool = False,
    downsample: DownsampleConfig | None = None,
    binarize_config: BinarizeConfig | None = None,
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
    housing : str
        Housing name for field mapping (e.g., 'M9', 'M10').
    hours : range, list of int, or None, optional
        Hours to restrict the plot to.
    normalize : bool, optional
        If True, normalize each signal by its maximum absolute value before plotting.
    downsample : DownsampleConfig or None, optional
        Downsampling configuration applied to all data sources before plotting.
    binarize_config : BinarizeConfig or None, optional
        Parameters forwarded to :func:`~python_magnetrun.processing.signal.binarize_signal`
        when building the voltage mask.  ``None`` uses the function's defaults (Otsu).

    Returns
    -------
    tuple[Figure, Axes]
        The matplotlib Figure and Axes objects.
    """
    load_opts = LoadOptions(downsample=downsample, binarize_config=binarize_config)
    fig, ax = plt.subplots(figsize=(14, 6))

    # Get mapped field names
    pupitre_field = HYBRID_TO_PUPITRE_MAP.get(hybrid_key)
    tdms_field = HYBRID_TO_TDMS_MAP.get(hybrid_key)

    # Compute t_ref: anchor t=0 to hours[0]:00:00 UTC when hours are given,
    # otherwise to hybrid origin.  t_ref is also used for the axis label.
    hybrid_ts = pd.Timestamp(hybrid_data.get_time_range()[0])
    if hours is not None:
        t_ref = hybrid_ts.replace(hour=hours[0], minute=0, second=0, microsecond=0)
    else:
        t_ref = hybrid_ts

    # Per-source offsets (seconds) so that source_time + offset aligns with t_ref.
    offsets = align_to_common_time(
        [hybrid_data] + pupitre_data + tdms_data,
        reference=t_ref.to_pydatetime(),
    )

    # Plot hybrid kHz data
    logger.info(f"Loading hybrid data for key: {hybrid_key}")
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
        result = hybrid_data.getData(hybrid_key, hours=hours, options=load_opts)
        assert isinstance(result, tuple)
        data, time = result
        logger.info(
            f"Hybrid data loaded: {len(data)} points, time range: "
            f"{time[0] if len(time) > 0 else 'N/A'} to {time[-1] if len(time) > 0 else 'N/A'} seconds"
        )
        # Convert time to seconds relative to t_ref.
        if len(time) > 0:
            if hasattr(time[0], "timestamp"):
                time_seconds = np.array([(t - t_ref).total_seconds() for t in time])
            else:
                time_seconds = time

        else:
            time_seconds = time

        max_abs = np.nanmax(np.abs(data))
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
        if "ALIM" in hybrid_key:
            for vkey in ["BITTER_V2", "PH_V8"]:
                result = hybrid_data.getData(
                    f"kHz/FEPC-AUX-LNCMI/{vkey}", hours=hours, options=load_opts
                )
                assert isinstance(result, tuple)
                data, time = result
                logger.info(
                    f"Hybrid data loaded: {len(data)} points, time range: "
                    f"{time[0] if len(time) > 0 else 'N/A'} to {time[-1] if len(time) > 0 else 'N/A'} seconds"
                )
                # Align to the same t=0 as the primary hybrid trace.
                if len(time) > 0:
                    if hasattr(time[0], "timestamp"):
                        time_seconds = np.array([(t - t_ref).total_seconds() for t in time])
                    else:
                        time_seconds = time
                else:
                    time_seconds = time

                max_abs = np.nanmax(np.abs(data)) if len(data) > 0 else 0.0
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
        logger.debug(f"  Error at {format_exception_location()}: {e}")

    # Plot pupitre data if available
    if pupitre_data and pupitre_field:
        for i, pdata in enumerate(pupitre_data):
            try:
                mdata = pdata.getMData()
                if pupitre_field in mdata.getKeys():
                    t_offset = offsets[id(pdata)]
                    logger.info(
                        f"Pupitre timerange={pdata.get_time_range()}, offset={t_offset:.1f} s from hybrid origin"
                    )
                    df = mdata.getData(["t", pupitre_field], downsample=downsample)
                    pupitre_values = df[pupitre_field].to_numpy()
                    pupitre_time = df["t"].to_numpy() + t_offset

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
                    logger.warning(f"Pupitre field '{pupitre_field}' not found")
            except (OSError, ValueError, RuntimeError, KeyError) as e:
                log_exception(
                    "Warning: Could not plot pupitre data",
                    e,
                    logger_instance=logger,
                    include_traceback=False,
                )
                logger.debug(f"  Error at {format_exception_location()}: {e}")

    # Plot TDMS data if available
    if tdms_data and tdms_field:
        for i, tdata in enumerate(tdms_data):
            try:
                mdata = tdata.getMData()
                # Get the appropriate group (usually 'Courants_Alimentations')
                tdms_keys = mdata.getKeys()
                if tdms_field in tdms_keys:
                    t_offset = offsets[id(tdata)]
                    logger.info(
                        f"TDMS timerange={tdata.get_time_range()}, offset={t_offset:.1f} s from hybrid origin"
                    )
                    group = tdms_field.split("/")[0]
                    assert isinstance(mdata, TdmsMagnetData)
                    mdata.addTdmsTime(group)
                    df = mdata.getData(
                        [f"{group}/t", tdms_field], downsample=downsample
                    )
                    channel = tdms_field.split("/")[1]
                    tdms_values = df[channel].to_numpy()
                    tdms_time = df["t"].to_numpy() + t_offset
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
                        f"TDMS field '{tdms_field}' not found. Available: {tdms_keys[:10]}"
                    )
            except (OSError, ValueError, RuntimeError, KeyError) as e:
                log_exception(
                    "Warning: Could not plot TDMS data",
                    e,
                    logger_instance=logger,
                    include_traceback=False,
                )
                logger.debug(f"  Error at {format_exception_location()}: {e}")

    ax.set_xlabel(f"Time (s from {t_ref.strftime('%H:%M:%S UTC')})")
    ax.set_ylabel("Normalized value (a.u.)" if normalize else "Value")
    ax.set_title(f"Comparison: {hybrid_key}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, ax


# =============================================================================
# Main script
# =============================================================================


def main() -> int:
    """Entry point: parse arguments, load data, and generate the comparison plot."""
    base_parser = create_base_parser(add_input_file=False)
    downsample_parser = create_downsampling_parser()
    parser = argparse.ArgumentParser(
        description="Plot hybrid kHz data with corresponding pupitre and TDMS data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[base_parser, downsample_parser],
        epilog="""
Examples:
  # Plot FEPC-AUX-LNCMI data for a specific date
  python %(prog)s -d 2025-01-27 -s FEPC-AUX-LNCMI -k ALIM1_J1 --housing M8

  # Specify custom data directories
  python %(prog)s -d 2025-01-27 -s FEPC-LNCMI -k I_H1 --housing M8 \\
      --hybrid-dir /path/to/hybrid/data \\
      --pupitre_datadir /path/to/pupitre \\
      --pigbrother_datadir /path/to/pigbrother

  # Plot only specific hours (comma-separated list)
  python %(prog)s -d 2025-01-27 -s FEPC-AUX-LNCMI -k ALIM1_J1 --housing M8 --hours 10,11,12

  # Plot a range of hours (colon notation: start:stop, stop excluded)
  python %(prog)s -d 2025-01-27 -s FEPC-AUX-LNCMI -k ALIM1_J1 --housing M8 --hours 10:13

  # Downsample to 5000 points using stride (fast, no extra dependency)
  python %(prog)s -d 2025-01-27 -s FEPC-AUX-LNCMI -k ALIM1_J1 --housing M8 \\
      --downsample-method stride --downsample-params '{"n_out": 5000}'

  # Downsample using minmax_lttb (requires tsdownsample)
  python %(prog)s -d 2025-01-27 -s FEPC-AUX-LNCMI -k ALIM1_J1 --housing M8 \\
      --downsample-method minmax_lttb --downsample-params '{"n_out": 10000}'

  # Use fixed threshold for voltage mask binarization
  python %(prog)s -d 2025-01-27 -s FEPC-AUX-LNCMI -k ALIM1_J1 --housing M8 \\
      --binarize-method fixed --binarize-params '{"tolerance": 0.01}'

  # Use noise-floor method with a custom percentile
  python %(prog)s -d 2025-01-27 -s FEPC-AUX-LNCMI -k ALIM1_J1 --housing M8 \\
      --binarize-method noise --binarize-params '{"noise_percentile": 30.0}'
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
        "--normalize",
        action="store_true",
        help="Normalize each signal by its maximum absolute value before plotting",
    )

    parser.add_argument(
        "--binarize-method",
        choices=["otsu", "fixed", "noise"],
        default="otsu",
        metavar="METHOD",
        help=(
            "thresholding method for the voltage-mask binarization: "
            "otsu (default, automatic), fixed (use --binarize-params tolerance), "
            "noise (noise-floor estimate)."
        ),
    )
    parser.add_argument(
        "--binarize-params",
        type=str,
        default=None,
        metavar="JSON",
        help=(
            "JSON object of binarization parameters. "
            "Supported keys: tolerance (float, fixed only, default 0.005), "
            "n_bins (int, otsu only, default 256), "
            "normalize (bool, default true), "
            "noise_percentile (float, noise only, default 40.0). "
            "Example: '{\"tolerance\": 0.01}'"
        ),
    )

    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--save", type=Path, help="Save plot to file (disables interactive display)"
    )
    output_group.add_argument(
        "--show", action="store_true", help="Show plot interactively (default)"
    )

    parser.set_defaults(housing="M8")
    args = parser.parse_args()

    setup_logging(
        level=args.log_level,
        log_file=args.log_file if args.log_file else None,
    )

    logger.debug(f"args: {args}")
    housing = args.housing

    # Parse date
    date = datetime.strptime(args.date, "%Y-%m-%d")
    logger.info(f"Date: {date.strftime('%Y-%m-%d')}")
    logger.info(f"Housing: {housing}")
    logger.info(f"FEPC System: {args.fepc_system}")

    # Parse hours if provided (supports '10,11,12' or '10:13' range notation).
    # args.hours is French local time; convert to UTC for hybrid data filtering.
    hours = None
    hours_utc = None
    if args.hours:
        if ":" in args.hours:
            parts = args.hours.split(":")
            hours = list(range(int(parts[0]), int(parts[1])))
        else:
            hours = [int(h.strip()) for h in args.hours.split(",")]
        hours_utc = [local_hour_to_utc(h, args.date) for h in hours]
        logger.debug(f"Hours (local): {hours}  →  UTC: {hours_utc}")

    # Build downsampling config from CLI args (None when --downsample-method none)
    downsample_config = args_to_downsample_config(args)
    logger.info(f"Downsample config: {downsample_config}")

    # Build binarize config from CLI args
    binarize_params: dict = {}
    if args.binarize_params:
        try:
            binarize_params = json.loads(args.binarize_params)
        except json.JSONDecodeError as exc:
            parser.error(f"--binarize-params is not valid JSON: {exc}")
    binarize_config = BinarizeConfig(method=args.binarize_method, **binarize_params)
    logger.info(f"Binarize config: {binarize_config}")

    # Construct hybrid key
    hybrid_key = f"kHz/{args.fepc_system}/{args.key}"
    logger.debug(f"Hybrid key: {hybrid_key}")

    # Load hybrid data
    logger.info(f"Loading hybrid data from: {args.hybrid_dir}")
    try:
        hrun = HybridRun.fromdir(
            base_dir=str(args.hybrid_dir),
            date_str=args.date,
            fepc_system=args.fepc_system,
            housing=args.housing,
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

    # Build time range for select_files
    h_list = list(hours) if hours is not None else []
    if h_list:
        start_ts = f"{date.strftime('%Y-%m-%d')} {min(h_list):02d}:00:00"
        end_ts = f"{date.strftime('%Y-%m-%d')} {max(h_list) + 1:02d}:00:00"
    else:
        start_ts = f"{date.strftime('%Y-%m-%d')} 00:00:00"
        end_ts = f"{date.strftime('%Y-%m-%d')} 23:59:59"

    # Find TDMS Overview files: glob by date, then filter by time range
    year_yy = date.year % 100
    overview_dir = Path(args.pigbrother_datadir) / housing / "Overview"
    overview_pattern = str(
        overview_dir
        / f"{housing}_Overview_{year_yy:02d}{date.month:02d}{date.day:02d}-*.tdms"
    )
    tdms_files = select_files(
        sorted(glob.glob(overview_pattern)), housing, start_ts, end_ts
    )

    # Find pupitre files: derive glob pattern from first overview file via find_files,
    # falling back to a date-only pattern when no overview files exist.
    if tdms_files:
        stem = os.path.splitext(os.path.basename(tdms_files[0]))[0]
        parts = stem.split("_")
        date_part, time_part = parts[2].split("-")
        pupitre_f, *_ = find_files(
            tdms_files[0], housing, date_part, time_part, args.pupitre_datadir
        )
    else:
        pupitre_dir = Path(args.pupitre_datadir) / housing
        pupitre_f = str(
            pupitre_dir / f"{date.year}.{date.month:02d}.{date.day:02d}*.txt"
        )

    pupitre_files = select_files(
        sorted(glob.glob(pupitre_f)), housing, start_ts, end_ts
    )

    # Load pupitre data
    pupitre_data = []
    if pupitre_files:
        logger.info(f"Found {len(pupitre_files)} pupitre file(s)")
        for pupitre_file in pupitre_files:
            try:
                pdata = load_pupitre_data(pupitre_file, housing, args.insert)
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

    # Load TDMS data
    tdms_data = []
    if tdms_files:
        logger.info(f"Found {len(tdms_files)} TDMS Overview file(s)")
        for tdms_file in tdms_files:
            try:
                tdata = load_tdms_data(tdms_file, housing, args.insert)
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
    logger.info("Generating comparison plot...")
    fig, _ = plot_comparison(
        hrun,
        pupitre_data,
        tdms_data,
        hybrid_key,
        args.housing,
        hours=hours_utc,
        normalize=args.normalize,
        downsample=downsample_config,
        binarize_config=binarize_config,
    )

    # Save or show plot (show is the default when --save is not given)
    if args.save:
        logger.info(f"Saving plot to: {args.save}")
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
    else:
        logger.debug("Displaying plot...")
        plt.show()

    logger.info("Done!")
    return 0


if __name__ == "__main__":
    exit(main())
