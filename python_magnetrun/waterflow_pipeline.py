"""
Waterflow pipeline: extract hydraulic data from MagnetRun files
and fit pump curves using python_magnetcooling.

This module handles the data-extraction side. The actual curve fitting
is delegated to python_magnetcooling.fitting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from python_magnetcooling.waterflow import WaterFlow

from .log_utils import setup_logging

logger = logging.getLogger(__name__)


@dataclass
class HydraulicData:
    """
    Clean hydraulic arrays extracted from a MagnetRun DataFrame.

    All arrays are filtered (e.g., I >= threshold) and aligned.

    Attributes
    ----------
    current : np.ndarray
        Current [A].
    pump_speed : np.ndarray
        Pump speed [rpm].
    flow_rate : np.ndarray
        Flow rate [l/s].
    pressure : np.ndarray
        Inlet pressure [bar].
    back_pressure : np.ndarray
        Back (outlet) pressure [bar].
    imax : float or None
        Known Imax if available (from plateaus or prior knowledge).
    name : str
        Identifier (e.g., "M9_M10" or magnet name).
    """

    current: np.ndarray
    pump_speed: np.ndarray
    flow_rate: np.ndarray
    pressure: np.ndarray
    back_pressure: np.ndarray
    imax: float | None = None
    name: str = ""


def extract_hydraulic_data(
    df: pd.DataFrame,
    current_col: str,
    rpm_col: str,
    flow_col: str,
    pressure_in_col: str,
    pressure_out_col: str,
    current_threshold: float = 300.0,
    name: str = "",
) -> HydraulicData:
    """
    Extract and filter hydraulic arrays from a MagnetRun DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from MagnetRun (via getMData().getData() or similar).
    current_col : str
        Column name for current (e.g., "IH", "Icoil", site-dependent).
    rpm_col : str
        Column name for pump speed (e.g., "Rpm").
    flow_col : str
        Column name for flow rate (e.g., "Flow", "FlowH").
    pressure_in_col : str
        Column name for inlet pressure (e.g., "HP", "HPH").
    pressure_out_col : str
        Column name for back pressure (e.g., "BP", "BPH").
    current_threshold : float
        Minimum current for filtering (default 300 A).
    name : str
        Identifier for this dataset.

    Returns
    -------
    HydraulicData
        Filtered and aligned arrays ready for fitting.

    Raises
    ------
    KeyError
        If any column is missing from the DataFrame.
    ValueError
        If fewer than 3 data points remain after filtering.
    """
    required = [current_col, rpm_col, flow_col, pressure_in_col, pressure_out_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in DataFrame: {missing}")

    mask = df[current_col] >= current_threshold
    filtered = df.loc[mask]

    if len(filtered) < 3:
        raise ValueError(
            f"Only {len(filtered)} points after filtering "
            f"{current_col} >= {current_threshold}. Need at least 3."
        )

    logger.info(
        "Extracted %d points from %d total (threshold=%.0f A)",
        len(filtered),
        len(df),
        current_threshold,
    )

    return HydraulicData(
        current=filtered[current_col].to_numpy(),
        pump_speed=filtered[rpm_col].to_numpy(),
        flow_rate=filtered[flow_col].to_numpy(),
        pressure=filtered[pressure_in_col].to_numpy(),
        back_pressure=filtered[pressure_out_col].to_numpy(),
        name=name,
    )


def detect_imax_from_plateaus(
    data: HydraulicData,
    plateau_threshold: float = 0.01,
    window: int = 20,
    min_plateau_samples: int = 10,
) -> float | None:
    """
    Detect Imax by identifying the onset of a plateau in the pump speed curve.

    Uses a rolling derivative (slope of pump_speed over a sliding window of
    current values) to detect where pump speed stops increasing relative to
    current. A plateau is confirmed when the normalised slope stays below
    `plateau_threshold` for at least `min_plateau_samples` consecutive points.

    Parameters
    ----------
    data : HydraulicData
        Extracted hydraulic data. Arrays must be sorted by current (ascending).
    plateau_threshold : float
        Maximum normalised slope (``abs(d(Vp)/d(I)) / Vp_range``) considered flat.
        Default 0.01 means the slope must be < 1% of the total pump speed
        range per unit current window.
    window : int
        Number of samples over which to compute the rolling slope.
        Larger values give smoother derivatives but lag at plateau onsets.
        Default 20 samples.
    min_plateau_samples : int
        Minimum number of consecutive samples that must all be below
        `plateau_threshold` before a plateau is declared. Guards against
        false positives from momentary flat spots. Default 10.

    Returns
    -------
    float or None
        Current value [A] at the onset of the first confirmed plateau,
        or None if no plateau is detected.

    Notes
    -----
    The rolling slope is computed as::

        slope[i] = (pump_speed[i] - pump_speed[i - window])
                   / (current[i] - current[i - window])

    and then normalised by the total pump speed range so that the threshold
    is dimensionless and independent of the absolute rpm scale.

    If the data are unevenly sampled in current, the window is in *samples*
    not in amperes — consider resampling to a uniform current grid first
    if the sampling density varies significantly along the ramp.
    """
    pump_speed = pd.Series(data.pump_speed)
    current = pd.Series(data.current)

    speed_range = pump_speed.max() - pump_speed.min()
    if speed_range == 0.0:
        logger.warning("Pump speed is constant — cannot detect plateau onset.")
        return None

    current_range = current.max() - current.min()
    if current_range == 0.0:
        logger.warning("Current range is zero — cannot detect plateau onset.")
        return None

    # Rolling slope: delta(Vp) / delta(I) over `window` samples.
    # Using .diff(window) rather than .rolling().apply() is faster and
    # gives the same central-window-free result.
    delta_vp = pump_speed.diff(window)
    delta_i = current.diff(window)

    # Guard against zero current increments (constant-current segments)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where(delta_i != 0, delta_vp / delta_i, np.nan)

    # Normalise to a dimensionless quantity: multiply by current_range/speed_range
    # so that a perfect linear ramp gives ~1.0 and a true plateau gives ~0.
    normalised = pd.Series(np.abs(slope) * current_range / speed_range)

    # A point is "in plateau" when its normalised slope is below threshold
    in_plateau = normalised < plateau_threshold

    # Require min_plateau_samples consecutive plateau points to filter noise.
    # rolling(min_plateau_samples).min() == 1.0 means every sample in the
    # window was True.
    confirmed = (
        in_plateau.astype(float)
        .rolling(min_plateau_samples)
        .min()
        .fillna(0.0)
        .astype(bool)
    )

    if not confirmed.any():
        logger.info("No plateau detected in pump speed curve for '%s'.", data.name)
        return None

    # The onset of the plateau is min_plateau_samples before the first
    # confirmed index (roll window trails behind the actual onset).
    confirmed_idx = int(confirmed.idxmax())
    onset_idx = max(0, confirmed_idx - min_plateau_samples)
    imax = float(current.iloc[onset_idx])

    logger.info(
        "Plateau detected for '%s': onset at I=%.0f A (confirmed at sample %d).",
        data.name,
        imax,
        confirmed_idx,
    )
    return imax


def compute_waterflow(
    data: HydraulicData,
    method: str = "simple",
) -> WaterFlow:
    """
    Full pipeline: HydraulicData → fit → WaterFlow.

    Delegates curve fitting entirely to python_magnetcooling.fitting.

    Parameters
    ----------
    data : HydraulicData
        Cleaned hydraulic arrays (from extract_hydraulic_data).
    method : str
        "simple" for scipy quadratic fits (requires data.imax to be set).
        "piecewise" for pwlf-based fitting with automatic Imax detection.

    Returns
    -------
    WaterFlow
        Fully configured WaterFlow object from python_magnetcooling.

    Raises
    ------
    ImportError
        If python_magnetcooling is not installed.
    ValueError
        If method="simple" and data.imax is None.
    """
    try:
        from python_magnetcooling.fitting import (  # type: ignore[import]
            build_waterflow,
            fit_hydraulic_system,
        )
    except ImportError as exc:
        raise ImportError(
            "python_magnetcooling is required for hydraulic fitting. "
            "Install it with: pip install python-magnetcooling[fitting]"
        ) from exc

    if method == "simple" and data.imax is None:
        raise ValueError(
            "imax must be set in HydraulicData for method='simple'. "
            "Either set it manually, call detect_imax_from_plateaus(), "
            "or use method='piecewise' for automatic detection."
        )

    pump_fit, flow_pressure_fit = fit_hydraulic_system(
        current=data.current,
        pump_speed=data.pump_speed,
        flow_rate=data.flow_rate,
        pressure=data.pressure,
        back_pressure=data.back_pressure,
        imax=data.imax,
        method=method,
    )

    logger.info(
        "Fitted '%s': Imax=%.0f A, Vpmax=%.1f rpm, Fmax=%.1f l/s, Pmax=%.1f bar",
        data.name,
        pump_fit.imax,
        pump_fit.vpmax,
        flow_pressure_fit.fmax,
        flow_pressure_fit.pmax,
    )

    return build_waterflow(pump_fit, flow_pressure_fit)


def compute_waterflow_from_run(
    mrun,
    current_col: str,
    rpm_col: str,
    flow_col: str,
    pressure_in_col: str,
    pressure_out_col: str,
    method: str = "simple",
    imax: float | None = None,
    current_threshold: float = 300.0,
) -> WaterFlow:
    """
    End-to-end convenience function: MagnetRun → HydraulicData → WaterFlow.

    Parameters
    ----------
    mrun : MagnetRun
        A loaded MagnetRun object.
    current_col : str
        Column name for current in the MagnetRun data (site-dependent).
    rpm_col : str
        Column name for pump speed.
    flow_col : str
        Column name for flow rate.
    pressure_in_col : str
        Column name for inlet pressure.
    pressure_out_col : str
        Column name for back (outlet) pressure.
    method : str
        Fitting method: "simple" or "piecewise".
    imax : float, optional
        Known Imax [A]. If None and method="piecewise", auto-detected via
        detect_imax_from_plateaus(). If None and method="simple", raises
        ValueError.
    current_threshold : float
        Minimum current [A] for filtering (default 300 A).

    Returns
    -------
    WaterFlow
        Fully configured WaterFlow object from python_magnetcooling.

    Raises
    ------
    ImportError
        If python_magnetcooling is not installed.
    KeyError
        If required columns are absent from the MagnetRun DataFrame.
    ValueError
        If insufficient data remain after filtering, or if method="simple"
        and no imax can be determined.
    """
    df = mrun.getMData().getData()
    name = mrun.getInsert() if hasattr(mrun, "getInsert") else ""

    data = extract_hydraulic_data(
        df=df,
        current_col=current_col,
        rpm_col=rpm_col,
        flow_col=flow_col,
        pressure_in_col=pressure_in_col,
        pressure_out_col=pressure_out_col,
        current_threshold=current_threshold,
        name=name,
    )

    if imax is not None:
        data.imax = imax
    elif method == "simple":
        # Attempt plateau detection as a fallback before raising in
        # compute_waterflow(); surface it here with a more helpful message.
        detected = detect_imax_from_plateaus(data)
        if detected is None:
            raise ValueError(
                "method='simple' requires imax but none was provided and "
                "plateau detection found no clear saturation in the pump "
                "speed curve. Provide imax explicitly or use method='piecewise'."
            )
        data.imax = detected
        logger.info("Auto-detected Imax=%.0f A for '%s'.", detected, name)

    return compute_waterflow(data, method=method)


def main() -> None:
    """
    CLI entry point: fit a waterflow model from a MagnetRun file.

    Usage
    -----
    magnetrun-waterflow <run_file> --current-col IH --rpm-col Rpm
        --flow-col Flow --pin-col HP --pout-col BP --imax 28000 --output wf.json
    """
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Fit hydraulic pump curves from a MagnetRun file."
    )
    parser.add_argument("run_file", help="Path to the MagnetRun data file.")
    parser.add_argument("--current-col", default="IH", help="Current column name.")
    parser.add_argument("--rpm-col", default="Rpm", help="Pump speed column name.")
    parser.add_argument("--flow-col", default="Flow", help="Flow rate column name.")
    parser.add_argument("--pin-col", default="HP", help="Inlet pressure column name.")
    parser.add_argument("--pout-col", default="BP", help="Back pressure column name.")
    parser.add_argument("--imax", type=float, default=None, help="Known Imax [A].")
    parser.add_argument(
        "--method",
        choices=["simple", "piecewise"],
        default="simple",
        help="Fitting method.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=300.0,
        help="Minimum current threshold [A] for filtering.",
    )
    parser.add_argument(
        "--output", default="waterflow.json", help="Output JSON file path."
    )
    args = parser.parse_args()

    setup_logging()

    from python_magnetrun.MagnetRun import MagnetRun  # type: ignore[import]

    mrun = MagnetRun.fromFile(args.run_file)
    wf = compute_waterflow_from_run(
        mrun=mrun,
        current_col=args.current_col,
        rpm_col=args.rpm_col,
        flow_col=args.flow_col,
        pressure_in_col=args.pin_col,
        pressure_out_col=args.pout_col,
        method=args.method,
        imax=args.imax,
        current_threshold=args.threshold,
    )

    result = wf.to_dict() if hasattr(wf, "to_dict") else str(wf)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Waterflow parameters written to %s", args.output)


if __name__ == "__main__":
    main()
