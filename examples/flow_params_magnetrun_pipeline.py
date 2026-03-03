"""
Complete Flow Parameter Extraction Pipeline (using python_magnetrun)

This script demonstrates the full workflow using python_magnetrun methods:
1. Load/generate experimental data as pandas DataFrame
2. Use piecewise linear fitting (pwlf) for pump speed with automatic Imax detection
3. Use python_magnetrun.fit for flow rate and pressure
4. Build flow_params dictionary
5. Create WaterFlow object using waterflow_factory
6. Perform hydraulic calculations

This is a standalone version of the compute() method from flow_params_magnetrun.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
import json
import os
import sys

# Import python_magnetrun methods
from python_magnetrun.processing.fit import fit, find_eqn
from tabulate import tabulate
from sympy import Symbol
import pwlf

# Import the factory module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from python_magnetcooling.waterflow_factory import from_flow_params
from python_magnetcooling import WaterFlow


def generate_synthetic_data(
    n_points: int = 500, noise_level: float = 0.02, include_plateau: bool = True
) -> pd.DataFrame:
    """
    Generate synthetic experimental data as pandas DataFrame.

    Simulates data with potential plateau at high current (Imax detection).

    Args:
        n_points: Number of data points to generate
        noise_level: Relative noise level (0.02 = 2% noise)
        include_plateau: If True, includes plateau region above Imax

    Returns:
        DataFrame with columns: current, rpm, flow, pressure, back_pressure
    """
    print("=" * 70)
    print("STEP 1: Generate Experimental Data (as pandas DataFrame)")
    print("=" * 70)
    print("(Using synthetic data for demonstration)")

    # True parameters
    Imax = 28000  # A
    Vpmax = 2840  # rpm
    Vp0 = 1000  # rpm
    Fmax = 140  # l/s
    F0 = 0  # l/s
    Pmax = 22  # bar
    Pmin = 4  # bar

    if include_plateau:
        # Generate data including plateau region
        n_normal = int(n_points * 0.7)
        n_plateau = n_points - n_normal

        # Normal operating range
        current_normal = np.random.uniform(300, Imax, n_normal)

        # Plateau region (above Imax, rpm stays constant)
        current_plateau = np.random.uniform(Imax, Imax * 1.15, n_plateau)

        current = np.concatenate([current_normal, current_plateau])
    else:
        current = np.random.uniform(300, Imax, n_points)

    def add_noise(values, level):
        return values * (1 + np.random.normal(0, level, len(values)))

    # Generate measurements with noise
    rpm = np.where(current <= Imax, Vpmax * (current / Imax) ** 2 + Vp0, Vpmax + Vp0)  # Plateau
    rpm = add_noise(rpm, noise_level)

    # Flow rate: F = F0 + Fmax·Vp/(Vpmax + Vp0)
    flow = F0 + Fmax * rpm / (Vpmax + Vp0)
    flow = add_noise(flow, noise_level)

    # Pressure: P = Pmin + Pmax·[Vp/(Vpmax + Vp0)]²
    pressure = Pmin + Pmax * (rpm / (Vpmax + Vp0)) ** 2
    pressure = add_noise(pressure, noise_level)

    # Back pressure (roughly constant)
    back_pressure = 4.0 + np.random.normal(0, 0.1, n_points)

    # Create DataFrame
    df = pd.DataFrame(
        {
            "current": current,
            "rpm": rpm,
            "flow": flow,
            "pressure": pressure,
            "back_pressure": back_pressure,
        }
    )

    # Sort by current for cleaner visualization
    df = df.sort_values("current").reset_index(drop=True)

    print(f"  Generated {len(df)} measurement points")
    print(f"  Current range: {df['current'].min():.0f} - {df['current'].max():.0f} A")
    print(f"  Rpm range: {df['rpm'].min():.0f} - {df['rpm'].max():.0f} rpm")
    print(f"  Flow range: {df['flow'].min():.1f} - {df['flow'].max():.1f} l/s")
    print(f"  Pressure range: {df['pressure'].min():.1f} - {df['pressure'].max():.1f} bar")
    print()

    return df


def setup_default_params() -> Dict:
    """
    Setup default flow parameters dictionary.

    Returns:
        Dictionary with default flow parameters
    """
    flow_params = {
        "Vp0": {"value": 1000, "unit": "rpm"},
        "Vpmax": {"value": 2840, "unit": "rpm"},
        "F0": {"value": 0, "unit": "l/s"},
        "Fmax": {"value": 61.71612272405876, "unit": "l/s"},
        "Pmax": {"value": 22, "unit": "bar"},
        "Pmin": {"value": 4, "unit": "bar"},
        "Pout": {"value": 4, "unit": "bar"},
        "Imax": {"value": 28000, "unit": "A"},
    }
    return flow_params


def pwlf_fit_pump_speed(
    df: pd.DataFrame,
    current_col: str,
    rpm_col: str,
    max_segments: int = 2,
    show: bool = False,
    debug: bool = False,
) -> Tuple[pwlf.PiecewiseLinFit, List, float]:
    """
    FIT #1: Pump Speed using Piecewise Linear Fitting (pwlf)

    This method automatically detects Imax by finding breakpoints.
    If two segments are found, the breakpoint is the detected Imax.

    Args:
        df: DataFrame with experimental data
        current_col: Column name for current
        rpm_col: Column name for rpm
        max_segments: Maximum number of segments to try
        show: Show plots
        debug: Debug output

    Returns:
        (pwlf_model, equations, detected_Imax)
    """
    print("=" * 70)
    print("STEP 2: Fit Pump Speed using Piecewise Linear Fitting (pwlf)")
    print("=" * 70)
    print("Model: Automatic breakpoint detection for Imax")
    print()

    x = df[current_col].to_numpy()
    y = df[rpm_col].to_numpy()

    best_pwlf = None
    best_eqns = None
    best_segment = 1

    for segment in range(1, max_segments + 1):
        print(f"Trying {segment} segment(s)...")

        my_pwlf = pwlf.PiecewiseLinFit(x, y, degree=2)
        res = my_pwlf.fit(segment)

        errors = my_pwlf.standard_errors()
        p_values = my_pwlf.p_values(method="non-linear", step_size=1e-4)

        print(f"  Breakpoints: {res}")
        print(f"  Beta coefficients: {my_pwlf.beta}")
        print(f"  Standard errors: {errors}")

        # Build equation list
        eqn_list, coeff_list = find_eqn(my_pwlf)

        # Check if fit is good by comparing last point
        final_y = float(eqn_list[-1].evalf(subs={Symbol("x"): x[-1]}))
        error_at_end = abs(final_y - y[-1])

        print(f"  Final y (predicted): {final_y:.2f}, actual: {y[-1]:.2f}")
        print(f"  Error at end: {error_at_end:.2f}")

        # Print fit statistics table
        parameters = np.concatenate((my_pwlf.beta, my_pwlf.fit_breaks[1:-1]))
        se = my_pwlf.se

        tables = []
        headers = [
            "Parameter type",
            "Parameter value",
            "Standard error",
            "t",
            "P > |t| (p-value)",
        ]

        values = np.zeros((parameters.size, 5), dtype=object)
        values[:, 1] = np.around(parameters, decimals=3)
        values[:, 2] = np.around(se, decimals=3)
        values[:, 3] = np.around(parameters / se, decimals=3)
        values[:, 4] = np.around(p_values, decimals=3)

        for i, row in enumerate(values):
            table = []
            if i < my_pwlf.beta.size:
                table.append("Beta")
            else:
                table.append("Breakpoint")
            table += row.tolist()[1:]
            tables.append(table)

        print(tabulate(tables, headers=headers, tablefmt="psql"))
        print()

        # Accept if error is small enough
        if error_at_end <= 10:
            best_pwlf = my_pwlf
            best_eqns = eqn_list
            best_segment = segment
            break

        best_pwlf = my_pwlf
        best_eqns = eqn_list
        best_segment = segment

    # Detect Imax from breakpoints
    if best_pwlf.n_segments == 2:
        detected_imax = best_pwlf.fit_breaks[1]
        print(f"**DETECTED Imax from breakpoint: {detected_imax:.0f} A**")
    else:
        # Use data range if no breakpoint detected
        detected_imax = x.max()
        print(f"No breakpoint detected, using max current: {detected_imax:.0f} A")

    # Calculate Vp0 and Vpmax from equation
    vp0 = float(best_eqns[0].evalf(subs={Symbol("x"): 0}))
    vpmax = float(best_eqns[0].evalf(subs={Symbol("x"): detected_imax}))

    print("\nExtracted Parameters:")
    print(f"  Vp0 (at I=0): {vp0:.2f} rpm")
    print(f"  Vpmax (at I=Imax): {vpmax:.2f} rpm")
    print(f"  Imax: {detected_imax:.0f} A")
    print()

    # Visualization
    if show:
        xHat = np.linspace(x.min(), x.max(), 1000)
        yHat = best_pwlf.predict(xHat)

        plt.figure(figsize=(10, 6))
        plt.plot(x, y, "o", alpha=0.5, label="Experimental data")
        plt.plot(xHat, yHat, "r-", linewidth=2, label=f"pwlf fit ({best_segment} segment(s))")

        if debug:
            for i, eqn in enumerate(best_eqns):
                eqnHat = [float(eqn.evalf(subs={Symbol("x"): val})) for val in xHat]
                plt.plot(xHat, eqnHat, ".", alpha=0.2, label=f"Segment {i+1}")

        if best_pwlf.n_segments == 2:
            plt.axvline(
                x=detected_imax,
                color="green",
                linestyle="--",
                label=f"Detected Imax = {detected_imax:.0f} A",
            )

        plt.xlabel("Current [A]")
        plt.ylabel("Pump Speed [rpm]")
        plt.title(f"Pump Speed vs Current (pwlf, {best_segment} segments)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("pwlf_pump_speed_fit.png", dpi=150)
        print("  Saved plot: pwlf_pump_speed_fit.png")
        plt.close()

    return best_pwlf, best_eqns, detected_imax


def fit_flow_rate_magnetrun(
    df: pd.DataFrame,
    current_col: str,
    flow_col: str,
    imax: float,
    vpmax: float,
    vp0: float,
    name: str = "flow",
    show: bool = False,
    debug: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    FIT #2: Flow Rate using python_magnetrun.fit

    Model: F(I) = F0 + Fmax·Vp(I)/(Vpmax + Vp0)

    Returns:
        (params, covariance) where params = [F0, Fmax]
    """
    print("=" * 70)
    print("STEP 3: Fit Flow Rate using python_magnetrun.fit")
    print("=" * 70)
    print("Model: F(I) = F0 + Fmax·Vp(I)/(Vpmax + Vp0)")
    print()

    def vpump_func(x, vpmax, vp0, imax):
        return vpmax * (x / imax) ** 2 + vp0

    def flow_func(x, a: float, b: float):
        """a=F0, b=Fmax"""
        vp = vpump_func(x, vpmax, vp0, imax)
        return a + b * vp / (vpmax + vp0)

    # Use python_magnetrun's fit function
    params, params_covariance = fit(
        current_col,
        flow_col,
        "Flow",
        imax,
        flow_func,
        df,
        os.getcwd(),
        name,
        show,
        debug,
    )

    print("Flow Rate Fit Results:")
    print(f"  F0 = {params[0]:.4f} l/s")
    print(f"  Fmax = {params[1]:.4f} l/s")
    print(f"  Covariance diagonal: {np.diag(params_covariance)}")
    print()

    return params, params_covariance


def fit_pressure_magnetrun(
    df: pd.DataFrame,
    current_col: str,
    pressure_col: str,
    imax: float,
    vpmax: float,
    vp0: float,
    name: str = "pressure",
    show: bool = False,
    debug: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    FIT #3: Pressure using python_magnetrun.fit

    Model: P(I) = Pmin + Pmax·[Vp(I)/(Vpmax + Vp0)]²

    Returns:
        (params, covariance) where params = [Pmin, Pmax]
    """
    print("=" * 70)
    print("STEP 4: Fit Pressure using python_magnetrun.fit")
    print("=" * 70)
    print("Model: P(I) = Pmin + Pmax·[Vp(I)/(Vpmax + Vp0)]²")
    print()

    def vpump_func(x, vpmax, vp0, imax):
        return vpmax * (x / imax) ** 2 + vp0

    def pressure_func(x, a: float, b: float):
        """a=Pmin, b=Pmax"""
        vp = vpump_func(x, vpmax, vp0, imax)
        return a + b * (vp / (vpmax + vp0)) ** 2

    # Use python_magnetrun's fit function
    params, params_covariance = fit(
        current_col,
        pressure_col,
        "Pressure",
        imax,
        pressure_func,
        df,
        os.getcwd(),
        name,
        show,
        debug,
    )

    print("Pressure Fit Results:")
    print(f"  Pmin = {params[0]:.4f} bar")
    print(f"  Pmax = {params[1]:.4f} bar")
    print(f"  Covariance diagonal: {np.diag(params_covariance)}")
    print()

    return params, params_covariance


def calculate_back_pressure_stats(
    df: pd.DataFrame,
    current_col: str,
    bp_col: str,
    imax: float,
    show: bool = False,
    debug: bool = False,
) -> Tuple[float, float]:
    """
    Calculate statistics for back pressure (roughly constant).

    Returns:
        (mean, std)
    """
    print("=" * 70)
    print("STEP 5: Calculate Back Pressure Statistics")
    print("=" * 70)

    # Filter data up to Imax
    result = df.query(f"{current_col} <= {imax}")

    if debug:
        print(f"  Total rows: {len(df)}, filtered rows: {len(result)}")
        print(f"  Result max current: {result[current_col].max()}")

    bp_mean = result[bp_col].mean()
    bp_std = result[bp_col].std()

    print("Back Pressure (Pout):")
    print(f"  Mean = {bp_mean:.4f} bar")
    print(f"  Std  = {bp_std:.4f} bar")
    print()

    if show:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(result[current_col], result[bp_col], "o", alpha=0.5)
        plt.axhline(y=bp_mean, color="r", linestyle="--", label=f"Mean = {bp_mean:.2f}")
        plt.xlabel("Current [A]")
        plt.ylabel("Back Pressure [bar]")
        plt.title("Back Pressure vs Current")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.hist(result[bp_col], bins=30, alpha=0.7, edgecolor="black")
        plt.axvline(x=bp_mean, color="r", linestyle="--", label=f"Mean = {bp_mean:.2f}")
        plt.xlabel("Back Pressure [bar]")
        plt.ylabel("Frequency")
        plt.title("Back Pressure Distribution")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("back_pressure_stats.png", dpi=150)
        print("  Saved plot: back_pressure_stats.png")
        plt.close()

    return bp_mean, bp_std


def build_flow_params_dict(
    vpmax: float,
    vp0: float,
    fmax: float,
    f0: float,
    pmax: float,
    pmin: float,
    bp: float,
    imax: float,
) -> Dict:
    """
    Build flow_params dictionary in the format expected by waterflow_factory.

    Returns:
        Dictionary with flow parameters
    """
    print("=" * 70)
    print("STEP 6: Build flow_params Dictionary")
    print("=" * 70)

    flow_params = {
        "Vp0": {"value": vp0, "unit": "rpm"},
        "Vpmax": {"value": vpmax, "unit": "rpm"},
        "F0": {"value": f0, "unit": "l/s"},
        "Fmax": {"value": fmax, "unit": "l/s"},
        "Pmax": {"value": pmax, "unit": "bar"},
        "Pmin": {"value": pmin, "unit": "bar"},
        "Pout": {"value": bp, "unit": "bar"},  # Using Pout as in magnetrun version
        "Imax": {"value": imax, "unit": "A"},
    }

    print("Flow parameters dictionary created:")
    print(json.dumps(flow_params, indent=2))
    print()

    return flow_params


def create_waterflow_object(flow_params: Dict) -> WaterFlow:
    """
    Create WaterFlow object using waterflow_factory.

    Note: waterflow_factory expects "BP" but magnetrun uses "Pout"
    The factory handles both keys.
    """
    print("=" * 70)
    print("STEP 7: Create WaterFlow Object using waterflow_factory")
    print("=" * 70)

    waterflow = from_flow_params(flow_params)

    print("WaterFlow object created successfully!")
    print(f"  Type: {type(waterflow)}")
    print(f"  Pump speed range: {waterflow.pump_speed_min} - {waterflow.pump_speed_max} rpm")
    print(f"  Flow range: {waterflow.flow_min} - {waterflow.flow_max} l/s")
    print(f"  Pressure range: {waterflow.pressure_min} - {waterflow.pressure_max} bar")
    print(f"  Max current: {waterflow.current_max} A")
    print()

    return waterflow


def demonstrate_calculations(waterflow: WaterFlow):
    """
    Demonstrate hydraulic calculations using the WaterFlow object.
    """
    print("=" * 70)
    print("STEP 8: Perform Hydraulic Calculations")
    print("=" * 70)

    test_currents = [10000, 15000, 20000, 25000, 28000]
    cross_section = 1e-4  # m²

    print(f"{'Current':>10} {'Pump Speed':>12} {'Flow Rate':>12} {'Pressure':>10} {'Velocity':>10}")
    print(f"{'[A]':>10} {'[rpm]':>12} {'[m³/s]':>12} {'[bar]':>10} {'[m/s]':>10}")
    print("-" * 70)

    for current in test_currents:
        speed = waterflow.pump_speed(current)
        flow = waterflow.flow_rate(current)
        pressure = waterflow.pressure(current)
        velocity = waterflow.velocity(current, cross_section)

        print(f"{current:10d} {speed:12.2f} {flow:12.6f} {pressure:10.2f} {velocity:10.2f}")

    print()


def save_flow_params(flow_params: Dict, filename: str = "flow_params_magnetrun_output.json"):
    """
    Save flow_params to JSON file.
    """
    print("=" * 70)
    print("STEP 9: Save Results")
    print("=" * 70)

    with open(filename, "w") as f:
        json.dump(flow_params, indent=4, fp=f)

    print(f"  Flow parameters saved to: {filename}")
    print(f"  Can be loaded later with: WaterFlow.from_file('{filename}')")
    print()


def main():
    """
    Main pipeline execution using python_magnetrun methods.
    """
    print("\n")
    print("*" * 70)
    print("FLOW PARAMETER EXTRACTION PIPELINE (python_magnetrun)")
    print("DataFrame → pwlf + magnetrun.fit → flow_params → WaterFlow Object")
    print("*" * 70)
    print("\n")

    # Configuration
    show_plots = "--show-plots" in sys.argv
    debug = "--debug" in sys.argv

    # Step 1: Generate experimental data as DataFrame
    df = generate_synthetic_data(n_points=500, noise_level=0.02, include_plateau=True)

    # Step 2: Fit pump speed using pwlf (with automatic Imax detection)
    pwlf_model, eqns, detected_imax = pwlf_fit_pump_speed(
        df, current_col="current", rpm_col="rpm", max_segments=2, show=show_plots, debug=debug
    )

    # Extract Vp0 and Vpmax from pwlf equations
    vp0 = float(eqns[0].evalf(subs={Symbol("x"): 0}))
    vpmax = float(eqns[0].evalf(subs={Symbol("x"): detected_imax}))

    # Step 3: Fit flow rate using python_magnetrun.fit
    flow_params_fit, _ = fit_flow_rate_magnetrun(
        df,
        current_col="current",
        flow_col="flow",
        imax=detected_imax,
        vpmax=vpmax,
        vp0=vp0,
        name="magnetrun_flow",
        show=show_plots,
        debug=debug,
    )
    f0, fmax = flow_params_fit[0], flow_params_fit[1]

    # Step 4: Fit pressure using python_magnetrun.fit
    pressure_params_fit, _ = fit_pressure_magnetrun(
        df,
        current_col="current",
        pressure_col="pressure",
        imax=detected_imax,
        vpmax=vpmax,
        vp0=vp0,
        name="magnetrun_pressure",
        show=show_plots,
        debug=debug,
    )
    pmin, pmax = pressure_params_fit[0], pressure_params_fit[1]

    # Step 5: Calculate back pressure statistics
    bp_mean, bp_std = calculate_back_pressure_stats(
        df,
        current_col="current",
        bp_col="back_pressure",
        imax=detected_imax,
        show=show_plots,
        debug=debug,
    )

    # Step 6: Build flow_params dictionary
    flow_params = build_flow_params_dict(
        vpmax=vpmax, vp0=vp0, fmax=fmax, f0=f0, pmax=pmax, pmin=pmin, bp=bp_mean, imax=detected_imax
    )

    # Step 7: Create WaterFlow object using factory
    waterflow = create_waterflow_object(flow_params)

    # Step 8: Demonstrate calculations
    demonstrate_calculations(waterflow)

    # Step 9: Save results
    save_flow_params(flow_params, "flow_params_magnetrun_output.json")

    print("*" * 70)
    print("PIPELINE COMPLETE!")
    print("*" * 70)
    print("\nSummary:")
    print("  ✓ Used pwlf for pump speed fit with automatic Imax detection")
    print("  ✓ Used python_magnetrun.fit for flow rate and pressure")
    print("  ✓ Created flow_params dictionary")
    print("  ✓ Built WaterFlow object using waterflow_factory")
    print("  ✓ Performed hydraulic calculations")
    print("  ✓ Saved results to JSON")
    print("\nKey Differences from basic pipeline:")
    print("  • Uses piecewise linear fitting (pwlf) for automatic breakpoint detection")
    print("  • Detects Imax from data (breakpoint in pump speed curve)")
    print("  • Uses python_magnetrun.fit instead of scipy.optimize.curve_fit")
    print("  • Works with pandas DataFrame (magnetrun's preferred format)")
    print("\n")

    return waterflow


if __name__ == "__main__":
    waterflow = main()
