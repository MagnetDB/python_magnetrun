"""
Complete Flow Parameter Extraction Pipeline

This script demonstrates the full workflow:
1. Load experimental data from database records
2. Fit pump speed, flow rate, and pressure curves
3. Build flow_params dictionary
4. Create WaterFlow object using waterflow_factory
5. Perform hydraulic calculations

This is a simplified, standalone version of the compute() method from flow_params.py
"""

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Callable
import json

# Import the factory module
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from python_magnetcooling.waterflow_factory import from_flow_params
from python_magnetcooling import WaterFlow


def generate_synthetic_data(
    n_points: int = 500, noise_level: float = 0.02
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic experimental data simulating magnet cooling system measurements.

    In real workflow, this would be loaded from database records.

    Args:
        n_points: Number of data points to generate
        noise_level: Relative noise level (0.02 = 2% noise)

    Returns:
        Dictionary with current and measurement arrays
    """
    print("=" * 70)
    print("STEP 1: Load Experimental Data from Database Records")
    print("=" * 70)
    print("(Using synthetic data for demonstration)")

    # True parameters (what we're trying to recover)
    Imax = 28000  # A
    Vpmax = 2840  # rpm
    Vp0 = 1000  # rpm
    Fmax = 140  # l/s
    F0 = 0  # l/s
    Pmax = 22  # bar
    Pmin = 4  # bar

    # Generate current values (operating points)
    current = np.random.uniform(5000, Imax, n_points)

    # Generate measurements with noise
    def add_noise(values, level):
        return values * (1 + np.random.normal(0, level, len(values)))

    # Pump speed: Vp = Vpmax·(I/Imax)² + Vp0
    rpm = Vpmax * (current / Imax) ** 2 + Vp0
    rpm = add_noise(rpm, noise_level)

    # Flow rate: F = F0 + Fmax·Vp/(Vpmax + Vp0)
    flow = F0 + Fmax * rpm / (Vpmax + Vp0)
    flow = add_noise(flow, noise_level)

    # Pressure: P = Pmin + Pmax·[Vp/(Vpmax + Vp0)]²
    pressure = Pmin + Pmax * (rpm / (Vpmax + Vp0)) ** 2
    pressure = add_noise(pressure, noise_level)

    # Back pressure (roughly constant with some variation)
    back_pressure = 4.0 + np.random.normal(0, 0.1, n_points)

    print(f"  Loaded {n_points} measurement points")
    print(f"  Current range: {current.min():.0f} - {current.max():.0f} A")
    print(f"  Rpm range: {rpm.min():.0f} - {rpm.max():.0f} rpm")
    print(f"  Flow range: {flow.min():.1f} - {flow.max():.1f} l/s")
    print(f"  Pressure range: {pressure.min():.1f} - {pressure.max():.1f} bar")
    print()

    return {
        "current": current,
        "rpm": rpm,
        "flow": flow,
        "pressure": pressure,
        "back_pressure": back_pressure,
        "Imax": Imax,
    }


def perform_fit(
    x_data: np.ndarray,
    y_data: np.ndarray,
    fit_function: Callable,
    param_names: List[str],
    quantity_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform curve fit using scipy.optimize.curve_fit

    This is the core fitting function used in flow_params.py

    Args:
        x_data: Independent variable (current)
        y_data: Dependent variable (rpm, flow, or pressure)
        fit_function: Function to fit (must accept x, a, b)
        param_names: Names of parameters [a, b]
        quantity_name: Name of quantity being fitted

    Returns:
        Fitted parameters and their standard errors
    """
    # Perform non-linear least squares fit
    params, params_covariance = optimize.curve_fit(fit_function, x_data, y_data)

    # Calculate standard errors
    stderr = np.sqrt(np.diag(params_covariance))

    # Calculate fit quality (R²)
    y_fit = fit_function(x_data, *params)
    residuals = y_data - y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    print(f"{quantity_name} Fit Results:")
    for name, value, error in zip(param_names, params, stderr):
        print(f"  {name} = {value:.4f} ± {error:.4f}")
    print(f"  R² = {r_squared:.6f}")
    print()

    return params, stderr


def fit_pump_speed(data: Dict) -> Tuple[float, float]:
    """
    FIT #1: Pump Speed vs Current

    Model: Vp(I) = Vpmax·(I/Imax)² + Vp0

    Returns:
        (Vpmax, Vp0)
    """
    print("=" * 70)
    print("STEP 2: Fit Pump Speed Curve")
    print("=" * 70)
    print("Model: Vp(I) = Vpmax·(I/Imax)² + Vp0")
    print()

    Imax = data["Imax"]

    def vpump_func(x, a: float, b: float):
        """a=Vpmax, b=Vp0"""
        return a * (x / Imax) ** 2 + b

    params, _ = perform_fit(
        data["current"], data["rpm"], vpump_func, ["Vpmax", "Vp0"], "Pump Speed"
    )

    return params[0], params[1]  # Vpmax, Vp0


def fit_flow_rate(data: Dict, vpmax: float, vp0: float) -> Tuple[float, float]:
    """
    FIT #2: Flow Rate vs Current

    Model: F(I) = F0 + Fmax·Vp(I)/(Vpmax + Vp0)

    Args:
        vpmax, vp0: Parameters from pump speed fit

    Returns:
        (F0, Fmax)
    """
    print("=" * 70)
    print("STEP 3: Fit Flow Rate Curve")
    print("=" * 70)
    print("Model: F(I) = F0 + Fmax·Vp(I)/(Vpmax + Vp0)")
    print()

    Imax = data["Imax"]

    def vpump_func(x, vpmax, vp0):
        return vpmax * (x / Imax) ** 2 + vp0

    def flow_func(x, a: float, b: float):
        """a=F0, b=Fmax"""
        vp = vpump_func(x, vpmax, vp0)
        return a + b * vp / (vpmax + vp0)

    params, _ = perform_fit(data["current"], data["flow"], flow_func, ["F0", "Fmax"], "Flow Rate")

    return params[0], params[1]  # F0, Fmax


def fit_pressure(data: Dict, vpmax: float, vp0: float) -> Tuple[float, float]:
    """
    FIT #3: Pressure vs Current

    Model: P(I) = Pmin + Pmax·[Vp(I)/(Vpmax + Vp0)]²

    Args:
        vpmax, vp0: Parameters from pump speed fit

    Returns:
        (Pmin, Pmax)
    """
    print("=" * 70)
    print("STEP 4: Fit Pressure Curve")
    print("=" * 70)
    print("Model: P(I) = Pmin + Pmax·[Vp(I)/(Vpmax + Vp0)]²")
    print()

    Imax = data["Imax"]

    def vpump_func(x, vpmax, vp0):
        return vpmax * (x / Imax) ** 2 + vp0

    def pressure_func(x, a: float, b: float):
        """a=Pmin, b=Pmax"""
        vp = vpump_func(x, vpmax, vp0)
        return a + b * (vp / (vpmax + vp0)) ** 2

    params, _ = perform_fit(
        data["current"], data["pressure"], pressure_func, ["Pmin", "Pmax"], "Pressure"
    )

    return params[0], params[1]  # Pmin, Pmax


def calculate_back_pressure_stats(data: Dict) -> Tuple[float, float]:
    """
    Calculate statistics for back pressure (roughly constant)

    Returns:
        (mean, std)
    """
    print("=" * 70)
    print("STEP 5: Calculate Back Pressure Statistics")
    print("=" * 70)

    bp_mean = np.mean(data["back_pressure"])
    bp_std = np.std(data["back_pressure"])

    print("Back Pressure (Pout):")
    print(f"  Mean = {bp_mean:.4f} bar")
    print(f"  Std  = {bp_std:.4f} bar")
    print()

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
    Build flow_params dictionary in the format expected by waterflow_factory

    This is the format saved by compute() in flow_params.py
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
        "BP": {"value": bp, "unit": "bar"},
        "Imax": {"value": imax, "unit": "A"},
    }

    print("Flow parameters dictionary created:")
    print(json.dumps(flow_params, indent=2))
    print()

    return flow_params


def create_waterflow_object(flow_params: Dict) -> WaterFlow:
    """
    Create WaterFlow object using waterflow_factory

    This is where the factory module comes in!
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
    Demonstrate hydraulic calculations using the WaterFlow object
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


def plot_results(data: Dict, waterflow: WaterFlow, save_plot: bool = True):
    """
    Create visualization of fitted curves vs experimental data
    """
    print("=" * 70)
    print("STEP 9: Visualize Results")
    print("=" * 70)

    # Generate smooth curve for plotting
    I_smooth = np.linspace(0, data["Imax"], 100)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Flow Parameter Fitting Results", fontsize=14, fontweight="bold")

    # Plot 1: Pump Speed
    ax = axes[0, 0]
    ax.scatter(data["current"], data["rpm"], alpha=0.5, s=20, label="Experimental data")
    rpm_fit = [waterflow.pump_speed(i) for i in I_smooth]
    ax.plot(I_smooth, rpm_fit, "r-", linewidth=2, label="Fitted curve")
    ax.set_xlabel("Current [A]")
    ax.set_ylabel("Pump Speed [rpm]")
    ax.set_title("Pump Speed vs Current")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Flow Rate
    ax = axes[0, 1]
    ax.scatter(data["current"], data["flow"], alpha=0.5, s=20, label="Experimental data")
    flow_fit = [waterflow.flow_rate(i) * 1000 for i in I_smooth]  # Convert m³/s to l/s
    ax.plot(I_smooth, flow_fit, "r-", linewidth=2, label="Fitted curve")
    ax.set_xlabel("Current [A]")
    ax.set_ylabel("Flow Rate [l/s]")
    ax.set_title("Flow Rate vs Current")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Pressure
    ax = axes[1, 0]
    ax.scatter(data["current"], data["pressure"], alpha=0.5, s=20, label="Experimental data")
    pressure_fit = [waterflow.pressure(i) for i in I_smooth]
    ax.plot(I_smooth, pressure_fit, "r-", linewidth=2, label="Fitted curve")
    ax.set_xlabel("Current [A]")
    ax.set_ylabel("Pressure [bar]")
    ax.set_title("Pressure vs Current")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Velocity (calculated)
    ax = axes[1, 1]
    cross_section = 1e-4  # m²
    velocity_fit = [waterflow.velocity(i, cross_section) for i in I_smooth]
    ax.plot(I_smooth, velocity_fit, "b-", linewidth=2)
    ax.set_xlabel("Current [A]")
    ax.set_ylabel("Velocity [m/s]")
    ax.set_title(f"Velocity vs Current (A = {cross_section} m²)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_plot:
        plot_filename = "flow_params_pipeline_results.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches="tight")
        print(f"  Plot saved as: {plot_filename}")

    print("  Visualization complete!")
    print()

    return fig


def save_flow_params(flow_params: Dict, filename: str = "flow_params_output.json"):
    """
    Save flow_params to JSON file (as done in compute() method)
    """
    print("=" * 70)
    print("STEP 10: Save Results")
    print("=" * 70)

    with open(filename, "w") as f:
        json.dump(flow_params, indent=4, fp=f)

    print(f"  Flow parameters saved to: {filename}")
    print(f"  Can be loaded later with: WaterFlow.from_file('{filename}')")
    print()


def main():
    """
    Main pipeline execution
    """
    print("\n")
    print("*" * 70)
    print("FLOW PARAMETER EXTRACTION PIPELINE")
    print("Database Records → Curve Fitting → flow_params → WaterFlow Object")
    print("*" * 70)
    print("\n")

    # Step 1: Load/generate experimental data
    data = generate_synthetic_data(n_points=500, noise_level=0.02)

    # Step 2-4: Perform curve fits
    vpmax, vp0 = fit_pump_speed(data)
    f0, fmax = fit_flow_rate(data, vpmax, vp0)
    pmin, pmax = fit_pressure(data, vpmax, vp0)

    # Step 5: Calculate back pressure statistics
    bp_mean, bp_std = calculate_back_pressure_stats(data)

    # Step 6: Build flow_params dictionary
    flow_params = build_flow_params_dict(
        vpmax=vpmax, vp0=vp0, fmax=fmax, f0=f0, pmax=pmax, pmin=pmin, bp=bp_mean, imax=data["Imax"]
    )

    # Step 7: Create WaterFlow object using factory
    waterflow = create_waterflow_object(flow_params)

    # Step 8: Demonstrate calculations
    demonstrate_calculations(waterflow)

    # Step 9: Visualize results
    plot_results(data, waterflow, save_plot=True)

    # Step 10: Save results
    save_flow_params(flow_params, "flow_params_output.json")

    print("*" * 70)
    print("PIPELINE COMPLETE!")
    print("*" * 70)
    print("\nSummary:")
    print("  ✓ Fitted 3 curves (pump speed, flow rate, pressure)")
    print("  ✓ Created flow_params dictionary")
    print("  ✓ Built WaterFlow object using waterflow_factory")
    print("  ✓ Performed hydraulic calculations")
    print("  ✓ Generated visualization")
    print("  ✓ Saved results to JSON")
    print("\n")

    return waterflow


if __name__ == "__main__":
    waterflow = main()

    # Optionally show plots
    import sys

    if "--show-plots" in sys.argv:
        plt.show()
