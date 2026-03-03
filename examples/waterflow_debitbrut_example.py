"""
Example demonstrating the debitbrut() method with hysteresis model for secondary cooling loop.

This example shows how to:
1. Configure hysteresis parameters in WaterFlow
2. Use the debitbrut() method to compute secondary flow rates from power
3. Visualize the hysteresis behavior

Note: 'debitbrut' refers to the secondary cooling loop flow rate (French term maintained
for compatibility). In CSV files, use the column name 'flow_secondary' for clarity.
"""

import numpy as np
import matplotlib.pyplot as plt
from python_magnetcooling.waterflow import WaterFlow


def example_basic():
    """Basic example of debitbrut() method for secondary flow with hysteresis"""
    print("=" * 60)
    print("Basic secondary flow (debitbrut) example")
    print("=" * 60)
    
    # Create WaterFlow instance with hysteresis parameters
    # Each threshold is a tuple: (ascending_threshold, descending_threshold)
    flow = WaterFlow(
        pump_speed_min=1000,
        pump_speed_max=2840,
        flow_min=0,
        flow_max=140,
        pressure_max=22,
        pressure_min=4,
        pressure_back=4,
        current_max=28000,
        # Hysteresis parameters: (ascending, descending) pairs
        hysteresis_thresholds=[(3, 2), (8, 6), (12, 10)],  # MW
        hysteresis_low_values=[100, 200, 300, 400],  # m³/h
        hysteresis_high_values=[100, 250, 350, 450]  # m³/h
    )
    
    # Simulate power cycle: increase then decrease
    power_sequence = [0, 2, 5, 8, 10, 12, 15, 18, 15, 12, 10, 8, 5, 2, 0]
    
    print("\nPower [MW] -> Secondary Flow [m³/h]")
    print("-" * 40)
    
    # Use the debitbrut() method which properly handles arrays
    power_array = np.array(power_sequence)
    flow_rates = flow.debitbrut(power_array)
    
    for power, flow_rate in zip(power_sequence, flow_rates):
        print(f"{power:6.1f} MW -> {flow_rate:6.1f} m³/h")
    
    return power_sequence, flow_rates.tolist()


def example_from_json():
    """Example loading hysteresis parameters from JSON file"""
    print("\n" + "=" * 60)
    print("Loading from JSON with hysteresis parameters")
    print("=" * 60)
    
    import json
    import tempfile
    import os
    
    # Create example JSON with hysteresis parameters
    config = {
        "Vp0": {"value": 1000, "unit": "rpm"},
        "Vpmax": {"value": 2840, "unit": "rpm"},
        "F0": {"value": 0, "unit": "l/s"},
        "Fmax": {"value": 61.7, "unit": "l/s"},
        "Pmax": {"value": 22, "unit": "bar"},
        "Pmin": {"value": 4, "unit": "bar"},
        "BP": {"value": 4, "unit": "bar"},
        "Imax": {"value": 28000, "unit": "A"},
        "hysteresis": {
            "thresholds": [5, 10, 15],
            "low_values": [100, 200, 300, 400],
            "high_values": [100, 250, 350, 450],
            "unit_thresholds": "MW",
            "unit_values": "m³/h"
        }
    }
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f, indent=2)
        temp_file = f.name
    
    try:
        # Load from file
        flow = WaterFlow.from_file(temp_file)
        
        print(f"\nLoaded hysteresis configuration:")
        print(f"  Thresholds: {flow.hysteresis_thresholds} MW")
        print(f"  Low values: {flow.hysteresis_low_values} m³/h")
        print(f"  High values: {flow.hysteresis_high_values} m³/h")
        
        # Test the configuration
        print(f"\nConfiguration is valid: {len(flow.hysteresis_thresholds) > 0}")
        
        # Save back to see the format
        output_file = temp_file.replace('.json', '_output.json')
        flow.to_file(output_file)
        print(f"\nSaved configuration to: {output_file}")
        
        with open(output_file, 'r') as f:
            print("\nSaved JSON content:")
            print(f.read())
        
        return flow
        
    finally:
        # Cleanup
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def example_with_array():
    """Example using array of power values for secondary flow"""
    print("\n" + "=" * 60)
    print("Using debitbrut() with power array for secondary flow")
    print("=" * 60)
    
    flow = WaterFlow(
        hysteresis_thresholds=[(3, 2), (8, 6), (12, 10)],
        hysteresis_low_values=[100, 200, 300, 400],
        hysteresis_high_values=[100, 250, 350, 450]
    )
    
    # Create power cycle with ramp up and ramp down
    t = np.linspace(0, 2*np.pi, 100)
    power = 10 + 8 * np.sin(t)  # Oscillates between 2 and 18 MW
    
    # Compute flow rates using debitbrut method
    flow_rates = flow.debitbrut(power)
    
    # Plot the hysteresis curve
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(t, power, 'b-', label='Power')
    plt.xlabel('Time [arbitrary]')
    plt.ylabel('Power [MW]')
    plt.title('Power vs Time')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(power, flow_rates, 'r.-', alpha=0.5, markersize=2)
    plt.xlabel('Power [MW]')
    plt.ylabel('Secondary Flow Rate [m³/h]')
    plt.title('Hysteresis: Secondary Flow Rate vs Power')
    plt.grid(True)
    
    # Add threshold lines
    for threshold in flow.hysteresis_thresholds:
        plt.axvline(threshold, color='gray', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('debitbrut_hysteresis_example.png', dpi=150)
    print("\nPlot saved to: debitbrut_hysteresis_example.png")
    plt.show()
    
    return power, flow_rates


def example_error_handling():
    """Example showing error handling"""
    print("\n" + "=" * 60)
    print("Error handling examples")
    print("=" * 60)
    
    # Example 1: Missing hysteresis parameters
    flow = WaterFlow()
    try:
        flow.debitbrut(10)
    except ValueError as e:
        print(f"\n✓ Expected error for missing parameters:\n  {e}")
    
    # Example 2: Mismatched array sizes
    flow.hysteresis_thresholds = [(3, 2), (8, 6), (12, 10)]
    flow.hysteresis_low_values = [100, 200]  # Wrong size!
    flow.hysteresis_high_values = [100, 250, 350, 450]
    
    try:
        flow.debitbrut(10)
    except ValueError as e:
        print(f"\n✓ Expected error for size mismatch:\n  {e}")


if __name__ == "__main__":
    # Run all examples
    example_basic()
    example_from_json()
    example_with_array()
    example_error_handling()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
