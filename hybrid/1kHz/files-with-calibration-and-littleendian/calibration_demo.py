"""
Calibration Example - Working with FEPC analog calibration

This script demonstrates:
1. Reading calibration parameters from CFG file
2. Loading CNV files (piecewise calibration)
3. Applying both linear and piecewise calibration
4. Comparing raw vs calibrated data
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from fepc_reader import (
    parse_cfg_file,
    read_hour_file,
    load_calibration,
    apply_calibration,
    calibrate_channel,
    CalibrationInfo
)


def display_calibration_info(config, verbose=True):
    """Display calibration parameters for all analog channels"""
    print("=" * 80)
    print("CALIBRATION PARAMETERS FROM CFG FILE")
    print("=" * 80)
    
    analog_slots = config.get_analog_slots()
    
    for slot in analog_slots:
        card = config.get_card_by_slot(slot)
        print(f"\n{'='*80}")
        print(f"SLOT {slot} - {card.card_type} CARD")
        print(f"{'='*80}")
        
        if verbose:
            print(f"\n{'Ch':<4} {'Variable':<20} {'Calibration Type':<15} {'Parameters'}")
            print("-" * 80)
            
            for i, (var_name, calib) in enumerate(zip(card.variable_names, card.calibrations)):
                if calib.cnv_file:
                    calib_type = "Piecewise (CNV)"
                    params = f"File: {calib.cnv_file}"
                else:
                    calib_type = "Linear"
                    # Formula: Signal = A * (COEF_A * N + COEF_B) + B
                    params = f"A={calib.a:.6e}, B={calib.b:.3f}, Ca={calib.coef_a:.3f}, Cb={calib.coef_b:.3f}"
                
                print(f"{i:<4} {var_name:<20} {calib_type:<15} {params}")
        else:
            # Summary only
            linear_count = sum(1 for c in card.calibrations if not c.cnv_file)
            piecewise_count = sum(1 for c in card.calibrations if c.cnv_file)
            print(f"  Linear calibrations: {linear_count}")
            print(f"  Piecewise calibrations (CNV): {piecewise_count}")


def demonstrate_linear_calibration():
    """Show how linear calibration works with example values"""
    print("\n" + "=" * 80)
    print("LINEAR CALIBRATION DEMONSTRATION")
    print("=" * 80)
    
    # Example from documentation
    print("\nExample from documentation:")
    print("Formula: Signal = A * (COEF_A * N + COEF_B) + B")
    print("Parameters: COEF_A=1, COEF_B=0, A=3.0518044E-4, B=-10")
    
    calib = CalibrationInfo(coef_a=1.0, coef_b=0.0, a=3.0518044e-4, b=-10.0)
    
    # Test values
    test_values = np.array([0, 16383, 32768, 49152, 65535], dtype=np.int16)
    calibrated = apply_calibration(test_values, calib_info=calib)
    
    print(f"\n{'N (raw)':<12} {'Signal (V)':<15}")
    print("-" * 27)
    for n, sig in zip(test_values, calibrated):
        print(f"{n:<12} {sig:<15.6f}")
    
    print("\nNote: N=0 → -10V, N=65535 → +10V (20V range)")


def demonstrate_cnv_calibration(cnv_file):
    """Show how piecewise calibration works with CNV file"""
    print("\n" + "=" * 80)
    print("PIECEWISE (CNV) CALIBRATION DEMONSTRATION")
    print("=" * 80)
    
    if not Path(cnv_file).exists():
        print(f"\n⚠ CNV file not found: {cnv_file}")
        print("This example requires a .CNV file from your FEPC system")
        return None
    
    print(f"\nLoading: {cnv_file}")
    
    # Load CNV file
    cnv_dict = load_calibration(cnv_file)
    
    print(f"\nCalibration points: {len(cnv_dict['n_values'])}")
    print(f"\n{'N (raw)':<12} {'Physical Value':<20}")
    print("-" * 32)
    
    for n, val in zip(cnv_dict['n_values'], cnv_dict['physical_values']):
        print(f"{n:<12} {val:<20.6f}")
    
    # Demonstrate interpolation
    print("\nInterpolation between points:")
    test_n = np.linspace(
        cnv_dict['n_values'][0],
        cnv_dict['n_values'][-1],
        10
    ).astype(np.int16)
    
    interpolated = apply_calibration(test_n, cnv_dict=cnv_dict)
    
    print(f"\n{'N (raw)':<12} {'Interpolated Value':<20}")
    print("-" * 32)
    for n, val in zip(test_n, interpolated):
        print(f"{n:<12} {val:<20.6f}")
    
    return cnv_dict


def apply_and_plot_calibration(config, slot=0, channel=0, num_blocks=1000):
    """
    Read data, apply calibration, and plot comparison
    """
    print("\n" + "=" * 80)
    print("CALIBRATION APPLICATION AND VISUALIZATION")
    print("=" * 80)
    
    card = config.get_card_by_slot(slot)
    
    if card.card_type != 'ANA':
        print(f"⚠ Slot {slot} is not an analog card")
        return
    
    var_name = card.variable_names[channel]
    print(f"\nSlot: {slot}")
    print(f"Channel: {channel}")
    print(f"Variable: {var_name}")
    
    # Read data
    filepath = f"00HOST_1_LIST_{slot}.bin"
    if not Path(filepath).exists():
        print(f"\n⚠ Data file not found: {filepath}")
        return
    
    print(f"\nReading {num_blocks} blocks from: {filepath}")
    raw_data = read_hour_file(filepath, "ANA", num_blocks=num_blocks)
    raw_channel = raw_data[:, channel]
    
    print(f"Raw data loaded: {raw_channel.shape[0]} samples")
    print(f"Raw value range: [{raw_channel.min()}, {raw_channel.max()}]")
    
    # Apply calibration
    print("\nApplying calibration...")
    try:
        calibrated = calibrate_channel(raw_channel, card, channel, cnv_directory=".")
        print(f"Calibrated value range: [{calibrated.min():.6f}, {calibrated.max():.6f}]")
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Statistics
    print("\nStatistics:")
    print(f"{'Metric':<15} {'Raw':<15} {'Calibrated':<15}")
    print("-" * 45)
    print(f"{'Mean':<15} {raw_channel.mean():<15.2f} {calibrated.mean():<15.6f}")
    print(f"{'Std Dev':<15} {raw_channel.std():<15.2f} {calibrated.std():<15.6f}")
    print(f"{'Min':<15} {raw_channel.min():<15} {calibrated.min():<15.6f}")
    print(f"{'Max':<15} {raw_channel.max():<15} {calibrated.max():<15.6f}")
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Time vector (10 kHz sampling)
    time = np.arange(len(raw_channel)) / 10000.0
    
    # Raw data
    ax1.plot(time, raw_channel, 'b-', linewidth=0.5)
    ax1.set_ylabel('Raw ADC Value')
    ax1.set_title(f'Slot {slot}, Channel {channel}: {var_name}')
    ax1.grid(True, alpha=0.3)
    
    # Calibrated data
    ax2.plot(time, calibrated, 'r-', linewidth=0.5)
    ax2.set_ylabel('Calibrated Value')
    ax2.set_xlabel('Time (s)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = f'calibration_slot{slot}_ch{channel}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {output_file}")
    
    return raw_channel, calibrated


def export_calibration_info(config, output_file="calibration_info.csv"):
    """Export all calibration parameters to CSV"""
    import csv
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Slot', 'Channel', 'Variable', 'Type', 
            'COEF_A', 'COEF_B', 'A', 'B', 'CNV_File'
        ])
        
        for card in config.cards:
            if card.card_type == 'ANA':
                for ch, (var, calib) in enumerate(zip(card.variable_names, card.calibrations)):
                    calib_type = 'Piecewise' if calib.cnv_file else 'Linear'
                    writer.writerow([
                        card.slot,
                        ch,
                        var,
                        calib_type,
                        calib.coef_a,
                        calib.coef_b,
                        calib.a,
                        calib.b,
                        calib.cnv_file if calib.cnv_file else ''
                    ])
    
    print(f"\nCalibration info exported to: {output_file}")


def main():
    """Main calibration demonstration"""
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 25 + "FEPC CALIBRATION DEMO" + " " * 32 + "║")
    print("╚" + "═" * 78 + "╝")
    
    cfg_file = "HOST_1_DATA.CFG"
    
    if not Path(cfg_file).exists():
        print(f"\n❌ Configuration file not found: {cfg_file}")
        print("Please provide the HOST_X_DATA.CFG file")
        return
    
    # Load configuration
    print(f"\nLoading configuration from: {cfg_file}")
    config = parse_cfg_file(cfg_file)
    
    # 1. Display calibration parameters
    display_calibration_info(config, verbose=True)
    
    # 2. Demonstrate linear calibration
    demonstrate_linear_calibration()
    
    # 3. Look for a CNV file to demonstrate
    cnv_files = list(Path(".").glob("*.CNV"))
    if cnv_files:
        print(f"\nFound {len(cnv_files)} CNV file(s)")
        demonstrate_cnv_calibration(str(cnv_files[0]))
    else:
        print("\n⚠ No CNV files found in current directory")
        print("CNV files contain piecewise calibration data")
    
    # 4. Export calibration info
    export_calibration_info(config)
    
    # 5. Apply calibration to real data if available
    analog_slots = config.get_analog_slots()
    if analog_slots and Path(f"00HOST_1_LIST_{analog_slots[0]}.bin").exists():
        apply_and_plot_calibration(config, slot=analog_slots[0], channel=0, num_blocks=500)
    else:
        print("\n⚠ No data files found for calibration demonstration")
        print("Place binary data files (XXHOST_Y_LIST_Z.bin) in the current directory")
    
    print("\n" + "=" * 80)
    print("Calibration demonstration complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
