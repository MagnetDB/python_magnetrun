"""
Example script to read FEPC kHz data

This demonstrates:
1. Reading the CFG file to get slot configuration
2. Loading data from specific slots
3. Accessing variables by name
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from fepc_reader import (
    parse_cfg_file,
    read_analog_block,
    read_hour_file,
)


def example_read_config():
    """Example: Read and display configuration"""
    print("=" * 60)
    print("EXAMPLE 1: Reading Configuration File")
    print("=" * 60)

    # Path to your CFG file
    cfg_path = "HOST_2_DATA.CFG"  # Adjust path as needed

    # Parse configuration
    config = parse_cfg_file(cfg_path)

    print(f"\nFEPC Name: {config.fepc_name}")
    print(f"Total Cards: {config.num_cards}")
    print(f"  Analog slots: {config.get_analog_slots()}")
    print(f"  Digital slots: {config.get_digital_slots()}")

    # Display info for each card
    for card in config.cards:
        print(f"\n{'=' * 50}")
        print(f"Slot {card.slot}: {card.card_type} Card")
        print(f"{'=' * 50}")
        print(f"  Sampling frequency: {card.sampling_freq} Hz")
        print(f"  Number of channels: {card.num_channels}")
        print(f"  Quench buffer: Pre={card.buffer_pre}s, Post={card.buffer_post}s")
        print(f"  Variables ({len(card.variable_names)}):")

        # Display first few variable names
        for i, var_name in enumerate(card.variable_names[:5], 1):
            print(f"    {i}. {var_name}")
        if len(card.variable_names) > 5:
            print(f"    ... and {len(card.variable_names) - 5} more")

    return config


def example_read_single_block():
    """Example: Read a single block from a file"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Reading Single Data Block")
    print("=" * 60)

    # Example: Read first block from an analog file
    filepath = "00HOST_2_LIST_0.bin"  # Hour 00, slot 0

    print(f"\nReading first block from: {filepath}")

    with open(filepath, "rb") as f:
        data, header = read_analog_block(f, block_idx=0)

    print(f"\nBlock data shape: {data.shape}")
    print(f"  Samples: {data.shape[0]}")
    print(f"  Channels: {data.shape[1]}")
    print(f"\nHeader values: {header['values']}")
    print("\nFirst sample (all 16 channels):")
    print(data[0, :])
    print("\nStatistics for channel 0:")
    print(f"  Min: {data[:, 0].min()}")
    print(f"  Max: {data[:, 0].max()}")
    print(f"  Mean: {data[:, 0].mean():.2f}")


def example_read_hour_data(config):
    """Example: Read complete hour of data for a specific slot"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Reading Complete Hour File")
    print("=" * 60)

    # Select a slot to read
    slot = 0  # First analog card
    hour = 0  # Hour 00

    filepath = f"{hour:02d}HOST_2_LIST_{slot}.bin"

    card = config.get_card_by_slot(slot)
    print(f"\nReading slot {slot} ({card.card_type} card)")
    print(f"  File: {filepath}")
    print(f"  Variables: {card.variable_names[:3]}...")

    # Read first 100 blocks (5 seconds of data) for quick demo
    # For full hour, use num_blocks=72000
    num_blocks = 100

    print(f"\nReading {num_blocks} blocks ({num_blocks * 50} samples)...")

    # Read data
    data = read_hour_file(filepath, card.card_type, num_blocks=num_blocks)

    print(f"\nData shape: {data.shape}")
    print(f"  Total samples: {data.shape[0]}")
    print(f"  Total channels: {data.shape[1]}")
    print(f"  Data type: {data.dtype}")
    print(f"  Memory size: {data.nbytes / 1024 / 1024:.2f} MB")

    return data, card


def example_extract_variable(config):
    """Example: Extract specific variable by name"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Extracting Specific Variable")
    print("=" * 60)

    # Find which slot contains a specific variable
    target_variable = "DUP1_V1"  # Example variable name

    for card in config.cards:
        if target_variable in card.variable_names:
            channel_idx = card.variable_names.index(target_variable)
            print(f"\nVariable '{target_variable}' found:")
            print(f"  Slot: {card.slot}")
            print(f"  Channel: {channel_idx}")
            print(f"  Card type: {card.card_type}")

            # Read the data for that slot (first 100 blocks for demo)
            filepath = f"00HOST_2_LIST_{card.slot}.bin"
            data = read_hour_file(filepath, card.card_type, num_blocks=100)

            # Extract the specific channel
            variable_data = data[:, channel_idx]

            print("\nVariable data:")
            print(f"  Shape: {variable_data.shape}")
            print(f"  Min: {variable_data.min()}")
            print(f"  Max: {variable_data.max()}")
            print(f"  Mean: {variable_data.mean():.2f}")

            return variable_data

    print(f"\nVariable '{target_variable}' not found in configuration")
    return None


def example_plot_data(data, card, num_samples=5000):
    """Example: Plot some channels"""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Plotting Data")
    print("=" * 60)

    # Plot first 3 channels
    num_channels_to_plot = min(3, data.shape[1])
    time = np.arange(num_samples) / 10000.0  # Convert to seconds (10 kHz sampling)

    fig, axes = plt.subplots(num_channels_to_plot, 1, figsize=(12, 8))
    if num_channels_to_plot == 1:
        axes = [axes]

    for i in range(num_channels_to_plot):
        axes[i].plot(time, data[:num_samples, i])
        axes[i].set_ylabel(card.variable_names[i])
        axes[i].grid(True, alpha=0.3)
        if i == num_channels_to_plot - 1:
            axes[i].set_xlabel("Time (s)")

    plt.suptitle(f"Slot {card.slot} - First {num_channels_to_plot} channels")
    plt.tight_layout()

    # Save plot
    output_path = f"fepc_data_slot_{card.slot}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")

    return output_path


def example_calibration():
    """Example: Apply calibration to analog data"""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Applying Calibration")
    print("=" * 60)

    from pathlib import Path

    from fepc_reader import (
        apply_calibration,
        calibrate_channel,
        parse_cfg_file,
        read_hour_file,
    )

    # Parse configuration
    cfg_path = "HOST_2_DATA.CFG"
    if not Path(cfg_path).exists():
        print(f"⚠ Configuration file not found: {cfg_path}")
        return

    config = parse_cfg_file(cfg_path)

    # Get first analog card
    analog_slots = config.get_analog_slots()
    if not analog_slots:
        print("No analog cards found")
        return

    slot = analog_slots[0]
    card = config.get_card_by_slot(slot)

    print(f"\nAnalyzing calibration for Slot {slot}")
    print(f"Card type: {card.card_type}")
    print(f"Variables: {card.variable_names[:3]}...")

    # Display calibration info for each channel
    print("\nCalibration parameters:")
    print(f"{'Ch':<4} {'Variable':<20} {'Type':<12} {'Parameters'}")
    print("-" * 70)

    for i, (var_name, calib) in enumerate(
        zip(card.variable_names[:5], card.calibrations[:5], strict=False)
    ):
        if calib.cnv_file:
            calib_type = "Piecewise"
            params = f"CNV: {calib.cnv_file}"
        else:
            calib_type = "Linear"
            params = (
                f"A={calib.a:.3e}, B={calib.b:.3e}, Ca={calib.coef_a:.3f}, Cb={calib.coef_b:.3f}"
            )
        print(f"{i:<4} {var_name:<20} {calib_type:<12} {params}")

    # Read some data
    filepath = f"00HOST_2_LIST_{slot}.bin"
    if not Path(filepath).exists():
        print(f"\n⚠ Data file not found: {filepath}")
        print("Calibration parameters displayed above are from CFG file.")
        return

    print(f"\nReading data from: {filepath}")
    raw_data = read_hour_file(filepath, "ANA", num_blocks=100)

    # Apply calibration to first channel
    channel_idx = 0
    print(f"\nApplying calibration to channel {channel_idx}: {card.variable_names[channel_idx]}")

    raw_channel = raw_data[:, channel_idx]
    print(f"Raw data range: [{raw_channel.min()}, {raw_channel.max()}]")

    # Try to calibrate
    try:
        calibrated = calibrate_channel(raw_channel, card, channel_idx, cnv_directory=".")
        print(f"Calibrated data range: [{calibrated.min():.3f}, {calibrated.max():.3f}]")

        # Show first few values
        print("\nFirst 5 samples:")
        print(f"{'Raw':<10} {'Calibrated':<15}")
        for i in range(5):
            print(f"{raw_channel[i]:<10} {calibrated[i]:<15.6f}")

    except (OSError, ValueError, RuntimeError) as e:
        print(f"Calibration error: {e}")
        print("Using linear calibration from CFG parameters")
        calib = card.calibrations[channel_idx]
        calibrated = apply_calibration(raw_channel, calib_info=calib)
        print(f"Calibrated data range: [{calibrated.min():.3f}, {calibrated.max():.3f}]")


def main():
    """Run all examples"""
    print("\n" + "#" * 60)
    print("# FEPC kHz Data Reader - Examples")
    print("#" * 60)

    try:
        # 1. Read configuration
        config = example_read_config()

        # Check if data files exist before proceeding
        sample_file = "00HOST_2_LIST_0.bin"
        has_data_files = Path(sample_file).exists()

        if not has_data_files:
            print(f"\n⚠ Warning: Data file '{sample_file}' not found")
            print("Place your data files in the same directory to run data reading examples")

            # Still run calibration example to show parameters
            print("\n" + "=" * 60)
            print("Running calibration parameter display...")
            example_calibration()
            return

        # 2. Read single block
        example_read_single_block()

        # 3. Read hour data
        data, card = example_read_hour_data(config)

        # 4. Extract specific variable
        example_extract_variable(config)

        # 5. Plot data
        example_plot_data(data, card)

        # 6. Calibration example
        example_calibration()

        print("\n" + "#" * 60)
        print("# Examples completed successfully!")
        print("#" * 60)

    except FileNotFoundError as e:
        print(f"\n⚠ Error: File not found - {e}")
        print("Make sure your CFG and data files are in the correct location")
    except (OSError, ValueError, RuntimeError) as e:
        print(f"\n⚠ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
