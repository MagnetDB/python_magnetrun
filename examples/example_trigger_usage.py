"""
Example script demonstrating trigger data reading and analysis

This shows:
1. Listing available triggers
2. Reading trigger metadata
3. Reading trigger data
4. Applying calibration
5. Plotting trigger events
6. Comparing PRE/POST windows
"""

from pathlib import Path

import matplotlib.pyplot as plt

from python_magnetrun.hybbrid.trigger.plot_trigger_data import plot_trigger_variable
from python_magnetrun.hybrid.trigger.trigger_reader import (
    create_time_array,
    find_trigger_directories,
    list_trigger_files,
    load_trigger_config,
    parse_trigger_directory,
    read_trigger_data,
)


def example_list_triggers():
    """Example 1: List all triggers in a directory"""
    print("=" * 70)
    print("EXAMPLE 1: Listing Trigger Directories")
    print("=" * 70)

    base_dir = Path("/data/hybrid")  # Adjust path as needed
    date = "2025-11-05"  # Adjust date as needed

    # Find all triggers for specific date
    triggers = find_trigger_directories(base_dir, date)

    print(f"\nFound {len(triggers)} trigger(s) for {date}:")
    for trigger_dir in triggers:
        print(f"  {trigger_dir.name}")

    return triggers


def example_trigger_info():
    """Example 2: Read trigger metadata"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Reading Trigger Metadata")
    print("=" * 70)

    trigger_dir = Path("/data/hybrid/trigger/TRIGGER_2025-11-05_08-16")  # Adjust path

    if not trigger_dir.exists():
        print(f"\nWarning: Trigger directory not found: {trigger_dir}")
        print("Please adjust the path in the script.")
        return None

    # Parse trigger info
    trigger_info = parse_trigger_directory(trigger_dir)

    print(f"\nTrigger: {trigger_info.trigger_name}")
    print(f"Timestamp: {trigger_info.timestamp}")
    print(f"Sample index: {trigger_info.sample_idx}")
    print(f"RT Block ID: {trigger_info.rtblock_id}")
    print(f"RT Block Phase: {trigger_info.rtblock_phase}")

    if trigger_info.trigger_approx_timestamp:
        print(f"Approx timestamp: {trigger_info.trigger_approx_timestamp}")

    print("\nData windows:")
    print(
        f"  PRE: {trigger_info.pre_samples} samples ({trigger_info.pre_samples / 10000:.1f}s)"
    )
    print(
        f"  POST: {trigger_info.post_samples} samples ({trigger_info.post_samples / 10000:.1f}s)"
    )
    print(
        f"  Total: {trigger_info.total_samples} samples ({trigger_info.total_samples / 10000:.1f}s)"
    )

    return trigger_dir


def example_list_files():
    """Example 3: List trigger binary files"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Listing Trigger Binary Files")
    print("=" * 70)

    trigger_dir = Path("/data/hybrid/trigger/TRIGGER_2025-11-05_08-16")  # Adjust path

    if not trigger_dir.exists():
        print(f"\nWarning: Trigger directory not found: {trigger_dir}")
        return

    # List files for each system
    for system in ["FEPC-LNCMI", "FEPC-AUX-LNCMI"]:
        files = list_trigger_files(trigger_dir, system)

        if files:
            print(f"\n{system}:")
            for tf in files:
                size_mb = tf.file_size / (1024**2)
                status = "✓" if tf.file_size == tf.expected_size else "⚠"
                print(
                    f"  {status} Slot {tf.slot} ({tf.card_type}): {tf.filepath.name} ({size_mb:.1f} MB)"
                )


def example_load_config():
    """Example 4: Load trigger configuration"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Loading Trigger Configuration")
    print("=" * 70)

    trigger_dir = Path("/data/hybrid/trigger/TRIGGER_2025-11-05_08-16")  # Adjust path
    system = "FEPC-LNCMI"

    if not trigger_dir.exists():
        print(f"\nWarning: Trigger directory not found: {trigger_dir}")
        return None

    # Load config
    config = load_trigger_config(trigger_dir, system)

    if config is None:
        print(f"Could not load config for {system}")
        return None

    print(f"\nFEPC Name: {config.fepc_name}")
    print(f"Number of cards: {config.num_cards}")
    print(f"Analog slots: {config.get_analog_slots()}")
    print(f"Digital slots: {config.get_digital_slots()}")

    # Display variables for each slot
    print("\nVariables by slot:")
    for card in config.cards:
        print(f"\n  Slot {card.slot} ({card.card_type}):")
        print(f"    Channels: {card.num_channels}")
        print(f"    Variables (first 5): {', '.join(card.variable_names[:5])}")
        if len(card.variable_names) > 5:
            print(f"    ... and {len(card.variable_names) - 5} more")

    return config


def example_read_variable():
    """Example 5: Read specific variable"""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Reading Specific Variable")
    print("=" * 70)

    trigger_dir = Path("/data/hybrid/trigger/TRIGGER_2025-11-05_08-16")  # Adjust path
    system = "FEPC-LNCMI"
    variable = "I_H1"  # Adjust variable name

    if not trigger_dir.exists():
        print(f"\nWarning: Trigger directory not found: {trigger_dir}")
        return None

    print(f"\nReading {variable} from {trigger_dir.name}...")

    try:
        # Read data
        data, timestamp, config = read_trigger_data(
            trigger_dir, system, variable_name=variable
        )

        print("\nData successfully read!")
        print(f"  Shape: {data.shape}")
        print(f"  Data type: {data.dtype}")
        print(f"  Trigger timestamp: {timestamp}")
        print(f"  Data range: [{data.min()}, {data.max()}]")
        print(f"  Mean: {data.mean():.3f}")
        print(f"  Std dev: {data.std():.3f}")

        # Show first few samples
        print("\nFirst 10 samples:")
        print(f"  {data[:10]}")

        return data, timestamp

    except (OSError, ValueError, RuntimeError) as e:
        print(f"\nError reading data: {e}")
        import traceback

        traceback.print_exc()
        return None


def example_read_slot():
    """Example 6: Read entire slot"""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Reading Entire Slot")
    print("=" * 70)

    trigger_dir = Path("/data/hybrid/trigger/TRIGGER_2025-11-05_08-16")  # Adjust path
    system = "FEPC-LNCMI"
    slot = 0  # Adjust slot number

    if not trigger_dir.exists():
        print(f"\nWarning: Trigger directory not found: {trigger_dir}")
        return None

    print(f"\nReading slot {slot} from {trigger_dir.name}...")

    try:
        # Read data
        data, timestamp, config = read_trigger_data(trigger_dir, system, slot=slot)

        print("\nData successfully read!")
        print(f"  Shape: {data.shape}")
        print(f"  Number of channels: {data.shape[1]}")
        print(f"  Trigger timestamp: {timestamp}")

        # Get card info
        card = config.get_card_by_slot(slot)
        print("\nCard info:")
        print(f"  Type: {card.card_type}")
        print(f"  Variables: {', '.join(card.variable_names[:5])}...")

        # Show statistics for each channel
        print("\nChannel statistics (first 5):")
        for i in range(min(5, data.shape[1])):
            var_name = card.variable_names[i]
            channel_data = data[:, i]
            print(
                f"  {i}: {var_name:12s} - mean: {channel_data.mean():10.3f}, "
                f"std: {channel_data.std():10.3f}, "
                f"range: [{channel_data.min():10.3f}, {channel_data.max():10.3f}]"
            )

        return data, timestamp

    except (OSError, ValueError, RuntimeError) as e:
        print(f"\nError reading data: {e}")
        import traceback

        traceback.print_exc()
        return None


def example_analyze_windows():
    """Example 7: Analyze PRE and POST windows"""
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Analyzing PRE and POST Windows")
    print("=" * 70)

    trigger_dir = Path("/data/hybrid/trigger/TRIGGER_2025-11-05_08-16")  # Adjust path
    system = "FEPC-LNCMI"
    variable = "I_H1"

    if not trigger_dir.exists():
        print(f"\nWarning: Trigger directory not found: {trigger_dir}")
        return

    print(f"\nAnalyzing {variable} PRE/POST windows...")

    try:
        # Read data
        data, timestamp, config = read_trigger_data(
            trigger_dir, system, variable_name=variable
        )

        # Split into PRE and POST
        trigger_info = parse_trigger_directory(trigger_dir)
        pre_samples = trigger_info.pre_samples

        pre_data = data[:pre_samples]
        post_data = data[pre_samples:]

        print("\nPRE window (before trigger):")
        print(f"  Samples: {len(pre_data)}")
        print(f"  Duration: {len(pre_data) / 10000:.1f}s")
        print(f"  Mean: {pre_data.mean():.3f}")
        print(f"  Std dev: {pre_data.std():.3f}")
        print(f"  Range: [{pre_data.min()}, {pre_data.max()}]")

        print("\nPOST window (after trigger):")
        print(f"  Samples: {len(post_data)}")
        print(f"  Duration: {len(post_data) / 10000:.1f}s")
        print(f"  Mean: {post_data.mean():.3f}")
        print(f"  Std dev: {post_data.std():.3f}")
        print(f"  Range: [{post_data.min()}, {post_data.max()}]")

        # Detect step change
        if abs(post_data.mean() - pre_data.mean()) > 3 * pre_data.std():
            print("\n⚠ Significant change detected at trigger!")
            print(f"  Delta: {post_data.mean() - pre_data.mean():.3f}")
        else:
            print("\n✓ No significant change detected")

    except (OSError, ValueError, RuntimeError) as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()


def example_plot_trigger():
    """Example 8: Plot trigger data"""
    print("\n" + "=" * 70)
    print("EXAMPLE 8: Plotting Trigger Data")
    print("=" * 70)

    trigger_dir = Path("/data/hybrid/trigger/TRIGGER_2025-11-05_08-16")  # Adjust path
    system = "FEPC-LNCMI"
    variable = "I_H1"

    if not trigger_dir.exists():
        print(f"\nWarning: Trigger directory not found: {trigger_dir}")
        return

    print(f"\nPlotting {variable}...")
    print("Note: This will display an interactive plot.")

    try:
        # Plot with calibration
        plot_trigger_variable(
            trigger_dir, system, variable, show_plot=True, apply_calib=True
        )

    except (OSError, ValueError, RuntimeError) as e:
        print(f"\nError plotting: {e}")
        import traceback

        traceback.print_exc()


def example_custom_analysis():
    """Example 9: Custom time window analysis"""
    print("\n" + "=" * 70)
    print("EXAMPLE 9: Custom Time Window Analysis")
    print("=" * 70)

    trigger_dir = Path("/data/hybrid/trigger/TRIGGER_2025-11-05_08-16")  # Adjust path
    system = "FEPC-LNCMI"
    variable = "I_H1"

    if not trigger_dir.exists():
        print(f"\nWarning: Trigger directory not found: {trigger_dir}")
        return

    print(f"\nAnalyzing custom time window for {variable}...")

    try:
        # Read data
        data, timestamp, config = read_trigger_data(
            trigger_dir, system, variable_name=variable
        )

        # Create time array
        time = create_time_array(len(data))

        # Get trigger info
        trigger_info = parse_trigger_directory(trigger_dir)
        pre_time = trigger_info.pre_samples / 10000.0  # 20s

        # Time relative to trigger
        time_rel = time - pre_time

        # Define custom window: -2s to +5s around trigger
        window_start = -2.0
        window_end = 5.0
        mask = (time_rel >= window_start) & (time_rel <= window_end)

        windowed_data = data[mask]
        windowed_time = time_rel[mask]

        print(f"\nCustom window: {window_start}s to {window_end}s")
        print(f"  Samples: {len(windowed_data)}")
        print(f"  Mean: {windowed_data.mean():.3f}")
        print(f"  Max: {windowed_data.max():.3f}")
        print(f"  Min: {windowed_data.min():.3f}")

        # Plot custom window
        plt.figure(figsize=(10, 5))
        plt.plot(windowed_time, windowed_data, linewidth=1)
        plt.axvline(0, color="red", linestyle="--", linewidth=2, label="Trigger")
        plt.xlabel("Time relative to trigger (s)")
        plt.ylabel(variable)
        plt.title(f"{variable} - Custom Window")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    except (OSError, ValueError, RuntimeError) as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Run all examples"""
    print("FEPC Trigger Data Reader - Example Usage")
    print("=" * 70)
    print("\nNote: These examples use placeholder paths.")
    print("Please adjust paths in the script to match your data location.")
    print("\nPress Enter to continue with each example...")

    # Example 1: List triggers
    try:
        triggers = example_list_triggers()
    except (OSError, ValueError, RuntimeError) as e:
        print(f"Example 1 failed: {e}")

    input("\nPress Enter to continue...")

    # Example 2: Trigger info
    try:
        trigger_dir = example_trigger_info()
    except (OSError, ValueError, RuntimeError) as e:
        print(f"Example 2 failed: {e}")
        trigger_dir = None

    if trigger_dir is None:
        print("\nRemaining examples skipped (trigger directory not found)")
        print("Please adjust the paths in the script to match your data location.")
        return

    input("\nPress Enter to continue...")

    # Example 3: List files
    try:
        example_list_files()
    except (OSError, ValueError, RuntimeError) as e:
        print(f"Example 3 failed: {e}")

    input("\nPress Enter to continue...")

    # Example 4: Load config
    try:
        config = example_load_config()
    except (OSError, ValueError, RuntimeError) as e:
        print(f"Example 4 failed: {e}")

    input("\nPress Enter to continue...")

    # Example 5: Read variable
    try:
        result = example_read_variable()
    except (OSError, ValueError, RuntimeError) as e:
        print(f"Example 5 failed: {e}")

    input("\nPress Enter to continue...")

    # Example 6: Read slot
    try:
        result = example_read_slot()
    except (OSError, ValueError, RuntimeError) as e:
        print(f"Example 6 failed: {e}")

    input("\nPress Enter to continue...")

    # Example 7: Analyze windows
    try:
        example_analyze_windows()
    except (OSError, ValueError, RuntimeError) as e:
        print(f"Example 7 failed: {e}")

    input("\nPress Enter to continue...")

    # Example 8: Plot (optional)
    response = input("\nWould you like to display plots? (y/n): ")
    if response.lower() == "y":
        try:
            example_plot_trigger()
        except (OSError, ValueError, RuntimeError) as e:
            print(f"Example 8 failed: {e}")

        input("\nPress Enter to continue...")

        try:
            example_custom_analysis()
        except (OSError, ValueError, RuntimeError) as e:
            print(f"Example 9 failed: {e}")

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
