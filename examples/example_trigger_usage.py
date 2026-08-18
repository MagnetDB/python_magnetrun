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

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from plot_trigger_data import plot_trigger_variable

from python_magnetrun.cli_args import create_hybrid_parser
from python_magnetrun.hybrid.trigger.trigger_reader import (
    create_time_array,
    find_trigger_directories,
    list_trigger_files,
    load_trigger_config,
    parse_trigger_directory,
    read_trigger_data,
)


def example_list_triggers(base_dir: Path, date: str):
    """Example 1: List all triggers in a directory"""
    print("=" * 70)
    print("EXAMPLE 1: Listing Trigger Directories")
    print("=" * 70)

    triggers = find_trigger_directories(base_dir, date)

    print(f"\nFound {len(triggers)} trigger(s) for {date}:")
    for trigger_dir in triggers:
        print(f"  {trigger_dir.name}")

    return triggers


def example_trigger_info(trigger_dir: Path):
    """Example 2: Read trigger metadata"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Reading Trigger Metadata")
    print("=" * 70)

    if not trigger_dir.exists():
        print(f"\nWarning: Trigger directory not found: {trigger_dir}")
        print("Please adjust the path or --hybrid_date argument.")
        return None

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


def example_list_files(trigger_dir: Path):
    """Example 3: List trigger binary files"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Listing Trigger Binary Files")
    print("=" * 70)

    if not trigger_dir.exists():
        print(f"\nWarning: Trigger directory not found: {trigger_dir}")
        return

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


def example_load_config(trigger_dir: Path, system: str):
    """Example 4: Load trigger configuration"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Loading Trigger Configuration")
    print("=" * 70)

    if not trigger_dir.exists():
        print(f"\nWarning: Trigger directory not found: {trigger_dir}")
        return None

    config = load_trigger_config(trigger_dir, system)

    if config is None:
        print(f"Could not load config for {system}")
        return None

    print(f"\nFEPC Name: {config.fepc_name}")
    print(f"Number of cards: {config.num_cards}")
    print(f"Analog slots: {config.get_analog_slots()}")
    print(f"Digital slots: {config.get_digital_slots()}")

    print("\nVariables by slot:")
    for card in config.cards:
        print(f"\n  Slot {card.slot} ({card.card_type}):")
        print(f"    Channels: {card.num_channels}")
        print(f"    Variables (first 5): {', '.join(card.variable_names[:5])}")
        if len(card.variable_names) > 5:
            print(f"    ... and {len(card.variable_names) - 5} more")

    return config


def example_read_variable(trigger_dir: Path, system: str, variable: str):
    """Example 5: Read specific variable"""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Reading Specific Variable")
    print("=" * 70)

    if not trigger_dir.exists():
        print(f"\nWarning: Trigger directory not found: {trigger_dir}")
        return None

    print(f"\nReading {variable} from {trigger_dir.name}...")

    try:
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

        print("\nFirst 10 samples:")
        print(f"  {data[:10]}")

        return data, timestamp

    except (OSError, ValueError, RuntimeError) as e:
        print(f"\nError reading data: {e}")
        import traceback

        traceback.print_exc()
        return None


def example_read_slot(trigger_dir: Path, system: str, slot: int = 0):
    """Example 6: Read entire slot"""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Reading Entire Slot")
    print("=" * 70)

    if not trigger_dir.exists():
        print(f"\nWarning: Trigger directory not found: {trigger_dir}")
        return None

    print(f"\nReading slot {slot} from {trigger_dir.name}...")

    try:
        data, timestamp, config = read_trigger_data(trigger_dir, system, slot=slot)

        print("\nData successfully read!")
        print(f"  Shape: {data.shape}")
        print(f"  Number of channels: {data.shape[1]}")
        print(f"  Trigger timestamp: {timestamp}")

        card = config.get_card_by_slot(slot)
        print("\nCard info:")
        print(f"  Type: {card.card_type}")
        print(f"  Variables: {', '.join(card.variable_names[:5])}...")

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


def example_analyze_windows(trigger_dir: Path, system: str, variable: str):
    """Example 7: Analyze PRE and POST windows"""
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Analyzing PRE and POST Windows")
    print("=" * 70)

    if not trigger_dir.exists():
        print(f"\nWarning: Trigger directory not found: {trigger_dir}")
        return

    print(f"\nAnalyzing {variable} PRE/POST windows...")

    try:
        data, timestamp, config = read_trigger_data(
            trigger_dir, system, variable_name=variable
        )

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

        if abs(post_data.mean() - pre_data.mean()) > 3 * pre_data.std():
            print("\n⚠ Significant change detected at trigger!")
            print(f"  Delta: {post_data.mean() - pre_data.mean():.3f}")
        else:
            print("\n✓ No significant change detected")

    except (OSError, ValueError, RuntimeError) as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()


def example_plot_trigger(trigger_dir: Path, system: str, variable: str):
    """Example 8: Plot trigger data"""
    print("\n" + "=" * 70)
    print("EXAMPLE 8: Plotting Trigger Data")
    print("=" * 70)

    if not trigger_dir.exists():
        print(f"\nWarning: Trigger directory not found: {trigger_dir}")
        return

    print(f"\nPlotting {variable}...")
    print("Note: This will display an interactive plot.")

    try:
        plot_trigger_variable(
            trigger_dir, system, variable, show_plot=True, apply_calib=True
        )

    except (OSError, ValueError, RuntimeError) as e:
        print(f"\nError plotting: {e}")
        import traceback

        traceback.print_exc()


def example_custom_analysis(trigger_dir: Path, system: str, variable: str):
    """Example 9: Custom time window analysis"""
    print("\n" + "=" * 70)
    print("EXAMPLE 9: Custom Time Window Analysis")
    print("=" * 70)

    if not trigger_dir.exists():
        print(f"\nWarning: Trigger directory not found: {trigger_dir}")
        return

    print(f"\nAnalyzing custom time window for {variable}...")

    try:
        data, timestamp, config = read_trigger_data(
            trigger_dir, system, variable_name=variable
        )

        time = create_time_array(len(data))
        trigger_info = parse_trigger_directory(trigger_dir)
        pre_time = trigger_info.pre_samples / 10000.0

        time_rel = time - pre_time

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
    hybrid_parser = create_hybrid_parser()
    parser = argparse.ArgumentParser(
        description="FEPC Trigger Data Reader - Example Usage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[hybrid_parser],
        epilog="""
Examples:
  # Run all examples for a specific date (picks first trigger found)
  python example_trigger_usage.py --hybrid_date 2025-11-05 --hybrid_datadir /data/hybrid

  # Specify a trigger directory directly
  python example_trigger_usage.py --trigger_dir /data/hybrid/trigger/TRIGGER__2025-11-05__08-16

  # Override default variable
  python example_trigger_usage.py --hybrid_date 2025-11-05 --variable I_BOB
        """,
    )
    parser.add_argument(
        "--trigger_dir",
        type=Path,
        default=None,
        help="Path to a specific trigger directory (overrides --hybrid_date lookup)",
    )
    parser.add_argument(
        "--variable",
        type=str,
        default="I_H1",
        help="Variable name for examples (default: I_H1)",
    )
    parser.add_argument(
        "--slot",
        type=int,
        default=0,
        help="Slot number for slot-read example (default: 0)",
    )

    args = parser.parse_args()

    # Resolve trigger directory
    trigger_dir: Path | None = args.trigger_dir

    if trigger_dir is None:
        if not args.hybrid_datadir or not args.hybrid_date:
            parser.error(
                "Provide either --trigger_dir or both --hybrid_datadir and --hybrid_date"
            )
        triggers = find_trigger_directories(Path(args.hybrid_datadir), args.hybrid_date)
        if not triggers:
            parser.error(
                f"No trigger directories found for {args.hybrid_date} in {args.hybrid_datadir}"
            )
        trigger_dir = triggers[0]
        print(f"Using trigger directory: {trigger_dir}")

    base_dir = trigger_dir.parent.parent  # …/hybrid
    date = trigger_dir.parent.name if trigger_dir.parent.name != "trigger" else (args.hybrid_date or "")
    system = args.fepc_system
    variable = args.variable

    print("FEPC Trigger Data Reader - Example Usage")
    print("=" * 70)
    print(f"\nBase dir : {base_dir}")
    print(f"Date     : {date}")
    print(f"Trigger  : {trigger_dir.name}")
    print(f"System   : {system}")
    print(f"Variable : {variable}")
    print("\nPress Enter to continue with each example...")

    try:
        example_list_triggers(base_dir, date)
    except (OSError, ValueError, RuntimeError) as e:
        print(f"Example 1 failed: {e}")

    input("\nPress Enter to continue...")

    try:
        example_trigger_info(trigger_dir)
    except (OSError, ValueError, RuntimeError) as e:
        print(f"Example 2 failed: {e}")

    input("\nPress Enter to continue...")

    try:
        example_list_files(trigger_dir)
    except (OSError, ValueError, RuntimeError) as e:
        print(f"Example 3 failed: {e}")

    input("\nPress Enter to continue...")

    try:
        example_load_config(trigger_dir, system)
    except (OSError, ValueError, RuntimeError) as e:
        print(f"Example 4 failed: {e}")

    input("\nPress Enter to continue...")

    try:
        example_read_variable(trigger_dir, system, variable)
    except (OSError, ValueError, RuntimeError) as e:
        print(f"Example 5 failed: {e}")

    input("\nPress Enter to continue...")

    try:
        example_read_slot(trigger_dir, system, args.slot)
    except (OSError, ValueError, RuntimeError) as e:
        print(f"Example 6 failed: {e}")

    input("\nPress Enter to continue...")

    try:
        example_analyze_windows(trigger_dir, system, variable)
    except (OSError, ValueError, RuntimeError) as e:
        print(f"Example 7 failed: {e}")

    input("\nPress Enter to continue...")

    response = input("\nWould you like to display plots? (y/n): ")
    if response.lower() == "y":
        try:
            example_plot_trigger(trigger_dir, system, variable)
        except (OSError, ValueError, RuntimeError) as e:
            print(f"Example 8 failed: {e}")

        input("\nPress Enter to continue...")

        try:
            example_custom_analysis(trigger_dir, system, variable)
        except (OSError, ValueError, RuntimeError) as e:
            print(f"Example 9 failed: {e}")

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
