"""
CFG File Analyzer - Read and display FEPC configuration

This tool reads the HOST_X_DATA.CFG file and displays:
- FEPC structure (number and type of cards)
- Slot assignments
- Variable names for each slot
- File naming conventions
"""

import argparse
import logging
import re
from pathlib import Path

from .fepc_reader import FEPCConfig, parse_cfg_file

logger = logging.getLogger(__name__)


def extract_host_number(cfg_path: str) -> str:
    """
    Extract HOST number from CFG filename

    Parameters:
    -----------
    cfg_path : str
        Path to HOST_X_DATA.CFG file

    Returns:
    --------
    str
        The HOST number (e.g., "1" from "HOST_1_DATA.CFG")
    """
    filename = Path(cfg_path).name
    match = re.search(r"HOST_(\d+)_DATA\.CFG", filename, re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Could not extract HOST number from filename: {filename}")


def analyze_cfg_file(cfg_path: str, verbose: bool = True) -> FEPCConfig:
    """
    Analyze CFG file and display structure

    Parameters:
    -----------
    cfg_path : str
        Path to HOST_X_DATA.CFG file
    verbose : bool
        If True, display detailed information

    Returns:
    --------
    config : FEPCConfig
        Parsed configuration object
    """
    if not Path(cfg_path).exists():
        raise FileNotFoundError(f"CFG file not found: {cfg_path}")

    logger.info(f"Reading configuration from: {cfg_path}")
    logger.info("=" * 80)

    # Extract HOST number from cfg filename
    host_number = extract_host_number(cfg_path)

    config = parse_cfg_file(cfg_path)
    config.host_number = host_number  # Set host_number attribute

    # Summary
    logger.info(f"\nFEPC NAME: {config.fepc_name}")
    logger.info(f"TOTAL CARDS: {config.num_cards}")

    analog_slots = config.get_analog_slots()
    digital_slots = config.get_digital_slots()

    logger.info("\nCARD DISTRIBUTION:")
    logger.info(f"  Analog cards (MIVA):  {len(analog_slots)} cards in slots {analog_slots}")
    logger.info(f"  Digital cards (MAD):  {len(digital_slots)} cards in slots {digital_slots}")

    # Files per day calculation
    files_per_hour = config.num_cards
    files_per_day = files_per_hour * 24
    logger.info("\nFILE GENERATION:")
    logger.info(f"  Files per hour: {files_per_hour}")
    logger.info(f"  Files per day:  {files_per_day}")

    # Detailed card information
    logger.debug("\n" + "=" * 80)
    logger.debug("DETAILED CARD INFORMATION")
    logger.debug("=" * 80)

    for card in config.cards:
        logger.debug(f"\n┌{'─' * 78}┐")
        logger.debug(f"│ SLOT {card.slot:2d} - {card.card_type:8s} CARD" + " " * 58 + "│")
        logger.debug(f"├{'─' * 78}┤")
        logger.debug(f"│ Sampling Frequency: {card.sampling_freq:,} Hz" + " " * 46 + "│")
        logger.debug(
            f"│ Quench Buffer:      Pre = {card.buffer_pre}s, Post = {card.buffer_post}s"
            + " " * 41
            + "│"
        )
        logger.debug(f"│ Number of Channels: {card.num_channels}" + " " * 56 + "│")
        logger.debug(f"├{'─' * 78}┤")
        logger.debug("│ FILE NAMING:" + " " * 66 + "│")
        logger.debug(
            f"│   XXHOST_{config.host_number}_LIST_{card.slot}.bin  (XX = hour 00-23)"
            + " " * (78 - 35 - len(config.host_number))
            + "│"
        )
        logger.debug(f"├{'─' * 78}┤")
        logger.debug(
            f"│ VARIABLES ({len(card.variable_names)} channels):"
            + " " * (78 - 22 - len(str(len(card.variable_names))))
            + "│"
        )

        # Display all variables with calibration info in columns
        if card.card_type == "ANA" and card.calibrations:
            # Show calibration type
            for i, (var, calib) in enumerate(
                zip(card.variable_names, card.calibrations, strict=False)
            ):
                if calib.cnv_file:
                    cal_info = f"[CNV: {calib.cnv_file[:20]}]"
                else:
                    cal_info = f"[Linear: A={calib.a:.2e}]"
                line = f"│   {i + 1:2d}. {var:25s} {cal_info}"
                line += " " * (78 - len(line)) + "│"
                logger.debug(line)
        else:
            # Display without calibration info
            vars_per_line = 3
            for i in range(0, len(card.variable_names), vars_per_line):
                var_group = card.variable_names[i : i + vars_per_line]
                var_strs = [f"{i + j + 1:2d}. {var:20s}" for j, var in enumerate(var_group)]
                line = "│   " + "  ".join(var_strs)
                line += " " * (78 - len(line)) + "│"
                logger.debug(line)

        logger.debug(f"└{'─' * 78}┘")

    logger.info("\n" + "=" * 80)
    return config


def display_slot_map(config: FEPCConfig) -> None:
    """
    Display a visual map of slots
    """
    logger.info("\nSLOT MAP:")
    logger.info("─" * 80)

    # Create slot map visualization
    logger.info("\n  Slot │ Type    │ Channels │ File Pattern")
    logger.info("  ─────┼─────────┼──────────┼─────────────────────────")

    for card in config.cards:
        file_pattern = f"XXHOST_{config.host_number}_LIST_{card.slot}.bin"
        logger.info(
            f"   {card.slot:2d}  │ {card.card_type:7s} │    {card.num_channels:2d}    │ {file_pattern}"
        )

    logger.info("─" * 80)
    logger.info("  (XX = hour from 00 to 23)")


def search_variable(config: FEPCConfig, var_name: str) -> None:
    """
    Search for a variable across all slots

    Parameters:
    -----------
    config : FEPCConfig
        Configuration object
    var_name : str
        Variable name to search for (partial match supported)
    """
    logger.info(f"\nSearching for variable: '{var_name}'")
    logger.info("=" * 80)

    found = False
    var_name_upper = var_name.upper()

    for card in config.cards:
        for channel_idx, var in enumerate(card.variable_names):
            if var_name_upper in var.upper():
                if not found:
                    logger.info(f"\n{'Slot':<6} {'Channel':<8} {'Variable Name':<30} {'Card Type'}")
                    logger.info("─" * 80)
                    found = True

                logger.info(f"{card.slot:<6} {channel_idx:<8} {var:<30} {card.card_type}")

    if not found:
        logger.info(f"No variables found matching '{var_name}'")
    else:
        logger.info("=" * 80)


def export_variable_list(config: FEPCConfig, output_file: str = "fepc_variables.csv") -> None:
    """
    Export complete variable list to CSV
    """
    import csv

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Slot", "Channel", "Variable Name", "Card Type", "Channels Total"])

        for card in config.cards:
            for channel_idx, var in enumerate(card.variable_names):
                writer.writerow([card.slot, channel_idx, var, card.card_type, card.num_channels])

    logger.info(f"\nVariable list exported to: {output_file}")


def main() -> FEPCConfig | None:
    """Main function"""
    logger.info("\n" + "╔" + "═" * 78 + "╗")
    logger.info("║" + " " * 25 + "FEPC CFG FILE ANALYZER" + " " * 31 + "║")
    logger.info("╚" + "═" * 78 + "╝")

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Analyze FEPC configuration file and display structure information"
    )
    parser.add_argument(
        "cfg_file",
        nargs="?",
        default="HOST_2_DATA.CFG",
        help="Path to HOST_X_DATA.CFG file (default: HOST_2_DATA.CFG)",
    )
    args = parser.parse_args()
    cfg_path = args.cfg_file

    try:
        # Parse and analyze
        config = analyze_cfg_file(cfg_path, verbose=True)

        # Display slot map
        display_slot_map(config)

        # Export variable list
        export_variable_list(config)

        # Interactive search
        logger.info("\n" + "=" * 80)
        logger.info("INTERACTIVE MODE")
        logger.info("=" * 80)
        logger.info("\nYou can search for variables by name.")
        logger.info("Examples to try:")
        logger.info("  - search_variable(config, 'DUP1')")
        logger.info("  - search_variable(config, 'V1')")
        logger.info("  - search_variable(config, 'DIGITAL')")

        # Example searches
        if len(config.cards) > 0 and len(config.cards[0].variable_names) > 0:
            # Search for first variable as example
            example_var = config.cards[0].variable_names[0][:4]
            search_variable(config, example_var)

        logger.info("\n" + "=" * 80)
        logger.info("Analysis complete!")
        logger.info("=" * 80)

        return config

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        return None
    except (OSError, ValueError, RuntimeError) as e:
        logger.error(f"Error analyzing CFG file: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    config = main()
