"""
Quick Test Script - Verify CFG File Parsing

This script provides a simple way to test if the CFG parser works with your file.
"""

import sys
from pathlib import Path


def test_cfg_parsing(cfg_path: str) -> bool:
    """Test parsing a CFG file"""
    print("=" * 70)
    print("FEPC CFG File Parser - Quick Test")
    print("=" * 70)
    print(f"\nTesting file: {cfg_path}")

    # Check file exists
    if not Path(cfg_path).exists():
        print(f"❌ ERROR: File not found: {cfg_path}")
        return False

    print("✓ File found")

    # Try to parse
    try:
        from fepc_reader import parse_cfg_file

        print("\nParsing configuration...")
        config = parse_cfg_file(cfg_path)

        print("✓ Parsing successful!")

        # Display summary
        print("\n" + "-" * 70)
        print("CONFIGURATION SUMMARY")
        print("-" * 70)
        print(f"FEPC Name:    {config.fepc_name}")
        print(f"Total Cards:  {config.num_cards}")
        print(f"Analog Slots: {config.get_analog_slots()}")
        print(f"Digital Slots: {config.get_digital_slots()}")

        print("\n" + "-" * 70)
        print("SLOT DETAILS")
        print("-" * 70)

        for card in config.cards:
            print(f"\nSlot {card.slot}: {card.card_type} card")
            print(f"  - Channels: {card.num_channels}")
            print(f"  - Variables: {len(card.variable_names)}")
            if card.variable_names:
                print(f"  - First variable: {card.variable_names[0]}")
                print(f"  - Last variable: {card.variable_names[-1]}")

        print("\n" + "=" * 70)
        print("✓ TEST PASSED - Configuration parsed successfully!")
        print("=" * 70)

        return True

    except (OSError, ValueError, RuntimeError) as e:
        print(f"\n❌ ERROR during parsing: {e}")
        import traceback

        print("\nFull traceback:")
        traceback.print_exc()
        return False


def show_raw_file(cfg_path: str, num_lines: int = 20) -> None:
    """Display first few lines of CFG file"""
    print("\n" + "=" * 70)
    print(f"RAW FILE CONTENT (first {num_lines} lines)")
    print("=" * 70)

    try:
        with open(cfg_path, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f, 1):
                if i > num_lines:
                    print("...")
                    break
                print(f"{i:3d}: {line.rstrip()}")
    except OSError as e:
        print(f"Error reading file: {e}")


if __name__ == "__main__":
    # Get CFG file path from command line or use default
    if len(sys.argv) > 1:
        cfg_file = sys.argv[1]
    else:
        cfg_file = "HOST_1_DATA.CFG"
        print(f"No file specified, using default: {cfg_file}")
        print("Usage: python test_cfg_parser.py <path_to_cfg_file>\n")

    # Show raw content first
    if Path(cfg_file).exists():
        show_raw_file(cfg_file, num_lines=15)

    # Test parsing
    success = test_cfg_parsing(cfg_file)

    if success:
        print("\n✓ Your CFG file is compatible with this reader!")
        print("\nNext steps:")
        print("  1. Use cfg_analyzer.py for detailed analysis")
        print("  2. Use example_fepc_usage.py to read data files")
        print("  3. Check README.md for full documentation")
    else:
        print("\n❌ There was an issue parsing your CFG file.")
        print("\nTroubleshooting:")
        print("  1. Check the raw file content above")
        print("  2. Verify the file format matches the expected structure")
        print("  3. Check for encoding issues or special characters")
        print("  4. You may need to adjust the parse_cfg_file() function")
