"""
CFG File Analyzer - Read and display FEPC configuration

This tool reads the HOST_X_DATA.CFG file and displays:
- FEPC structure (number and type of cards)
- Slot assignments
- Variable names for each slot
- File naming conventions
"""

import sys
from pathlib import Path
from fepc_reader import parse_cfg_file


def analyze_cfg_file(cfg_path: str, verbose: bool = True):
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
    
    print(f"Reading configuration from: {cfg_path}")
    print("=" * 80)
    
    config = parse_cfg_file(cfg_path)
    
    # Summary
    print(f"\nFEPC NAME: {config.fepc_name}")
    print(f"TOTAL CARDS: {config.num_cards}")
    
    analog_slots = config.get_analog_slots()
    digital_slots = config.get_digital_slots()
    
    print(f"\nCARD DISTRIBUTION:")
    print(f"  Analog cards (MIVA):  {len(analog_slots)} cards in slots {analog_slots}")
    print(f"  Digital cards (MAD):  {len(digital_slots)} cards in slots {digital_slots}")
    
    # Files per day calculation
    files_per_hour = config.num_cards
    files_per_day = files_per_hour * 24
    print(f"\nFILE GENERATION:")
    print(f"  Files per hour: {files_per_hour}")
    print(f"  Files per day:  {files_per_day}")
    
    # Detailed card information
    if verbose:
        print("\n" + "=" * 80)
        print("DETAILED CARD INFORMATION")
        print("=" * 80)
        
        for card in config.cards:
            print(f"\n┌{'─' * 78}┐")
            print(f"│ SLOT {card.slot:2d} - {card.card_type:8s} CARD" + " " * 58 + "│")
            print(f"├{'─' * 78}┤")
            print(f"│ Sampling Frequency: {card.sampling_freq:,} Hz" + " " * 46 + "│")
            print(f"│ Quench Buffer:      Pre = {card.buffer_pre}s, Post = {card.buffer_post}s" + " " * 41 + "│")
            print(f"│ Number of Channels: {card.num_channels}" + " " * 56 + "│")
            print(f"├{'─' * 78}┤")
            print(f"│ FILE NAMING:" + " " * 66 + "│")
            print(f"│   XXHOST_1_LIST_{card.slot}.bin  (XX = hour 00-23)" + " " * 38 + "│")
            print(f"├{'─' * 78}┤")
            print(f"│ VARIABLES ({len(card.variable_names)} channels):" + " " * (78 - 22 - len(str(len(card.variable_names)))) + "│")
            
            # Display all variables with calibration info in columns
            if card.card_type == 'ANA' and card.calibrations:
                # Show calibration type
                for i, (var, calib) in enumerate(zip(card.variable_names, card.calibrations)):
                    if calib.cnv_file:
                        cal_info = f"[CNV: {calib.cnv_file[:20]}]"
                    else:
                        cal_info = f"[Linear: A={calib.a:.2e}]"
                    line = f"│   {i+1:2d}. {var:25s} {cal_info}"
                    line += " " * (78 - len(line)) + "│"
                    print(line)
            else:
                # Display without calibration info
                vars_per_line = 3
                for i in range(0, len(card.variable_names), vars_per_line):
                    var_group = card.variable_names[i:i+vars_per_line]
                    var_strs = [f"{i+j+1:2d}. {var:20s}" for j, var in enumerate(var_group)]
                    line = "│   " + "  ".join(var_strs)
                    line += " " * (78 - len(line)) + "│"
                    print(line)
            
            print(f"└{'─' * 78}┘")
    
    print("\n" + "=" * 80)
    return config


def display_slot_map(config):
    """
    Display a visual map of slots
    """
    print("\nSLOT MAP:")
    print("─" * 80)
    
    # Create slot map visualization
    print("\n  Slot │ Type    │ Channels │ File Pattern")
    print("  ─────┼─────────┼──────────┼─────────────────────────")
    
    for card in config.cards:
        file_pattern = f"XXHOST_1_LIST_{card.slot}.bin"
        print(f"   {card.slot:2d}  │ {card.card_type:7s} │    {card.num_channels:2d}    │ {file_pattern}")
    
    print("─" * 80)
    print("  (XX = hour from 00 to 23)")


def search_variable(config, var_name: str):
    """
    Search for a variable across all slots
    
    Parameters:
    -----------
    config : FEPCConfig
        Configuration object
    var_name : str
        Variable name to search for (partial match supported)
    """
    print(f"\nSearching for variable: '{var_name}'")
    print("=" * 80)
    
    found = False
    var_name_upper = var_name.upper()
    
    for card in config.cards:
        for channel_idx, var in enumerate(card.variable_names):
            if var_name_upper in var.upper():
                if not found:
                    print(f"\n{'Slot':<6} {'Channel':<8} {'Variable Name':<30} {'Card Type'}")
                    print("─" * 80)
                    found = True
                
                print(f"{card.slot:<6} {channel_idx:<8} {var:<30} {card.card_type}")
    
    if not found:
        print(f"No variables found matching '{var_name}'")
    else:
        print("=" * 80)


def export_variable_list(config, output_file: str = "fepc_variables.csv"):
    """
    Export complete variable list to CSV
    """
    import csv
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Slot', 'Channel', 'Variable Name', 'Card Type', 'Channels Total'])
        
        for card in config.cards:
            for channel_idx, var in enumerate(card.variable_names):
                writer.writerow([
                    card.slot,
                    channel_idx,
                    var,
                    card.card_type,
                    card.num_channels
                ])
    
    print(f"\nVariable list exported to: {output_file}")


def main():
    """Main function"""
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 25 + "FEPC CFG FILE ANALYZER" + " " * 31 + "║")
    print("╚" + "═" * 78 + "╝")
    
    # Check command line arguments
    if len(sys.argv) > 1:
        cfg_path = sys.argv[1]
    else:
        # Default path
        cfg_path = "HOST_1_DATA.CFG"
        print(f"\nNo CFG file specified, using default: {cfg_path}")
        print("Usage: python cfg_analyzer.py <path_to_cfg_file>")
    
    try:
        # Parse and analyze
        config = analyze_cfg_file(cfg_path, verbose=True)
        
        # Display slot map
        display_slot_map(config)
        
        # Export variable list
        export_variable_list(config)
        
        # Interactive search
        print("\n" + "=" * 80)
        print("INTERACTIVE MODE")
        print("=" * 80)
        print("\nYou can search for variables by name.")
        print("Examples to try:")
        print("  - search_variable(config, 'DUP1')")
        print("  - search_variable(config, 'V1')")
        print("  - search_variable(config, 'DIGITAL')")
        
        # Example searches
        if len(config.cards) > 0 and len(config.cards[0].variable_names) > 0:
            # Search for first variable as example
            example_var = config.cards[0].variable_names[0][:4]
            search_variable(config, example_var)
        
        print("\n" + "=" * 80)
        print("Analysis complete!")
        print("=" * 80)
        
        return config
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        return None
    except Exception as e:
        print(f"\n❌ Error analyzing CFG file: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    config = main()
