"""
FEPC kHz Data Reader
Reads configuration and binary data files from FEPC acquisition system
"""

import numpy as np
import struct
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class CardInfo:
    """Information about a single acquisition card"""
    slot: int
    card_type: str  # 'ANA' or 'DIG'
    sampling_freq: int
    buffer_pre: int
    buffer_post: int
    num_channels: int
    variable_names: List[str]


@dataclass
class FEPCConfig:
    """Complete FEPC configuration"""
    fepc_name: str
    num_cards: int
    cards: List[CardInfo]
    
    def get_card_by_slot(self, slot: int) -> CardInfo:
        """Get card information by slot number"""
        for card in self.cards:
            if card.slot == slot:
                return card
        raise ValueError(f"Slot {slot} not found")
    
    def get_analog_slots(self) -> List[int]:
        """Return list of analog card slots"""
        return [card.slot for card in self.cards if card.card_type == 'ANA']
    
    def get_digital_slots(self) -> List[int]:
        """Return list of digital card slots"""
        return [card.slot for card in self.cards if card.card_type == 'DIG']


def parse_cfg_file(cfg_path: str) -> FEPCConfig:
    """
    Parse HOST_X_DATA.CFG file to extract FEPC configuration
    
    Parameters:
    -----------
    cfg_path : str
        Path to the CFG file
        
    Returns:
    --------
    FEPCConfig : Configuration object with all card information
    """
    with open(cfg_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    # First line contains the header with FEPC info
    header = lines[0]
    
    # Parse header to get FEPC name and basic structure
    # Format: FEPC_NAME;freq;pre;post;card_info;...
    parts = header.split(';')
    
    # Extract FEPC name (usually first element before numbers)
    fepc_name = parts[0].split('|')[0] if '|' in parts[0] else parts[0]
    
    # Count number of data lines (exclude header)
    num_data_lines = len(lines) - 1
    
    # Parse each card's information
    cards = []
    
    for line_idx, line in enumerate(lines[1:], start=1):
        # Each line represents one card's variables
        # Format: VAR1;VAR2;VAR3;...
        variables = [v.strip() for v in line.split(';') if v.strip()]
        
        if not variables:
            continue
            
        # Determine card type and slot from line position
        # Lines 2-7 (index 1-6): analog cards
        # Lines 8-9 (index 7-8): digital cards
        slot = line_idx - 1
        
        if line_idx <= 6:  # Analog cards
            card_type = 'ANA'
            num_channels = 16
        else:  # Digital cards
            card_type = 'DIG'
            num_channels = 32
        
        # Extract info from header (assuming 10kHz sampling)
        card = CardInfo(
            slot=slot,
            card_type=card_type,
            sampling_freq=10000,
            buffer_pre=20,
            buffer_post=50,
            num_channels=num_channels,
            variable_names=variables[:num_channels]
        )
        cards.append(card)
    
    config = FEPCConfig(
        fepc_name=fepc_name,
        num_cards=len(cards),
        cards=cards
    )
    
    return config


def read_analog_block(file, block_idx: int, endian: str = 'big') -> Tuple[np.ndarray, Dict]:
    """
    Read one analog data block (1614 bytes)
    
    Block structure:
    - Header: 7 × 16 bits = 14 bytes
    - Data: 50 samples × 16 channels × 16 bits = 1600 bytes
    
    Parameters:
    -----------
    file : file object
        Open binary file
    block_idx : int
        Block index to read
    endian : str
        Endianness: 'big' or 'little' (default: 'big')
    
    Returns:
    --------
    data : np.ndarray
        Shape (50, 16) - 50 samples, 16 channels, int16 values
    header : dict
        Header information
    """
    HEADER_SIZE = 14  # bytes
    DATA_SIZE = 1600  # bytes
    BLOCK_SIZE = 1614  # bytes
    
    # Seek to block position
    file.seek(block_idx * BLOCK_SIZE)
    
    # Read header (7 × uint16)
    endian_char = '>' if endian == 'big' else '<'
    header_bytes = file.read(HEADER_SIZE)
    header_data = struct.unpack(f'{endian_char}7H', header_bytes)
    
    header = {
        'values': header_data
    }
    
    # Read data (50 samples × 16 channels)
    data_bytes = file.read(DATA_SIZE)
    dtype = np.dtype(np.int16).newbyteorder('>' if endian == 'big' else '<')
    data = np.frombuffer(data_bytes, dtype=dtype).reshape(50, 16)
    
    return data, header


def read_digital_block(file, block_idx: int, endian: str = 'big') -> Tuple[np.ndarray, Dict]:
    """
    Read one digital data block (212 bytes)
    
    Block structure:
    - Header: 6 × 16 bits = 12 bytes
    - Data: 50 samples × 32 bits = 200 bytes
    
    Parameters:
    -----------
    file : file object
        Open binary file
    block_idx : int
        Block index to read
    endian : str
        Endianness: 'big' or 'little' (default: 'big')
    
    Returns:
    --------
    data : np.ndarray
        Shape (50, 32) - 50 samples, 32 channels, boolean values
    header : dict
        Header information
    """
    HEADER_SIZE = 12  # bytes
    DATA_SIZE = 200  # bytes
    BLOCK_SIZE = 212  # bytes
    
    # Seek to block position
    file.seek(block_idx * BLOCK_SIZE)
    
    # Read header (6 × uint16)
    endian_char = '>' if endian == 'big' else '<'
    header_bytes = file.read(HEADER_SIZE)
    header_data = struct.unpack(f'{endian_char}6H', header_bytes)
    
    header = {
        'values': header_data
    }
    
    # Read data (50 samples × 32 bits)
    # Each sample is stored as 2 × 16 bits
    data_bytes = file.read(DATA_SIZE)
    dtype = np.dtype(np.uint16).newbyteorder('>' if endian == 'big' else '<')
    data_uint16 = np.frombuffer(data_bytes, dtype=dtype).reshape(50, 2)
    
    # Convert to individual bits (32 channels per sample)
    data = np.zeros((50, 32), dtype=bool)
    for sample_idx in range(50):
        # First 16 bits (channels 0-15)
        word1 = data_uint16[sample_idx, 0]
        for bit in range(16):
            data[sample_idx, bit] = bool((word1 >> bit) & 1)
        
        # Second 16 bits (channels 16-31)
        word2 = data_uint16[sample_idx, 1]
        for bit in range(16):
            data[sample_idx, 16 + bit] = bool((word2 >> bit) & 1)
    
    return data, header


def read_hour_file(filepath: str, card_type: str, 
                   num_blocks: int = 72000, endian: str = 'big') -> np.ndarray:
    """
    Read complete hour file (all blocks)
    
    Parameters:
    -----------
    filepath : str
        Path to the binary file
    card_type : str
        'ANA' for analog or 'DIG' for digital
    num_blocks : int
        Number of blocks to read (default 72000 = 1 hour)
    endian : str
        Endianness: 'big' or 'little' (default: 'big')
        
    Returns:
    --------
    data : np.ndarray
        For analog: shape (3600000, 16) - 3.6M samples, 16 channels
        For digital: shape (3600000, 32) - 3.6M samples, 32 channels
    """
    num_channels = 16 if card_type == 'ANA' else 32
    total_samples = num_blocks * 50
    
    # Pre-allocate array
    data = np.zeros((total_samples, num_channels), 
                    dtype=np.int16 if card_type == 'ANA' else bool)
    
    with open(filepath, 'rb') as f:
        for block_idx in range(num_blocks):
            if card_type == 'ANA':
                block_data, _ = read_analog_block(f, block_idx, endian)
            else:
                block_data, _ = read_digital_block(f, block_idx, endian)
            
            # Store in output array
            start_idx = block_idx * 50
            end_idx = start_idx + 50
            data[start_idx:end_idx, :] = block_data
    
    return data


def load_calibration(cnv_path: str) -> Dict:
    """
    Load calibration file (.CNV) for converting raw values to physical units
    
    Returns:
    --------
    calib : dict
        Calibration parameters
    """
    # Placeholder - actual implementation depends on .CNV file format
    # Typically contains linear or piecewise linear calibration coefficients
    calib = {
        'type': 'linear',  # or 'piecewise'
        'coefficients': []
    }
    return calib


def apply_calibration(raw_data: np.ndarray, calib: Dict) -> np.ndarray:
    """
    Apply calibration to convert raw int16 values to physical units
    
    Parameters:
    -----------
    raw_data : np.ndarray
        Raw data from analog channels
    calib : dict
        Calibration parameters
        
    Returns:
    --------
    physical_data : np.ndarray
        Calibrated data in physical units
    """
    # Placeholder implementation
    # Actual calibration depends on .CNV file format
    return raw_data.astype(float)


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Read FEPC kHz acquisition data files',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to HOST_X_DATA.CFG configuration file'
    )
    
    parser.add_argument(
        '--datafile', '-d',
        type=str,
        help='Path to binary data file'
    )
    
    parser.add_argument(
        '--card-type', '-t',
        type=str,
        choices=['ANA', 'DIG'],
        default='ANA',
        help='Card type: ANA (analog) or DIG (digital), default: ANA'
    )
    
    parser.add_argument(
        '--num-blocks', '-n',
        type=int,
        default=72000,
        help='Number of blocks to read (default: 72000 = 1 hour)'
    )
    
    parser.add_argument(
        '--endian', '-e',
        type=str,
        choices=['big', 'little'],
        default='big',
        help='Endianness of binary data: big or little (default: big)'
    )
    
    args = parser.parse_args()
    
    # Parse configuration if provided
    if args.config:
        config = parse_cfg_file(args.config)
        print(f"FEPC: {config.fepc_name}")
        print(f"Number of cards: {config.num_cards}")
        
        for card in config.cards:
            print(f"\nSlot {card.slot}: {card.card_type} card")
            print(f"  Sampling frequency: {card.sampling_freq} Hz")
            print(f"  Number of channels: {card.num_channels}")
            print(f"  Variables: {', '.join(card.variable_names[:5])}{', ...' if len(card.variable_names) > 5 else ''}")
    
    # Read data file if provided
    if args.datafile:
        print(f"\nReading data file: {args.datafile}")
        print(f"Card type: {args.card_type}")
        print(f"Endianness: {args.endian}")
        print(f"Number of blocks: {args.num_blocks}")
        
        data = read_hour_file(args.datafile, args.card_type, args.num_blocks, args.endian)
        print(f"\nData shape: {data.shape}")
        print(f"Data type: {data.dtype}")
        
        if args.card_type == 'ANA':
            print(f"Data range: [{data.min()}, {data.max()}]")
            print(f"First 5 samples (channel 0): {data[:5, 0]}")
    
    if not args.config and not args.datafile:
        print("FEPC Reader module loaded successfully!")
        print("\nUsage examples:")
        print("  Parse config:  python fepc_reader.py --config HOST_2_DATA.CFG")
        print("  Read data:     python fepc_reader.py --datafile data.bin --card-type ANA --endian big")
        print("  Full example:  python fepc_reader.py -c HOST_2_DATA.CFG -d data.bin -t ANA -n 1000 -e big")
        parser.print_help()
