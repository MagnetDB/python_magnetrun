"""
FEPC kHz Data Reader
Reads configuration and binary data files from FEPC acquisition system
"""

import numpy as np
import struct
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


@dataclass
class CalibrationInfo:
    """Calibration parameters for a channel"""
    coef_a: float = 1.0
    coef_b: float = 0.0
    a: float = 1.0
    b: float = 0.0
    cnv_file: str = None  # Path to CNV file if piecewise calibration


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
    calibrations: List[CalibrationInfo] = None  # Calibration for each channel


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
        # Format: VAR1;VAR2;VAR3;... with embedded calibration parameters
        
        if not line:
            continue
        
        # Split by semicolons but handle complex parameter strings
        # Variables can contain parameters like: NAME=VAR;COEFA=1.0;COEFB=0.0;...
        segments = line.split(';')
        
        # Determine card type and slot from line position
        slot = line_idx - 1
        
        if line_idx <= 6:  # Analog cards (adjust based on your system)
            card_type = 'ANA'
            num_channels = 16
        else:  # Digital cards
            card_type = 'DIG'
            num_channels = 32
        
        # Parse variables and calibration info
        variable_names = []
        calibrations = []
        
        current_var = {}
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
                
            # Check if this segment contains parameter assignments
            if '=' in segment:
                # Parse parameter
                key_val = segment.split('=', 1)
                if len(key_val) == 2:
                    key, val = key_val
                    key = key.upper()
                    current_var[key] = val
                    
                    # If we have a NAME, we've started a new variable
                    if key == 'NAME':
                        # Save previous variable if exists
                        if 'prev_name' in current_var:
                            variable_names.append(current_var['prev_name'])
                            calibrations.append(_extract_calibration(current_var))
                        current_var = {'NAME': val, 'prev_name': val}
            else:
                # Simple variable name without parameters
                if segment and not any(c in segment for c in '=;'):
                    variable_names.append(segment)
                    calibrations.append(CalibrationInfo())  # Default calibration
        
        # Don't forget the last variable
        if 'NAME' in current_var:
            variable_names.append(current_var['NAME'])
            calibrations.append(_extract_calibration(current_var))
        
        # If no parametric parsing worked, fall back to simple split
        if not variable_names:
            variable_names = [v.strip() for v in segments if v.strip() and '=' not in v]
            calibrations = [CalibrationInfo() for _ in variable_names]
        
        # Ensure we have the right number of channels
        while len(variable_names) < num_channels:
            variable_names.append(f"CH{len(variable_names)}")
            calibrations.append(CalibrationInfo())
        
        # Trim to exact number of channels
        variable_names = variable_names[:num_channels]
        calibrations = calibrations[:num_channels]
        
        # Extract info from header (assuming 10kHz sampling)
        card = CardInfo(
            slot=slot,
            card_type=card_type,
            sampling_freq=10000,
            buffer_pre=20,
            buffer_post=50,
            num_channels=num_channels,
            variable_names=variable_names,
            calibrations=calibrations
        )
        cards.append(card)
    
    config = FEPCConfig(
        fepc_name=fepc_name,
        num_cards=len(cards),
        cards=cards
    )
    
    return config


def _extract_calibration(var_dict: dict) -> CalibrationInfo:
    """
    Extract calibration parameters from variable dictionary
    
    Parameters:
    -----------
    var_dict : dict
        Dictionary with variable parameters (COEFA, COEFB, A, B, FILE_PROC, etc.)
        
    Returns:
    --------
    CalibrationInfo : Calibration object
    """
    def safe_float(val, default=0.0):
        try:
            return float(val.replace(',', '.'))  # Handle French decimal notation
        except (ValueError, AttributeError):
            return default
    
    calib = CalibrationInfo()
    
    # Extract linear calibration parameters
    if 'COEFA' in var_dict:
        calib.coef_a = safe_float(var_dict['COEFA'], 1.0)
    if 'COEFB' in var_dict:
        calib.coef_b = safe_float(var_dict['COEFB'], 0.0)
    if 'A' in var_dict:
        calib.a = safe_float(var_dict['A'], 1.0)
    if 'B' in var_dict:
        calib.b = safe_float(var_dict['B'], 0.0)
    
    # Extract CNV file path if present
    if 'FILE_PROC' in var_dict:
        cnv_path = var_dict['FILE_PROC']
        # Extract just the filename from full path
        if '\\' in cnv_path:
            cnv_path = cnv_path.split('\\')[-1]
        calib.cnv_file = cnv_path
    
    return calib


def read_analog_block(file, block_idx: int) -> Tuple[np.ndarray, Dict]:
    """
    Read one analog data block (1614 bytes)
    
    Block structure:
    - Header: 7 × 16 bits = 14 bytes
    - Data: 50 samples × 16 channels × 16 bits = 1600 bytes
    
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
    header_bytes = file.read(HEADER_SIZE)
    header_data = struct.unpack('<7H', header_bytes)  # Little-endian, 7 unsigned shorts
    
    header = {
        'values': header_data
    }
    
    # Read data (50 samples × 16 channels)
    data_bytes = file.read(DATA_SIZE)
    data = np.frombuffer(data_bytes, dtype=np.int16).reshape(50, 16)
    
    return data, header


def read_digital_block(file, block_idx: int) -> Tuple[np.ndarray, Dict]:
    """
    Read one digital data block (212 bytes)
    
    Block structure:
    - Header: 6 × 16 bits = 12 bytes
    - Data: 50 samples × 32 bits = 200 bytes
    
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
    header_bytes = file.read(HEADER_SIZE)
    header_data = struct.unpack('<6H', header_bytes)  # Little-endian, 6 unsigned shorts
    
    header = {
        'values': header_data
    }
    
    # Read data (50 samples × 32 bits)
    # Each sample is stored as 2 × 16 bits
    data_bytes = file.read(DATA_SIZE)
    data_uint16 = np.frombuffer(data_bytes, dtype=np.uint16).reshape(50, 2)
    
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
                   num_blocks: int = 72000) -> np.ndarray:
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
                block_data, _ = read_analog_block(f, block_idx)
            else:
                block_data, _ = read_digital_block(f, block_idx)
            
            # Store in output array
            start_idx = block_idx * 50
            end_idx = start_idx + 50
            data[start_idx:end_idx, :] = block_data
    
    return data


def load_calibration(cnv_path: str) -> Dict:
    """
    Load calibration file (.CNV) for piecewise linear calibration
    
    CNV file format:
    N1;Value1
    N2;Value2
    ...
    
    Where N is the raw ADC value and Value is the physical value
    
    Parameters:
    -----------
    cnv_path : str
        Path to the .CNV file
        
    Returns:
    --------
    calib : dict
        Dictionary with 'n_values' and 'physical_values' arrays,
        plus 'type' = 'piecewise'
    """
    n_values = []
    physical_values = []
    
    try:
        with open(cnv_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split(';')
                if len(parts) >= 2:
                    # Parse N value (integer)
                    n = int(parts[0].strip())
                    # Parse physical value (float, handle French decimal notation)
                    phys = float(parts[1].strip().replace(',', '.'))
                    
                    n_values.append(n)
                    physical_values.append(phys)
        
        # Sort by N values
        sorted_pairs = sorted(zip(n_values, physical_values))
        n_values, physical_values = zip(*sorted_pairs)
        
        return {
            'type': 'piecewise',
            'n_values': np.array(n_values),
            'physical_values': np.array(physical_values)
        }
        
    except FileNotFoundError:
        raise FileNotFoundError(f"CNV file not found: {cnv_path}")
    except Exception as e:
        raise ValueError(f"Error parsing CNV file {cnv_path}: {e}")


def apply_calibration(raw_data: np.ndarray, 
                      calib_info: CalibrationInfo = None,
                      cnv_dict: Dict = None) -> np.ndarray:
    """
    Apply calibration to convert raw int16 values to physical units
    
    Two calibration methods:
    1. Linear: Signal_converted = A * (COEF_A * N + COEF_B) + B
    2. Piecewise linear: Use CNV file lookup table with linear interpolation
    
    Parameters:
    -----------
    raw_data : np.ndarray
        Raw data from analog channels (int16)
    calib_info : CalibrationInfo
        Calibration parameters from CFG file
    cnv_dict : dict
        Piecewise calibration data from CNV file (optional)
        If provided, overrides linear calibration
        
    Returns:
    --------
    physical_data : np.ndarray
        Calibrated data in physical units (float)
    """
    if cnv_dict is not None and cnv_dict['type'] == 'piecewise':
        # Piecewise linear calibration
        return _apply_piecewise_calibration(raw_data, cnv_dict)
    
    elif calib_info is not None:
        # Linear calibration: Signal = A * (COEF_A * N + COEF_B) + B
        return _apply_linear_calibration(raw_data, calib_info)
    
    else:
        # No calibration, return as float
        return raw_data.astype(np.float64)


def _apply_linear_calibration(raw_data: np.ndarray, 
                              calib: CalibrationInfo) -> np.ndarray:
    """
    Apply linear calibration: Signal = A * (COEF_A * N + COEF_B) + B
    
    Example from documentation:
    COEF_A = 1, COEF_B = 0, A = 3.0518044E-4, B = -10
    N = 0 → Signal = -10V
    N = 65535 → Signal = +10V
    """
    # Convert to float for calculation
    n = raw_data.astype(np.float64)
    
    # Apply formula
    signal = calib.a * (calib.coef_a * n + calib.coef_b) + calib.b
    
    return signal


def _apply_piecewise_calibration(raw_data: np.ndarray, 
                                  cnv_dict: Dict) -> np.ndarray:
    """
    Apply piecewise linear calibration using CNV lookup table
    
    Uses linear interpolation between calibration points
    """
    n_values = cnv_dict['n_values']
    physical_values = cnv_dict['physical_values']
    
    # Use numpy's interpolation
    # extrapolation uses nearest value for out-of-range points
    physical_data = np.interp(
        raw_data.flatten(), 
        n_values, 
        physical_values
    ).reshape(raw_data.shape)
    
    return physical_data


def calibrate_channel(raw_data: np.ndarray,
                      card: CardInfo,
                      channel_idx: int,
                      cnv_directory: str = ".") -> np.ndarray:
    """
    Convenience function to calibrate a single channel
    
    Automatically selects calibration method based on CardInfo
    
    Parameters:
    -----------
    raw_data : np.ndarray
        Raw data for the channel
    card : CardInfo
        Card information containing calibration parameters
    channel_idx : int
        Channel index (0-15 for analog)
    cnv_directory : str
        Directory containing CNV files
        
    Returns:
    --------
    calibrated_data : np.ndarray
        Data in physical units
    """
    if card.card_type != 'ANA':
        raise ValueError("Calibration only applies to analog channels")
    
    if channel_idx >= len(card.calibrations):
        raise ValueError(f"Channel index {channel_idx} out of range")
    
    calib = card.calibrations[channel_idx]
    
    # Check if CNV file calibration should be used
    if calib.cnv_file:
        cnv_path = Path(cnv_directory) / calib.cnv_file
        if cnv_path.exists():
            cnv_dict = load_calibration(str(cnv_path))
            return apply_calibration(raw_data, cnv_dict=cnv_dict)
        else:
            print(f"Warning: CNV file {cnv_path} not found, using linear calibration")
    
    # Use linear calibration
    return apply_calibration(raw_data, calib_info=calib)


# Example usage
if __name__ == "__main__":
    # Example: Parse configuration file
    # config = parse_cfg_file("/path/to/HOST_1_DATA.CFG")
    # print(f"FEPC: {config.fepc_name}")
    # print(f"Number of cards: {config.num_cards}")
    # 
    # for card in config.cards:
    #     print(f"\nSlot {card.slot}: {card.card_type} card")
    #     print(f"  Variables: {card.variable_names[:3]}...")
    #
    # # Read data from a specific hour/slot
    # data = read_hour_file("00HOST_1_LIST_0.bin", "ANA")
    # print(f"Data shape: {data.shape}")
    
    print("FEPC Reader module loaded successfully!")
    print("Use parse_cfg_file() to read configuration")
    print("Use read_hour_file() to read binary data")
