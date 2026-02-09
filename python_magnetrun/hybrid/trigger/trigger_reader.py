"""
FEPC Trigger Data Reader
Reads trigger data files from FEPC acquisition system

Trigger files contain:
- PRE seconds of data before trigger (currently 20s)
- POST seconds of data after trigger (currently 50s)  
- Total: 70s of data at 10 kHz = 700,000 samples
- Format: IEEE big endian
"""

import logging
import numpy as np
import struct
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import sys

# Import FEPC config parser from kHz module
sys.path.insert(0, str(Path(__file__).parent.parent / "kHz"))
from fepc_reader import FEPCConfig, parse_cfg_file, CalibrationInfo

# Setup logger
logger = logging.getLogger(__name__)


@dataclass
class TriggerInfo:
    """Information about a trigger event"""
    
    trigger_dir: Path
    timestamp: datetime
    sample_idx: int
    rtblock_id: int
    rtblock_phase: int
    trigger_approx_timestamp: Optional[datetime] = None
    pre_samples: int = 200000  # 20s * 10kHz
    post_samples: int = 500000  # 50s * 10kHz
    total_samples: int = 700000
    
    @property
    def trigger_name(self) -> str:
        """Get trigger name from directory"""
        return self.trigger_dir.name


@dataclass
class TriggerFileInfo:
    """Information about a single trigger binary file"""
    
    filepath: Path
    card_type: str  # 'ANA' or 'DIG'
    slot: int
    file_size: int
    expected_size: int
    

def parse_eventinfo_properties(properties_file: Path) -> Dict:
    """
    Parse EventInfo.properties file to extract trigger metadata
    
    Parameters:
    -----------
    properties_file : Path
        Path to EventInfo.properties file
        
    Returns:
    --------
    dict : Trigger metadata
    """
    metadata = {}
    
    try:
        with open(properties_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        metadata[key.strip()] = value.strip()
    except Exception as e:
        logger.error(f"Error parsing EventInfo.properties: {e}")
        return metadata
    
    return metadata


def parse_trigger_directory(trigger_dir: Path) -> TriggerInfo:
    """
    Parse trigger directory to extract metadata
    
    Parameters:
    -----------
    trigger_dir : Path
        Path to trigger directory (e.g., TRIGGER_2025-11-05_08-16)
        
    Returns:
    --------
    TriggerInfo : Trigger metadata
    """
    # Parse directory name to get timestamp
    dir_name = trigger_dir.name
    parts = dir_name.split('_')
    
    if len(parts) != 3 or parts[0] != 'TRIGGER':
        raise ValueError(f"Invalid trigger directory name: {dir_name}")
    
    date_str = parts[1]
    time_str = parts[2]
    
    # Parse timestamp
    timestamp = datetime.strptime(f"{date_str}_{time_str}", "%Y-%m-%d_%H-%M")
    
    # Look for EventInfo.properties in subdirectories
    fepc_lncmi_dir = trigger_dir / "FEPC-LNCMI"
    fepc_aux_dir = trigger_dir / "FEPC-AUX-LNCMI"
    
    metadata = {}
    
    # Try FEPC-LNCMI first
    if fepc_lncmi_dir.exists():
        props_file = fepc_lncmi_dir / "EventInfo.properties"
        if props_file.exists():
            metadata = parse_eventinfo_properties(props_file)
    
    # Try FEPC-AUX-LNCMI if not found
    if not metadata and fepc_aux_dir.exists():
        props_file = fepc_aux_dir / "EventInfo.properties"
        if props_file.exists():
            metadata = parse_eventinfo_properties(props_file)
    
    # Extract key metadata
    sample_idx = int(metadata.get('trig.sample.idx', 199999))
    rtblock_id = int(metadata.get('trig.rtblock.id', 0))
    rtblock_phase = int(metadata.get('trig.rtblock.phase', 0))
    
    # Parse trigger approximate timestamp if available
    trig_timestamp = None
    if 'trig.timestamp.approx' in metadata:
        # Format: 06/06/2025-05:44:16.921 UTC[+0000]
        timestamp_str = metadata['trig.timestamp.approx'].split('[')[0].strip()
        try:
            trig_timestamp = datetime.strptime(timestamp_str, "%d/%m/%Y-%H:%M:%S.%f UTC")
        except:
            logger.warning(f"Could not parse trigger timestamp: {timestamp_str}")
    
    return TriggerInfo(
        trigger_dir=trigger_dir,
        timestamp=timestamp,
        sample_idx=sample_idx,
        rtblock_id=rtblock_id,
        rtblock_phase=rtblock_phase,
        trigger_approx_timestamp=trig_timestamp
    )


def read_trigger_file_header(filepath: Path, card_type: str = 'ANA') -> datetime:
    """
    Read 8-byte timestamp header from trigger binary file
    
    Parameters:
    -----------
    filepath : Path
        Path to trigger binary file
    card_type : str
        'ANA' or 'DIG'
        
    Returns:
    --------
    datetime : Trigger timestamp (ms since epoch)
    """
    with open(filepath, 'rb') as f:
        # Read 8 bytes for timestamp (big endian)
        timestamp_bytes = f.read(8)
        if len(timestamp_bytes) != 8:
            raise ValueError(f"Could not read 8-byte header from {filepath}")
        
        # Unpack as big-endian uint64
        timestamp_ms = struct.unpack('>Q', timestamp_bytes)[0]
        
        # Convert to datetime (ms since 1970-01-01)
        timestamp = datetime.fromtimestamp(timestamp_ms / 1000.0)
        
    return timestamp


def read_trigger_file(
    filepath: Path,
    card_type: str = 'ANA',
    endian: str = 'big',
    num_samples: Optional[int] = None
) -> Tuple[np.ndarray, datetime]:
    """
    Read trigger binary file
    
    Parameters:
    -----------
    filepath : Path
        Path to trigger binary file
    card_type : str
        'ANA' for analog (MIVA) or 'DIG' for digital (MAD)
    endian : str
        'big' or 'little' endian
    num_samples : int, optional
        Number of samples to read (default: all 700,000)
        
    Returns:
    --------
    tuple : (data array, trigger timestamp)
        - For ANA: shape (num_samples, 16)
        - For DIG: shape (num_samples, 32)
    """
    # Read header first
    timestamp = read_trigger_file_header(filepath, card_type)
    
    # Expected file sizes
    if card_type == 'ANA':
        expected_size = 700000 * 16 * 2 + 8  # 22,400,008 bytes
        num_channels = 16
        dtype = '>u2' if endian == 'big' else '<u2'  # uint16
    elif card_type == 'DIG':
        expected_size = 700000 * 4 + 8  # 2,800,008 bytes
        num_channels = 32
        dtype = '>u4' if endian == 'big' else '<u4'  # uint32
    else:
        raise ValueError(f"Unknown card type: {card_type}")
    
    # Check file size
    file_size = filepath.stat().st_size
    if file_size != expected_size:
        logger.warning(
            f"File size mismatch for {filepath.name}: "
            f"expected {expected_size}, got {file_size}"
        )
    
    # Determine actual number of samples
    if num_samples is None:
        if card_type == 'ANA':
            actual_samples = (file_size - 8) // (16 * 2)
        else:
            actual_samples = (file_size - 8) // 4
        num_samples = actual_samples
    
    # Read data
    with open(filepath, 'rb') as f:
        # Skip 8-byte header
        f.seek(8)
        
        if card_type == 'ANA':
            # Read analog data
            data = np.fromfile(f, dtype=dtype, count=num_samples * 16)
            data = data.reshape(-1, 16)
        else:
            # Read digital data
            data = np.fromfile(f, dtype=dtype, count=num_samples)
            # Unpack bits (each uint32 contains 32 boolean channels)
            bits = np.unpackbits(data.view(np.uint8)).reshape(-1, 32)
            # Reverse bit order (bit 31 to bit 0)
            data = np.flip(bits, axis=1).astype(bool)
    
    return data, timestamp


def load_trigger_config(trigger_dir: Path, system: str = 'FEPC-LNCMI') -> Optional[FEPCConfig]:
    """
    Load FEPC configuration from trigger directory
    
    Parameters:
    -----------
    trigger_dir : Path
        Path to trigger directory
    system : str
        'FEPC-LNCMI' or 'FEPC-AUX-LNCMI'
        
    Returns:
    --------
    FEPCConfig or None : Configuration object
    """
    system_dir = trigger_dir / system
    
    if not system_dir.exists():
        logger.error(f"System directory not found: {system_dir}")
        return None
    
    # Look for CFG file (e.g., HOST_1_DATA.CFG or HOST_2_DATA.CFG)
    cfg_files = list(system_dir.glob("HOST_*_DATA.CFG"))
    
    if not cfg_files:
        logger.error(f"No CFG file found in {system_dir}")
        return None
    
    if len(cfg_files) > 1:
        logger.warning(f"Multiple CFG files found in {system_dir}, using first one")
    
    cfg_file = cfg_files[0]
    logger.info(f"Loading config from: {cfg_file}")
    
    return parse_cfg_file(str(cfg_file))


def list_trigger_files(trigger_dir: Path, system: str = 'FEPC-LNCMI') -> List[TriggerFileInfo]:
    """
    List all trigger binary files in a directory
    
    Parameters:
    -----------
    trigger_dir : Path
        Path to trigger directory
    system : str
        'FEPC-LNCMI' or 'FEPC-AUX-LNCMI'
        
    Returns:
    --------
    list : List of TriggerFileInfo objects
    """
    system_dir = trigger_dir / system
    
    if not system_dir.exists():
        return []
    
    files = []
    
    # Find all .bin files
    for bin_file in system_dir.glob("host_*.bin"):
        # Parse filename to determine card type and slot
        # Expected format: host_1_trig_0.bin (slot 0, analog)
        #                  host_1_trig_1.bin (slot 1, analog)
        #                  host_2_trig_0.bin (slot 0, digital)
        parts = bin_file.stem.split('_')
        
        if len(parts) < 4:
            logger.warning(f"Unexpected filename format: {bin_file.name}")
            continue
        
        slot = int(parts[3])
        
        # Determine card type from file size
        file_size = bin_file.stat().st_size
        if file_size == 22400008:
            card_type = 'ANA'
            expected_size = 22400008
        elif file_size == 2800008:
            card_type = 'DIG'
            expected_size = 2800008
        else:
            logger.warning(f"Unexpected file size for {bin_file.name}: {file_size}")
            # Try to guess from config
            card_type = 'ANA' if file_size > 10000000 else 'DIG'
            expected_size = 22400008 if card_type == 'ANA' else 2800008
        
        files.append(TriggerFileInfo(
            filepath=bin_file,
            card_type=card_type,
            slot=slot,
            file_size=file_size,
            expected_size=expected_size
        ))
    
    return sorted(files, key=lambda x: x.slot)


def read_trigger_data(
    trigger_dir: Path,
    system: str = 'FEPC-LNCMI',
    variable_name: Optional[str] = None,
    slot: Optional[int] = None,
    endian: str = 'big',
    num_samples: Optional[int] = None
) -> Tuple[np.ndarray, datetime, FEPCConfig]:
    """
    Read trigger data for a specific variable or slot
    
    Parameters:
    -----------
    trigger_dir : Path
        Path to trigger directory
    system : str
        'FEPC-LNCMI' or 'FEPC-AUX-LNCMI'
    variable_name : str, optional
        Variable name to extract
    slot : int, optional
        Slot number to read
    endian : str
        'big' or 'little' endian
    num_samples : int, optional
        Number of samples to read
        
    Returns:
    --------
    tuple : (data array, timestamp, config)
    """
    # Load configuration
    config = load_trigger_config(trigger_dir, system)
    if config is None:
        raise ValueError(f"Could not load config for {system}")
    
    # Determine which slot to read
    if variable_name:
        # Find slot containing this variable
        found = False
        for card in config.cards:
            if variable_name in card.variable_names:
                slot = card.slot
                card_type = card.card_type
                channel = card.variable_names.index(variable_name)
                found = True
                break
        
        if not found:
            raise ValueError(f"Variable '{variable_name}' not found in config")
        
        logger.info(f"Found {variable_name} in slot {slot}, channel {channel}")
    
    elif slot is not None:
        # Get card info for this slot
        card = config.get_card_by_slot(slot)
        card_type = card.card_type
    
    else:
        raise ValueError("Must specify either variable_name or slot")
    
    # Find the binary file for this slot
    trigger_files = list_trigger_files(trigger_dir, system)
    
    bin_file = None
    for tf in trigger_files:
        if tf.slot == slot and tf.card_type == card_type:
            bin_file = tf.filepath
            break
    
    if bin_file is None:
        raise ValueError(f"No binary file found for slot {slot}, card type {card_type}")
    
    logger.info(f"Reading data from: {bin_file}")
    
    # Read the data
    data, timestamp = read_trigger_file(bin_file, card_type, endian, num_samples)
    
    # Extract specific channel if variable was specified
    if variable_name:
        data = data[:, channel]
    
    return data, timestamp, config


def create_time_array(num_samples: int, sampling_freq: float = 10000.0) -> np.ndarray:
    """
    Create time array for trigger data
    
    Parameters:
    -----------
    num_samples : int
        Number of samples (default 700,000)
    sampling_freq : float
        Sampling frequency in Hz (default 10 kHz)
        
    Returns:
    --------
    np.ndarray : Time array in seconds
    """
    dt = 1.0 / sampling_freq
    return np.arange(num_samples) * dt


def find_trigger_directories(base_dir: Path, date: Optional[str] = None) -> List[Path]:
    """
    Find all trigger directories in base directory
    
    Parameters:
    -----------
    base_dir : Path
        Base directory containing trigger subdirectories
    date : str, optional
        Date string in YYYY-MM-DD format to filter triggers
        
    Returns:
    --------
    list : List of trigger directory paths
    """
    trigger_dir = base_dir / "trigger"
    
    if not trigger_dir.exists():
        logger.error(f"Trigger directory not found: {trigger_dir}")
        return []
    
    # Find all TRIGGER_* directories
    if date:
        # Filter by date
        pattern = f"TRIGGER_{date}_*"
    else:
        pattern = "TRIGGER_*"
    
    triggers = sorted(trigger_dir.glob(pattern))
    
    return triggers


# =============================
# Main / CLI
# =============================

def main():
    """Command-line interface for trigger reader"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="FEPC Trigger Data Reader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List trigger directories
  python trigger_reader.py --base-dir /data/hybrid --list-triggers
  
  # List trigger directories for a specific date
  python trigger_reader.py --base-dir /data/hybrid --list-triggers --date 2025-11-05
  
  # Show trigger info
  python trigger_reader.py --trigger-dir /data/hybrid/trigger/TRIGGER_2025-11-05_08-16 --info
  
  # Read specific variable
  python trigger_reader.py --trigger-dir /data/hybrid/trigger/TRIGGER_2025-11-05_08-16 \\
                           --system FEPC-LNCMI --variable I_H1
  
  # Read specific slot
  python trigger_reader.py --trigger-dir /data/hybrid/trigger/TRIGGER_2025-11-05_08-16 \\
                           --system FEPC-LNCMI --slot 0
        """
    )
    
    parser.add_argument(
        '--base-dir',
        type=Path,
        help='Base directory containing trigger subdirectory'
    )
    
    parser.add_argument(
        '--trigger-dir',
        type=Path,
        help='Path to specific trigger directory'
    )
    
    parser.add_argument(
        '--date',
        type=str,
        help='Date in YYYY-MM-DD format'
    )
    
    parser.add_argument(
        '--system',
        type=str,
        choices=['FEPC-LNCMI', 'FEPC-AUX-LNCMI'],
        default='FEPC-LNCMI',
        help='FEPC system name'
    )
    
    parser.add_argument(
        '--variable',
        type=str,
        help='Variable name to read'
    )
    
    parser.add_argument(
        '--slot',
        type=int,
        help='Slot number to read'
    )
    
    parser.add_argument(
        '--list-triggers',
        action='store_true',
        help='List all trigger directories'
    )
    
    parser.add_argument(
        '--info',
        action='store_true',
        help='Show trigger information'
    )
    
    parser.add_argument(
        '--endian',
        type=str,
        choices=['big', 'little'],
        default='big',
        help='Endianness of binary data'
    )
    
    parser.add_argument(
        '--num-samples',
        type=int,
        help='Number of samples to read (default: all)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    # List triggers
    if args.list_triggers:
        if not args.base_dir:
            print("Error: --base-dir required for --list-triggers")
            return
        
        triggers = find_trigger_directories(args.base_dir, args.date)
        
        if not triggers:
            print("No trigger directories found")
            return
        
        print(f"\nFound {len(triggers)} trigger(s):")
        for trig in triggers:
            print(f"  {trig.name}")
        return
    
    # Show trigger info
    if args.info:
        if not args.trigger_dir:
            print("Error: --trigger-dir required for --info")
            return
        
        try:
            trigger_info = parse_trigger_directory(args.trigger_dir)
            
            print("\n" + "="*60)
            print("TRIGGER INFORMATION")
            print("="*60)
            print(f"Trigger: {trigger_info.trigger_name}")
            print(f"Timestamp: {trigger_info.timestamp}")
            print(f"Sample index: {trigger_info.sample_idx}")
            print(f"RT Block ID: {trigger_info.rtblock_id}")
            print(f"RT Block Phase: {trigger_info.rtblock_phase}")
            if trigger_info.trigger_approx_timestamp:
                print(f"Approx timestamp: {trigger_info.trigger_approx_timestamp}")
            print(f"PRE samples: {trigger_info.pre_samples}")
            print(f"POST samples: {trigger_info.post_samples}")
            print(f"Total samples: {trigger_info.total_samples}")
            
            # List files
            for system in ['FEPC-LNCMI', 'FEPC-AUX-LNCMI']:
                files = list_trigger_files(args.trigger_dir, system)
                if files:
                    print(f"\n{system} files:")
                    for tf in files:
                        size_mb = tf.file_size / (1024**2)
                        print(f"  Slot {tf.slot} ({tf.card_type}): {tf.filepath.name} ({size_mb:.1f} MB)")
            
            # Show config
            for system in ['FEPC-LNCMI', 'FEPC-AUX-LNCMI']:
                config = load_trigger_config(args.trigger_dir, system)
                if config:
                    print(f"\n{system} configuration:")
                    print(f"  FEPC: {config.fepc_name}")
                    print(f"  Cards: {config.num_cards}")
                    for card in config.cards:
                        print(f"    Slot {card.slot} ({card.card_type}): {len(card.variable_names)} channels")
        
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        
        return
    
    # Read data
    if args.variable or args.slot is not None:
        if not args.trigger_dir:
            print("Error: --trigger-dir required")
            return
        
        try:
            data, timestamp, config = read_trigger_data(
                args.trigger_dir,
                args.system,
                args.variable,
                args.slot,
                args.endian,
                args.num_samples
            )
            
            print(f"\nData shape: {data.shape}")
            print(f"Data type: {data.dtype}")
            print(f"Trigger timestamp: {timestamp}")
            
            if data.ndim == 1:
                print(f"Data range: [{data.min()}, {data.max()}]")
                print(f"First 10 samples: {data[:10]}")
            else:
                print(f"Number of channels: {data.shape[1]}")
                print(f"Data range (channel 0): [{data[:, 0].min()}, {data[:, 0].max()}]")
        
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        
        return
    
    # No action specified
    print("No action specified. Use --help for usage information.")
    parser.print_help()


if __name__ == "__main__":
    main()
