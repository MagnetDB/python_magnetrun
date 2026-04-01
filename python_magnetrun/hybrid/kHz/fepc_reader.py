"""
FEPC kHz Data Reader
Reads configuration and binary data files from FEPC acquisition system
"""

import argparse
import logging
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...utils.validation import FileFormatError

# Setup logger
logger = logging.getLogger(__name__)

# FEPC kHz format constants
SAMPLES_PER_BLOCK = 50  # Number of samples in each data block
KHZ_SAMPLING_FREQUENCY = 1000.0  # Sampling frequency in Hz for kHz files
SAMPLE_INTERVAL = (
    1.0 / KHZ_SAMPLING_FREQUENCY
)  # Time interval between samples in seconds

# Analog card constants
ANALOG_CHANNELS = 16
ANALOG_HEADER_SIZE = 14  # bytes (7 × 16 bits)
ANALOG_DATA_SIZE = 1600  # bytes (50 samples × 16 channels × 16 bits)
ANALOG_BLOCK_SIZE = 1614  # bytes (header + data)

# Digital card constants
DIGITAL_CHANNELS = 32
DIGITAL_HEADER_SIZE = 12  # bytes (6 × 16 bits)
DIGITAL_DATA_SIZE = 200  # bytes (50 samples × 32 bits)
DIGITAL_BLOCK_SIZE = 212  # bytes (header + data)


@dataclass
class CalibrationInfo:
    """Calibration parameters for a channel"""

    coef_a: float = 1.0
    coef_b: float = 0.0
    a: float = 1.0
    b: float = 0.0
    cnv_file: str | None = None  # Path to CNV file if piecewise calibration
    unit: str | None = None  # Physical unit string (e.g., 'A', 'V', 'bar')


@dataclass
class CardInfo:
    """Information about a single acquisition card"""

    slot: int
    card_type: str  # 'ANA' or 'DIG'
    buffer_freq: int
    buffer_pre: int
    buffer_post: int
    num_channels: int
    variable_names: list[str]
    calibrations: list[CalibrationInfo] | None = None  # Calibration for each channel
    digital_info: list[dict] | None = (
        None  # Additional info for digital variables (NUM_BIT, etc.)
    )
    analog_info: list[dict] | None = (
        None  # Additional info for analog variables (calibration, unit, etc.)
    )


@dataclass
class FEPCConfig:
    """Complete FEPC configuration"""

    fepc_name: str
    num_cards: int
    cards: list[CardInfo]
    host_number: str | None = None  # HOST number extracted from CFG filename
    digital_variables: dict[int, list[str]] | None = (
        None  # Slot -> variable names for digital cards
    )
    analog_variables: dict[int, list[str]] | None = (
        None  # Slot -> variable names for analog cards
    )

    def __post_init__(self):
        if self.digital_variables is None:
            self.digital_variables = {}
        if self.analog_variables is None:
            self.analog_variables = {}

    def get_card_by_slot(self, slot: int) -> CardInfo:
        """Get card information by slot number"""
        for card in self.cards:
            if card.slot == slot:
                return card
        raise ValueError(f"Slot {slot} not found")

    def get_analog_slots(self) -> list[int]:
        """Return list of analog card slots"""
        return [card.slot for card in self.cards if card.card_type == "ANA"]

    def get_digital_slots(self) -> list[int]:
        """Return list of digital card slots"""
        return [card.slot for card in self.cards if card.card_type == "DIG"]


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

    Header format:
    FEPC_NAME;num_cards;freq1;pre1;post1;type1;nchannels1;freq2;pre2;post2;type2;nchannels2;...
    Where:
        - freq: buffer frequency in Hz
        - pre: pre-trigger buffer in seconds
        - post: post-trigger buffer in seconds
        - type: card type (0=ANA, 1=DIG)
        - nchannels: number of channels
    """
    with open(cfg_path) as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    # First line contains the header with FEPC info
    header = lines[0]

    # Parse header to get FEPC name and basic structure
    # Format: FEPC_NAME;num_cards;freq1;pre1;post1;type1;nchannels1;...
    parts = header.split(";")

    # Extract FEPC name (usually first element before numbers)
    fepc_name = parts[0].split("|")[0] if "|" in parts[0] else parts[0]
    logger.debug(f"FEPC Name: {fepc_name}")

    # Extract number of cards from parts[1]
    num_cards = int(parts[1]) if len(parts) > 1 else len(lines) - 1
    logger.debug(f"Number of cards: {num_cards}")

    # Parse card descriptions from header
    # Each card has 5 fields: freq, pre, post, type, nchannels
    card_descriptions = []
    header_idx = 2  # Start after FEPC_NAME and num_cards
    for card_idx in range(num_cards):
        if header_idx + 4 < len(parts):
            freq = int(parts[header_idx])
            pre = int(parts[header_idx + 1])
            post = int(parts[header_idx + 2])
            card_type_code = parts[header_idx + 3]
            nchannels = int(parts[header_idx + 4])
            card_type = "ANA" if card_type_code == "ANALOG" else "DIG"
            card_descriptions.append(
                {
                    "freq": freq,
                    "pre": pre,
                    "post": post,
                    "card_type": card_type,
                    "nchannels": nchannels,
                }
            )
            logger.debug(
                f"Card {card_idx}: freq={freq}, pre={pre}, post={post}, type={card_type}, nchannels={nchannels}"
            )
            header_idx += 5
        else:
            # Fallback defaults if header doesn't have enough info
            card_descriptions.append(
                {
                    "freq": 10000,
                    "pre": 20,
                    "post": 50,
                    "card_type": "ANA" if card_idx < 6 else "DIG",
                    "nchannels": ANALOG_CHANNELS if card_idx < 6 else DIGITAL_CHANNELS,
                }
            )

    # Parse each card's information
    cards = []

    for line_idx, line in enumerate(lines[1 : num_cards + 1], start=1):
        # Each line contains the variable names for that slot
        # Format: VAR1;VAR2;VAR3;...
        variable_names = [v.strip() for v in line.split(";") if v.strip()]

        # Get card description from header
        card_idx = line_idx - 1
        card_desc = (
            card_descriptions[card_idx] if card_idx < len(card_descriptions) else {}
        )

        slot = card_idx
        card_type = card_desc.get("card_type", "ANA" if line_idx <= 6 else "DIG")  # type: ignore[assignment]
        num_channels = card_desc.get(  # type: ignore[assignment]
            "nchannels", ANALOG_CHANNELS if card_type == "ANA" else DIGITAL_CHANNELS
        )
        buffer_freq = card_desc.get("freq", 10000)  # type: ignore[assignment]
        buffer_pre = card_desc.get("pre", 20)  # type: ignore[assignment]
        buffer_post = card_desc.get("post", 50)  # type: ignore[assignment]

        # Ensure we have the right number of channels
        while len(variable_names) < num_channels:  # type: ignore[operator]
            variable_names.append(f"CH{len(variable_names)}")

        # Trim to exact number of channels
        variable_names = variable_names[:num_channels]  # type: ignore[misc]

        # Default calibrations (no calibration info in variable lines)
        calibrations = None

        # Create card with info from header
        card = CardInfo(
            slot=slot,
            card_type=card_type,
            buffer_freq=buffer_freq,  # type: ignore[arg-type]
            buffer_pre=buffer_pre,  # type: ignore[arg-type]
            buffer_post=buffer_post,  # type: ignore[arg-type]
            num_channels=num_channels,  # type: ignore[arg-type]
            variable_names=variable_names,
            calibrations=calibrations,
        )
        logger.debug(
            f"Parsed Card Slot {slot}: Type={card_type}, Channels={num_channels}, variables={variable_names}"
        )
        cards.append(card)

    # Parse additional lines for DIGITAL and ANALOG variable definitions
    # Lines are key=value pairs separated by semicolons
    # TYPE=DIGITAL: ACTIVED=true;NUM_BIT=0;TYPE=DIGITAL;NAME=xxx;...
    # TYPE=ANALOG: DISPLAY=3;TYPE=ANALOG;NAME=xxx;COEFA=1.0;COEFB=0.0;A=-0.003;B=100.0;FILE_PROC=xxx.CNV;...

    # Get lists of digital and analog slots for sequential assignment
    digital_slots = [card for card in cards if card.card_type == "DIG"]
    analog_slots = [card for card in cards if card.card_type == "ANA"]

    # Dicts to store variables by slot
    digital_variables = {}
    analog_variables = {}

    # Lists to collect variable info before assigning to slots
    digital_var_list: list[dict] = []
    analog_var_list: list[dict] = []

    def safe_float(val, default=0.0):
        """Convert string to float, handling French decimal notation"""
        try:
            return float(val.replace(",", "."))
        except (ValueError, AttributeError):
            return default

    for line in lines[num_cards + 1 :]:
        line = line.strip()
        if not line:
            continue

        # print(f"Parsing extra line: {line[:80]}...", flush=True)

        # Parse line as dict of key=value pairs
        line_dict = {}
        for segment in line.split(";"):
            if "=" in segment:
                key, value = segment.split("=", 1)
                line_dict[key.strip().upper()] = value.strip()

        # Get TYPE to determine if DIGITAL or ANALOG
        var_type = line_dict.get("TYPE", "").upper()
        var_name = line_dict.get("NAME", "")
        num_channel = line_dict.get("NUMCHANNEL", "")

        if var_type == "DIGITAL":
            num_bit = line_dict.get("NUM_BIT", "")
            digital_var_list.append(
                {
                    "name": var_name,
                    "num_bit": int(num_bit) if num_bit else len(digital_var_list),
                    "num_channel": int(num_channel) if num_channel else 0,
                    "info": line_dict,
                }
            )

        elif var_type == "ANALOG":
            # Extract calibration parameters
            coef_a = safe_float(line_dict.get("COEFA", "1.0"), 1.0)
            coef_b = safe_float(line_dict.get("COEFB", "0.0"), 0.0)
            a = safe_float(line_dict.get("A", "1.0"), 1.0)
            b = safe_float(line_dict.get("B", "0.0"), 0.0)

            # Extract CNV file path if present
            cnv_file = line_dict.get("FILE_PROC", "")
            if cnv_file and "\\" in cnv_file:
                cnv_file = cnv_file.split("\\")[-1]  # Extract just filename

            # Extract unit string if present
            unit = line_dict.get("UNIT", "")

            # Create calibration info
            calib = CalibrationInfo(
                coef_a=coef_a,
                coef_b=coef_b,
                a=a,
                b=b,
                cnv_file=cnv_file if cnv_file else None,
                unit=unit if unit else None,
            )

            analog_var_list.append(
                {
                    "name": var_name,
                    "num_channel": (
                        int(num_channel) if num_channel else len(analog_var_list)
                    ),
                    "calibration": calib,
                    "info": line_dict,
                }
            )

        else:
            # Unknown line format - log it
            logger.warning(f"Unknown line format (TYPE={var_type}): {line[:50]}...")

    # Sort variables by num_bit/num_channel for proper ordering
    digital_var_list.sort(key=lambda x: x["num_bit"])  # type: ignore[return-value]
    analog_var_list.sort(key=lambda x: x["num_channel"])  # type: ignore[return-value]

    # Assign collected digital variables to slots
    for card in digital_slots:
        # Store digital_var_list as additional info, keep existing variable_names
        card.digital_info = digital_var_list
        digital_variables[card.slot] = card.variable_names
        logger.debug(
            f"Updated DIGITAL slot {card.slot} with {len(digital_var_list)} digital variable infos"
        )

    # Assign collected analog variables to slots
    for card in analog_slots:
        # Build calibrations list from analog_var_list, matching to existing variable_names
        calibrations = []
        for var_name in card.variable_names:
            # Find matching calibration from analog_var_list
            calib = CalibrationInfo()  # Default
            for var_info in analog_var_list:
                if var_info["name"] == var_name:
                    calib = var_info["calibration"]  # type: ignore[assignment]
                    break
            calibrations.append(calib)

        # Store analog_var_list as additional info, keep existing variable_names
        card.analog_info = analog_var_list
        card.calibrations = calibrations
        analog_variables[card.slot] = card.variable_names
        logger.debug(
            f"Updated ANALOG slot {card.slot} with {len(analog_var_list)} analog variable infos and calibrations"
        )

    logger.debug(f"Total cards parsed: {len(cards)}")
    config = FEPCConfig(
        fepc_name=fepc_name,
        num_cards=len(cards),
        cards=cards,
        digital_variables=digital_variables,
        analog_variables=analog_variables,
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
            return float(val.replace(",", "."))  # Handle French decimal notation
        except (ValueError, AttributeError):
            return default

    calib = CalibrationInfo()

    # Extract linear calibration parameters
    if "COEFA" in var_dict:
        calib.coef_a = safe_float(var_dict["COEFA"], 1.0)
    if "COEFB" in var_dict:
        calib.coef_b = safe_float(var_dict["COEFB"], 0.0)
    if "A" in var_dict:
        calib.a = safe_float(var_dict["A"], 1.0)
    if "B" in var_dict:
        calib.b = safe_float(var_dict["B"], 0.0)

    # Extract CNV file path if present
    if "FILE_PROC" in var_dict:
        cnv_path = var_dict["FILE_PROC"]
        # Extract just the filename from full path
        if "\\" in cnv_path:
            cnv_path = cnv_path.split("\\")[-1]
        calib.cnv_file = cnv_path

    return calib


def read_analog_block(
    file, block_idx: int, endian: str = "big", verbose: bool = False
) -> tuple[np.ndarray, np.ndarray]:
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
    timestamps : np.ndarray
        Shape (50,) - timestamp for each sample in seconds
        Note: header timestamp corresponds to last sample (index 49),
              so each previous sample has 1 ms subtracted
    """

    # Seek to block position
    file.seek(block_idx * ANALOG_BLOCK_SIZE)

    # Read header (14 bytes total)
    # Header structure:
    #   Bytes 0-3:   Integer value (uint32)
    #   Bytes 4-7:   Timestamp in seconds (uint32)
    #   Bytes 8-9:   Timestamp in milliseconds (uint16)
    #   Bytes 10-13: Unused (4 bytes)
    endian_char = ">" if endian == "big" else "<"
    header_bytes = file.read(ANALOG_HEADER_SIZE)
    # Unpack as: uint32 (integer), uint32 (seconds), uint16 (milliseconds), 2×uint16 (unused)
    header_data = struct.unpack(f"{endian_char}IIH2H", header_bytes)
    actual_block_idx = header_data[0]
    t_seconds = header_data[1]
    t_milliseconds = header_data[2]
    # Reconstruct timestamp: seconds + milliseconds converted to seconds
    # This timestamp corresponds to the LAST sample (index 49) in the block
    t_last = t_seconds + (t_milliseconds / KHZ_SAMPLING_FREQUENCY)

    header = {"values": header_data}
    logger.debug(
        f"read_analog_block: block[{block_idx}] header={header}, t_last={t_last}, actual_block_idx={actual_block_idx}"
    )

    # Read data (50 samples × 16 channels)
    data_bytes = file.read(ANALOG_DATA_SIZE)
    dtype = np.dtype(np.uint16).newbyteorder(">" if endian == "big" else "<")
    data = np.frombuffer(data_bytes, dtype=dtype).reshape(
        SAMPLES_PER_BLOCK, ANALOG_CHANNELS
    )
    # print(f"read_analog_block: data shape={data.shape}")

    # Create timestamp array: header timestamp is for last sample (index 49)
    # Each previous sample is one sample interval earlier
    timestamps = np.array(
        [
            t_last - (SAMPLES_PER_BLOCK - 1 - i) * SAMPLE_INTERVAL
            for i in range(SAMPLES_PER_BLOCK)
        ]
    )

    return data, timestamps


def read_digital_block(
    file, block_idx: int, endian: str = "big", verbose: bool = False
) -> tuple[np.ndarray, np.ndarray]:
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
    timestamps : np.ndarray
        Shape (50,) - timestamp for each sample in seconds
        Note: header timestamp corresponds to last sample (index 49),
              so each previous sample has 1 ms subtracted
    """

    # Seek to block position
    file.seek(block_idx * DIGITAL_BLOCK_SIZE)

    # Read header (12 bytes total)
    # Header structure:
    #   Bytes 0-3:  Integer value (uint32)
    #   Bytes 4-7:  Timestamp in seconds (uint32)
    #   Bytes 8-9:  Timestamp in milliseconds (uint16)
    #   Bytes 10-11: Unused (2 bytes)
    endian_char = ">" if endian == "big" else "<"
    header_bytes = file.read(DIGITAL_HEADER_SIZE)
    # Unpack as: uint32 (integer), uint32 (seconds), uint16 (milliseconds), uint16 (unused)
    header_data = struct.unpack(f"{endian_char}IIH H", header_bytes)
    int_value = header_data[0]
    t_seconds = header_data[1]
    t_milliseconds = header_data[2]
    # Reconstruct timestamp: seconds + milliseconds converted to seconds
    # This timestamp corresponds to the LAST sample (index 49) in the block
    t_last = t_seconds + (t_milliseconds / KHZ_SAMPLING_FREQUENCY)

    header = {"values": header_data}
    if verbose:
        # try to reverse engineer header data (see claude "decoding numeric timestamp" chat)
        logger.warning(
            f"read_digital_block: block[{block_idx}] header={header}, t_last={t_last}, actual_block_idx={int_value}"
        )

    # Read data (50 samples × 32 bits per sample)
    # Each sample contains 32 digital bits (one per channel)
    data_bytes = file.read(DIGITAL_DATA_SIZE)
    dtype = np.dtype(np.uint16).newbyteorder(">" if endian == "big" else "<")
    data_uint16 = np.frombuffer(data_bytes, dtype=dtype).reshape(SAMPLES_PER_BLOCK, 2)

    # Convert to individual boolean bits (32 channels per sample)
    # Extract bits using vectorized numpy operations for efficiency
    data = np.zeros((SAMPLES_PER_BLOCK, DIGITAL_CHANNELS), dtype=bool)
    for bit in range(16):
        # Extract bit position from word1 (channels 0-15)
        data[:, bit] = ((data_uint16[:, 0] >> bit) & 1).astype(bool)
        # Extract bit position from word2 (channels 16-31)
        data[:, 16 + bit] = ((data_uint16[:, 1] >> bit) & 1).astype(bool)

    # Create timestamp array: header timestamp is for last sample (index 49)
    # Each previous sample is one sample interval earlier
    timestamps = np.array(
        [
            t_last - (SAMPLES_PER_BLOCK - 1 - i) * SAMPLE_INTERVAL
            for i in range(SAMPLES_PER_BLOCK)
        ]
    )

    logger.debug(f"read_digital_block: data shape = {data.shape}")
    return data, timestamps


def read_hour_file(
    filepath: str,
    card_type: str,
    num_blocks: int = 72000,
    endian: str = "big",
    debug: bool = False,
    debug_channel: int = 0,
    debug_interval: int = 1000,
) -> np.ndarray:
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
    debug : bool
        If True, plot data incrementally during reading (default: False)
    debug_channel : int
        Channel index to plot in debug mode (default: 0)
    debug_interval : int
        Update plot every N blocks (default: 1000)

    Returns:
    --------
    data : np.ndarray
        For analog: shape (3600000, 16) - 3.6M samples, 16 channels
        For digital: shape (3600000, 32) - 3.6M samples, 32 channels
    """
    import matplotlib.pyplot as plt

    num_channels = ANALOG_CHANNELS if card_type == "ANA" else DIGITAL_CHANNELS
    total_samples = num_blocks * SAMPLES_PER_BLOCK
    logger.info(
        f"read_hour_file: file={filepath}, card_type={card_type}, num_blocks={num_blocks}, num_channels={num_channels}, total_samples={total_samples}, endian={endian}"
    )

    # Estimate actual block numbers based on file size
    from pathlib import Path

    block_size = ANALOG_BLOCK_SIZE if card_type == "ANA" else DIGITAL_BLOCK_SIZE
    file_path = Path(filepath)

    file_size = file_path.stat().st_size
    actual_num_blocks = file_size // block_size
    remainder = file_size % block_size

    if remainder != 0:
        raise FileFormatError(
            f"{filepath}: file size {file_size} is not a multiple of block size "
            f"{block_size} for card_type={card_type} (remainder={remainder} bytes)"
        )

    verbose = False
    if num_blocks != actual_num_blocks:
        verbose = True
        logger.warning(
            f"{filepath}: expected {num_blocks} blocks got {actual_num_blocks}"
        )

    # Pre-allocate array
    data = np.zeros(
        (total_samples, num_channels), dtype=np.uint16 if card_type == "ANA" else bool
    )
    # TODO set NaN instead of zero
    data = np.empty((total_samples, num_channels))
    data.fill(np.nan)
    logger.info(
        f"read_hour_file: file={filepath}, Allocated data array of shape {data.shape} for card type {card_type}"
    )

    # Setup debug plot if requested
    fig, ax, line = None, None, None
    if debug:
        plt.ion()  # Interactive mode
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.set_xlabel("Sample")
        ax.set_ylabel(f"Channel {debug_channel}")
        ax.set_title(f"Debug: Reading {filepath} - Block 0/{num_blocks}")
        ax.grid(True, alpha=0.3)
        (line,) = ax.plot([], [], linewidth=0.5, alpha=0.8)
        plt.show(block=False)

    with open(filepath, "rb") as f:
        for block_idx in range(num_blocks):
            try:
                if card_type == "ANA":
                    block_data, timestamps = read_analog_block(
                        f, block_idx, endian, verbose
                    )
                else:
                    block_data, timestamps = read_digital_block(
                        f, block_idx, endian, verbose
                    )

                # Store in output array
                start_idx = block_idx * SAMPLES_PER_BLOCK
                end_idx = start_idx + SAMPLES_PER_BLOCK
                data[start_idx:end_idx, :] = block_data
                logger.debug(
                    f"read_hour_file: file={filepath}, block_idx={block_idx}/{num_blocks}, start_idx={start_idx}, end_idx={end_idx} stored, block_data shape={block_data.shape}"
                )

                # Update debug plot at intervals
                if debug and (block_idx + 1) % debug_interval == 0:
                    current_end = end_idx
                    x_data = np.arange(current_end)
                    y_data = data[:current_end, debug_channel]
                    if line is not None:
                        line.set_data(x_data, y_data)
                    if ax is not None:
                        ax.relim()
                        ax.autoscale_view()
                        ax.set_title(
                            f"Debug: Reading {Path(filepath).name} - Block {block_idx + 1}/{num_blocks}"
                        )
                    if fig is not None:
                        fig.canvas.draw()
                        fig.canvas.flush_events()
                    plt.pause(0.01)
            except struct.error as e:
                logger.warning(
                    f"read_hour_file: Error reading block {block_idx}/{num_blocks} in {filepath}: {e}"
                )

        logger.info(f"Reading complete from {filepath}, data shape: {data.shape}")

    if debug:
        # Final update
        if line is not None:
            line.set_data(np.arange(total_samples), data[:, debug_channel])
        if ax is not None:
            ax.relim()
            ax.autoscale_view()
            ax.set_title(
                f"Debug: {Path(filepath).name} - Complete ({num_blocks} blocks)"
            )
        if fig is not None:
            fig.canvas.draw()
            fig.canvas.flush_events()
        plt.ioff()
        plt.show()  # Keep final plot open

    return data


def load_calibration(cnv_path: str) -> dict:
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
        with open(cnv_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split(";")
                if len(parts) >= 2:
                    # Parse N value (integer)
                    n = int(parts[0].strip())
                    # Parse physical value (float, handle French decimal notation)
                    phys = float(parts[1].strip().replace(",", "."))

                    n_values.append(n)
                    physical_values.append(phys)

        # Sort by N values
        sorted_pairs = sorted(zip(n_values, physical_values, strict=False))
        n_values, physical_values = zip(*sorted_pairs, strict=False)  # type: ignore[assignment]

        return {
            "type": "piecewise",
            "n_values": np.array(n_values),
            "physical_values": np.array(physical_values),
        }

    except FileNotFoundError:
        raise FileNotFoundError(f"CNV file not found: {cnv_path}") from None
    except (OSError, ValueError, RuntimeError) as e:
        raise ValueError(f"Error parsing CNV file {cnv_path}: {e}") from e


def apply_calibration(
    raw_data: np.ndarray,
    calib_info: CalibrationInfo | None = None,
    cnv_dict: dict | None = None,
) -> np.ndarray:
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
    if cnv_dict is not None and cnv_dict["type"] == "piecewise":
        logger.debug(
            f"apply piecewise calibration using CNV data (cvn_dict={cnv_dict})"
        )
        # Piecewise linear calibration
        return _apply_piecewise_calibration(raw_data, cnv_dict)

    elif calib_info is not None:
        logger.debug(
            f"apply linear calibration: COEF_A={calib_info.coef_a}, COEF_B={calib_info.coef_b}, A={calib_info.a}, B={calib_info.b}"
        )
        # Linear calibration: Signal = A * (COEF_A * N + COEF_B) + B
        return _apply_linear_calibration(raw_data, calib_info)

    else:
        # No calibration, return as float
        return raw_data.astype(np.float64)


def _apply_linear_calibration(
    raw_data: np.ndarray, calib: CalibrationInfo
) -> np.ndarray:
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


def _apply_piecewise_calibration(raw_data: np.ndarray, cnv_dict: dict) -> np.ndarray:
    """
    Apply piecewise linear calibration using CNV lookup table

    Uses linear interpolation between calibration points
    """
    n_values = cnv_dict["n_values"]
    physical_values = cnv_dict["physical_values"]

    # Use numpy's interpolation
    # extrapolation uses nearest value for out-of-range points
    physical_data = np.interp(raw_data.flatten(), n_values, physical_values).reshape(
        raw_data.shape
    )

    return physical_data


def calibrate_channel(
    raw_data: np.ndarray, card: CardInfo, channel_idx: int, cnv_directory: str = "."
) -> np.ndarray:
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
    if card.card_type != "ANA":
        raise ValueError("Calibration only applies to analog channels")

    if card.calibrations is None:
        raise ValueError(f"No calibrations available for card slot {card.slot}")

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
            logger.warning(f"CNV file {cnv_path} not found, using linear calibration")

    # Use linear calibration
    return apply_calibration(raw_data, calib_info=calib)


def main():
    """Main function for CLI usage"""
    parser = argparse.ArgumentParser(
        description="Read FEPC kHz acquisition data files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config", "-c", type=str, help="Path to HOST_X_DATA.CFG configuration file"
    )

    parser.add_argument("--datafile", "-d", type=str, help="Path to binary data file")

    parser.add_argument(
        "--card-type",
        "-t",
        type=str,
        choices=["ANA", "DIG"],
        default="ANA",
        help="Card type: ANA (analog) or DIG (digital), default: ANA",
    )

    parser.add_argument(
        "--num-blocks",
        "-n",
        type=int,
        default=72000,
        help="Number of blocks to read (default: 72000 = 1 hour)",
    )

    parser.add_argument(
        "--endian",
        "-e",
        type=str,
        choices=["big", "little"],
        default="big",
        help="Endianness of binary data: big or little (default: big)",
    )

    parser.add_argument(
        "--kHz-datadir",
        type=str,
        default="~/LNCMIG-Data/CEA/kHz/",
        help="Directory containing kHz binary data files (default: current directory)",
    )

    parser.add_argument(
        "--trigger-datadir",
        type=str,
        default="~/LNCMIG-Data/CEA/trigger/",
        help="Directory containing trigger data files (default: current directory)",
    )

    parser.add_argument(
        "--rms-datadir",
        type=str,
        default="~/LNCMIG-Data/CEA/rms",
        help="Directory containing RMS data files (default: current directory)",
    )

    args = parser.parse_args()

    # Parse configuration if provided
    if args.config:
        config = parse_cfg_file(args.config)
        logger.info(f"FEPC: {config.fepc_name}")
        logger.info(f"Number of cards: {config.num_cards}")

        for card in config.cards:
            logger.info(f"\nSlot {card.slot}: {card.card_type} card")
            logger.info(f"  Sampling frequency: {KHZ_SAMPLING_FREQUENCY:.0f} Hz")
            logger.info(f"  Number of channels: {card.num_channels}")
            logger.info(
                f"  Variables: {', '.join(card.variable_names[:5])}{', ...' if len(card.variable_names) > 5 else ''}"
            )

    # Read data file if provided
    if args.datafile:
        logger.info(f"Reading data file: {args.datafile}")
        logger.info(f"Card type: {args.card_type}")
        logger.info(f"Endianness: {args.endian}")
        logger.info(f"Number of blocks: {args.num_blocks}")

        data = read_hour_file(
            args.datafile, args.card_type, args.num_blocks, args.endian
        )
        logger.info(f"Data shape: {data.shape}")
        logger.info(f"Data type: {data.dtype}")

        if args.card_type == "ANA":
            logger.info(f"Data range: [{data.min()}, {data.max()}]")
            logger.info(f"First 5 samples (channel 0): {data[:5, 0]}")

    if not args.config and not args.datafile:
        logger.info("FEPC Reader module loaded successfully!")
        logger.info("\nUsage examples:")
        logger.info("  Parse config:  python fepc_reader.py --config HOST_2_DATA.CFG")
        logger.info(
            "  Read data:     python fepc_reader.py --datafile data.bin --card-type ANA --endian big"
        )
        logger.info(
            "  Full example:  python fepc_reader.py -c HOST_2_DATA.CFG -d data.bin -t ANA -n 1000 -e big"
        )
        parser.print_help()

    return args


if __name__ == "__main__":
    main()
