"""
Shared base class for LNCMI binary data-file readers (RMS, VProcess).

Both formats use an ASCII header followed by a binary block where each
sample is: ``timestamp (8-byte double) + N channel values (float32 or uint8)``.
Only the encoding, format-specific header lines, and timestamp conversion
differ between the two.
"""

import logging
import re
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Binary format constants
FLOAT32_SIZE = 4   # bytes — analog channel value
DIGITAL_SIZE = 1   # bytes — digital/bit channel value
TIMESTAMP_SIZE = 8  # bytes — double-precision Unix timestamp


@dataclass
class ChannelVariable:
    """A single channel described in the file header.

    Parameters
    ----------
    name : str
        Channel name as it appears in the header.
    var_type : str
        Type string from the header (e.g. ``"float32"``, ``"bit"``, ``"dig"``).
    unit : str, optional
        Physical unit [e.g. ``"K"``, ``"A"``].
    min_val : float, optional
        Minimum value declared in the header.
    max_val : float, optional
        Maximum value declared in the header.
    display_format : str, optional
        Printf-style display format string (e.g. ``"%.2f"``).
    """

    name: str
    var_type: str
    unit: str | None = None
    min_val: float | None = None
    max_val: float | None = None
    display_format: str | None = None
    is_analog: bool = field(init=False)
    byte_size: int = field(init=False)

    def __post_init__(self) -> None:
        self.is_analog = "float" in self.var_type.lower()
        self.byte_size = FLOAT32_SIZE if self.is_analog else DIGITAL_SIZE

    def __repr__(self) -> str:
        return f"ChannelVariable({self.name}, {self.var_type}, {self.unit})"


class _BinaryFileReaderBase(ABC):
    """Abstract base for binary data-file readers with ASCII headers.

    Subclasses must implement :meth:`_validate_format`,
    :meth:`_parse_variables`, and :meth:`_make_timestamps`.
    They may also override :meth:`_parse_extra_header` to handle
    format-specific header lines (e.g. ``"# format"`` or
    ``"# processed on"``).

    Parameters
    ----------
    filepath : str
        Path to the data file.
    endian : str, optional
        Endianness of binary data: ``"big"`` (default) or ``"little"``.
    """

    _encoding: str = "US-ASCII"
    _default_sample_width: int = 0
    _default_data_offset: int = 0

    def __init__(self, filepath: str, endian: str = "big") -> None:
        self.filepath = Path(filepath)
        self.header_lines: list[str] = []
        self.variables: list[ChannelVariable] = []
        self.metadata: dict[str, Any] = {}
        self.data: pd.DataFrame | None = None
        self.endian = ">" if endian == "big" else "<"

    # ── header parsing ────────────────────────────────────────────────────

    def parse_header(self) -> None:
        """Parse the ASCII header, populating :attr:`variables` and :attr:`metadata`."""
        self.variables = []
        self.metadata = {}
        self._validate_format()
        with open(self.filepath, "rb") as f:
            lines: list[str] = []
            while True:
                raw = f.readline()
                if not raw.startswith(b"#"):
                    break
                lines.append(raw.decode(self._encoding, errors="ignore").strip())
        self.header_lines = lines
        for line in self.header_lines:
            if line.startswith("# variables"):
                self._parse_variables(line)
            elif line.startswith("# windows"):
                self._parse_windows(line)
            elif line.startswith("# frequency"):
                self._parse_frequency(line)
            elif line.startswith("# data-helper"):
                self._parse_data_helper(line)
            else:
                self._parse_extra_header(line)

    @abstractmethod
    def _validate_format(self) -> None:
        """Call the format-specific file validator."""

    @abstractmethod
    def _parse_variables(self, line: str) -> None:
        """Parse the ``# variables = …`` header line into :attr:`variables`."""

    def _parse_extra_header(self, line: str) -> None:  # noqa: B027
        """Hook for format-specific header lines not handled by the base class.

        Default: no-op.  Subclasses override to handle e.g. ``"# format"``
        or ``"# processed on"`` lines.
        """

    def _parse_windows(self, line: str) -> None:
        """Parse the ``# windows = [UTC] …`` time-range line."""
        match = re.search(r"\[UTC\]\s+(.+?)\s+->\s+(.+)", line)
        if not match:
            return
        start_str = match.group(1).strip()
        end_str = match.group(2).strip()
        fmt = "%d/%m/%Y-%H:%M:%S.%f"
        try:
            from datetime import datetime
            self.metadata["start_time"] = datetime.strptime(start_str, fmt)
            self.metadata["end_time"] = datetime.strptime(end_str, fmt)
        except ValueError:
            fmt = "%d/%m/%Y-%H:%M:%S"
            from datetime import datetime
            self.metadata["start_time"] = datetime.strptime(start_str.split(".")[0], fmt)
            self.metadata["end_time"] = datetime.strptime(end_str.split(".")[0], fmt)

    def _parse_frequency(self, line: str) -> None:
        """Parse the ``# frequency = … Hz`` line."""
        match = re.search(r"# frequency\s*=\s*([\d.]+)\s*Hz", line)
        if match:
            self.metadata["frequency"] = float(match.group(1))

    def _parse_data_helper(self, line: str) -> None:
        """Parse the ``# data-helper …`` line for offset, timestamp size, and sample width."""
        m = re.search(r"offset:(0x[\da-fA-F]+)", line)
        if m:
            self.metadata["data_offset"] = int(m.group(1), 16)
        m = re.search(r"time:(\d+)\(B\)", line)
        if m:
            self.metadata["timestamp_bytes"] = int(m.group(1))
        m = re.search(r"width:(\d+)\(B\)", line)
        if m:
            self.metadata["sample_width"] = int(m.group(1))

    # ── binary data reading ───────────────────────────────────────────────

    @abstractmethod
    def _make_timestamps(self, raw: list[float]) -> pd.DatetimeIndex:
        """Convert a list of raw Unix-epoch floats into a :class:`~pandas.DatetimeIndex`.

        Parameters
        ----------
        raw : list of float
            Unix timestamps (seconds since 1970-01-01 00:00:00 UTC).

        Returns
        -------
        pd.DatetimeIndex
            UTC-aware index.
        """

    def read_binary_data(self) -> pd.DataFrame:
        """Read the binary data block and return as a :class:`~pandas.DataFrame`.

        Returns
        -------
        pd.DataFrame
            Timestamp index (UTC) with one column per channel.

        Raises
        ------
        ValueError
            If :meth:`parse_header` has not been called yet.
        """
        if not self.variables:
            raise ValueError("Header must be parsed before reading binary data")

        timestamp_size = self.metadata.get("timestamp_bytes", TIMESTAMP_SIZE)
        sample_width = self.metadata.get("sample_width", self._default_sample_width)
        data_offset = self.metadata.get("data_offset", self._default_data_offset)

        with open(self.filepath, "rb") as f:
            f.seek(data_offset)
            binary_data = f.read()

        num_samples = len(binary_data) // sample_width
        logger.debug(
            "read_binary_data: %s — %d samples, %d variables",
            self.filepath.name, num_samples, len(self.variables),
        )

        raw_timestamps: list[float] = []
        data_arrays: dict[str, list] = {var.name: [] for var in self.variables}

        for i in range(num_samples):
            sample_start = i * sample_width
            sample_data = binary_data[sample_start : sample_start + sample_width]

            raw_timestamps.append(
                struct.unpack(f"{self.endian}d", sample_data[:timestamp_size])[0]
            )

            byte_offset = timestamp_size
            for var in self.variables:
                if var.is_analog:
                    value = struct.unpack(
                        f"{self.endian}f",
                        sample_data[byte_offset : byte_offset + FLOAT32_SIZE],
                    )[0]
                    byte_offset += FLOAT32_SIZE
                else:
                    value = struct.unpack(
                        "B", sample_data[byte_offset : byte_offset + DIGITAL_SIZE]
                    )[0]
                    byte_offset += DIGITAL_SIZE
                data_arrays[var.name].append(value)

        datetime_index = self._make_timestamps(raw_timestamps)
        df = pd.DataFrame(data_arrays, index=datetime_index)
        df.index.name = "timestamp"
        return df

    # ── public interface ──────────────────────────────────────────────────

    def read(self) -> pd.DataFrame:
        """Parse the header and read the binary data block.

        Returns
        -------
        pd.DataFrame
            Timestamp index (UTC) with one column per channel.
        """
        self.parse_header()
        self.data = self.read_binary_data()
        logger.debug("read: %s — %s", self.filepath.name, list(self.data.columns))
        return self.data

    def get_variable_info(self) -> pd.DataFrame:
        """Return channel metadata as a :class:`~pandas.DataFrame`.

        Returns
        -------
        pd.DataFrame
            Columns: ``name``, ``type``, ``unit``, ``min``, ``max``, ``display_format``.
        """
        if not self.variables:
            self.parse_header()
        return pd.DataFrame(
            [
                {
                    "name": v.name,
                    "type": v.var_type,
                    "unit": v.unit,
                    "min": v.min_val,
                    "max": v.max_val,
                    "display_format": v.display_format,
                }
                for v in self.variables
            ]
        )

    def get_metadata(self) -> dict[str, Any]:
        """Return a copy of the metadata dictionary.

        Returns
        -------
        dict
            File metadata (start/end time, frequency, data offset, …).
        """
        if not self.metadata:
            self.parse_header()
        return self.metadata.copy()

    def print_summary(self) -> None:
        """Log a formatted summary of the file contents."""
        if not self.metadata:
            self.parse_header()
        logger.info("%s: %s", self.__class__.__name__, self.filepath.name)
        logger.info("  Format   : %s", self.metadata.get("format", "unknown"))
        logger.info("  Frequency: %s Hz", self.metadata.get("frequency", "unknown"))
        logger.info("  Start    : %s", self.metadata.get("start_time", "unknown"))
        logger.info("  End      : %s", self.metadata.get("end_time", "unknown"))
        logger.info(
            "  Offset   : 0x%x", self.metadata.get("data_offset", 0)
        )
        logger.info(
            "  Width    : %s B", self.metadata.get("sample_width", "unknown")
        )
        analog = sum(1 for v in self.variables if v.is_analog)
        logger.info(
            "  Variables: %d (%d analog, %d digital)",
            len(self.variables), analog, len(self.variables) - analog,
        )
        if self.data is not None:
            logger.info("  Samples  : %d", len(self.data))
