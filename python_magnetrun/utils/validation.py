"""File-format validation utilities.

Validators are called before parsing to give clear, early errors instead of
cryptic failures deep inside pandas/nptdms/struct.

All format-specific validators raise :exc:`FileFormatError` (a ``ValueError``
subclass), so existing ``except (OSError, ValueError, RuntimeError)`` catch
blocks in callers continue to work without modification.
"""

from __future__ import annotations

import os


class FileFormatError(ValueError):
    """Raised when a file fails pre-parse format validation."""


def validate_file_exists(path: str) -> None:
    """Raise :exc:`FileNotFoundError` if *path* does not exist."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"file not found: {path}")


def validate_txt_format(path: str) -> None:
    """Validate a pupitre ``.txt`` file before parsing.

    Checks:
    - File exists.
    - Extension is ``.txt``.
    - File is non-empty.
    - Second line (header row) contains both ``Date`` and ``Time`` tokens,
      which are required downstream for :meth:`addTime`.

    :raises FileNotFoundError: if *path* does not exist
    :raises FileFormatError: if the file fails structural checks
    """
    validate_file_exists(path)
    ext = os.path.splitext(path)[-1]
    if ext != ".txt":
        raise FileFormatError(f"{path}: expected .txt extension, got '{ext}'")
    try:
        with open(path) as f:
            lines = [f.readline() for _ in range(2)]
    except OSError as exc:
        raise FileFormatError(f"{path}: cannot read file: {exc}") from exc
    if not any(line.strip() for line in lines):
        raise FileFormatError(f"{path}: file is empty")
    header = lines[1] if len(lines) > 1 else ""
    if "Date" not in header or "Time" not in header:
        raise FileFormatError(
            f"{path}: missing required header columns ['Date', 'Time'] in second line"
        )


def validate_tdms_format(path: str) -> None:
    """Validate a TDMS file by checking its 4-byte magic number ``TDSm``.

    :raises FileNotFoundError: if *path* does not exist
    :raises FileFormatError: if the magic bytes do not match
    """
    validate_file_exists(path)
    expected = b"TDSm"
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
    except OSError as exc:
        raise FileFormatError(f"{path}: cannot read file: {exc}") from exc
    if magic != expected:
        raise FileFormatError(
            f"{path}: expected TDMS magic {expected!r} at offset 0, got {magic!r}"
        )


def validate_csv_format(path: str, required_columns: list[str] | None = None) -> None:
    """Validate a CSV file before parsing.

    Checks:
    - File exists and is non-empty.
    - First line is non-empty and readable as text.
    - If *required_columns* is provided, all column names appear in the first line.

    :raises FileNotFoundError: if *path* does not exist
    :raises FileFormatError: if the file fails structural checks
    """
    validate_file_exists(path)
    if os.path.getsize(path) == 0:
        raise FileFormatError(f"{path}: file is empty")
    try:
        with open(path) as f:
            first_line = f.readline()
    except (OSError, UnicodeDecodeError) as exc:
        raise FileFormatError(f"{path}: cannot read as text: {exc}") from exc
    if not first_line.strip():
        raise FileFormatError(f"{path}: first line is empty")
    if required_columns:
        missing = [col for col in required_columns if col not in first_line]
        if missing:
            raise FileFormatError(
                f"{path}: missing required columns {missing} in header line"
            )


def validate_rms_format(path: str) -> None:
    """Validate an RMS binary file by checking that the first byte is ``#``.

    The RMS ASCII header starts with ``#``-prefixed lines; a file that does
    not begin with ``#`` is not a valid RMS file.

    :raises FileNotFoundError: if *path* does not exist
    :raises FileFormatError: if the first byte is not ``b'#'``
    """
    validate_file_exists(path)
    try:
        with open(path, "rb") as f:
            first_byte = f.read(1)
    except OSError as exc:
        raise FileFormatError(f"{path}: cannot read file: {exc}") from exc
    if first_byte != b"#":
        got = hex(first_byte[0]) if first_byte else "empty file"
        raise FileFormatError(
            f"{path}: expected ASCII header marker '#' at byte 0, got {got}"
        )


def validate_vprocess_format(path: str) -> None:
    """Validate a VProcess binary file by checking that the first byte is ``#``.

    :raises FileNotFoundError: if *path* does not exist
    :raises FileFormatError: if the first byte is not ``b'#'``
    """
    validate_file_exists(path)
    try:
        with open(path, "rb") as f:
            first_byte = f.read(1)
    except OSError as exc:
        raise FileFormatError(f"{path}: cannot read file: {exc}") from exc
    if first_byte != b"#":
        got = hex(first_byte[0]) if first_byte else "empty file"
        raise FileFormatError(
            f"{path}: expected ASCII header marker '#' at byte 0, got {got}"
        )


def validate_fepc_binary_format(path: str, card_type: str) -> None:
    """Validate a FEPC kHz binary file by checking block-size alignment.

    :param path: path to the binary file
    :param card_type: ``'ANA'`` for analog cards, ``'DIG'`` for digital cards
    :raises FileNotFoundError: if *path* does not exist
    :raises FileFormatError: if the file size is not a multiple of the block size
    """
    from python_magnetrun.hybrid.kHz.fepc_reader import ANALOG_BLOCK_SIZE, DIGITAL_BLOCK_SIZE

    validate_file_exists(path)
    block_size = ANALOG_BLOCK_SIZE if card_type == "ANA" else DIGITAL_BLOCK_SIZE
    file_size = os.path.getsize(path)
    remainder = file_size % block_size
    if remainder != 0:
        raise FileFormatError(
            f"{path}: file size {file_size} is not a multiple of block size "
            f"{block_size} for card_type={card_type} (remainder={remainder} bytes)"
        )
