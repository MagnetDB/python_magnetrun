"""
VProcess module for LNCMI data

Provides readers and utilities for VProcess process monitoring data files.

Main Components:
- vprocess_reader: Core file reader
- validate: File validation utility

Example scripts (in examples/):
- vprocess_args.py: CLI argument definitions
- vprocess_batch.py: Batch processing
- vprocess_cli.py: Unified command-line interface
"""

from .vprocess_reader import (
    VProcessFileReader,
    VProcessVariable,
    find_vprocess_files_for_date,
    get_vprocess_info,
    parse_vprocess_filename,
    read_vprocess_file,
)

__all__ = [
    "VProcessFileReader",
    "VProcessVariable",
    "read_vprocess_file",
    "get_vprocess_info",
    "parse_vprocess_filename",
    "find_vprocess_files_for_date",
]

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("python_magnetrun")
except PackageNotFoundError:
    __version__ = "0.0.0"
