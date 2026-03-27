"""
FEPC Trigger Data Module

This module provides tools for reading and analyzing FEPC trigger data files.

Main components:
- trigger_reader: Read trigger binary files and configuration
- plot_trigger_data: Visualize trigger data
"""

from .plot_trigger_data import (
    apply_calibration,
    plot_multiple_triggers,
    plot_trigger_variable,
)
from .trigger_reader import (
    TriggerFileInfo,
    TriggerInfo,
    create_time_array,
    find_trigger_directories,
    list_trigger_files,
    load_trigger_config,
    parse_eventinfo_properties,
    parse_trigger_directory,
    read_trigger_data,
    read_trigger_file,
    read_trigger_file_header,
)

__all__ = [
    "TriggerInfo",
    "TriggerFileInfo",
    "parse_eventinfo_properties",
    "parse_trigger_directory",
    "read_trigger_file_header",
    "read_trigger_file",
    "load_trigger_config",
    "list_trigger_files",
    "read_trigger_data",
    "create_time_array",
    "find_trigger_directories",
    "apply_calibration",
    "plot_trigger_variable",
    "plot_multiple_triggers",
]
