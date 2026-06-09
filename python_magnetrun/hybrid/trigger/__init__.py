"""
FEPC Trigger Data Module

This module provides tools for reading and analyzing FEPC trigger data files.

Main components:
- trigger_reader: Read trigger binary files and configuration
"""

from .trigger_reader import (
    TRIGGER_SAMPLING_FREQUENCY,
    TriggerFileInfo,
    TriggerFileReader,
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
    "TRIGGER_SAMPLING_FREQUENCY",
    "TriggerInfo",
    "TriggerFileInfo",
    "TriggerFileReader",
    "parse_eventinfo_properties",
    "parse_trigger_directory",
    "read_trigger_file_header",
    "read_trigger_file",
    "load_trigger_config",
    "list_trigger_files",
    "read_trigger_data",
    "create_time_array",
    "find_trigger_directories",
]
