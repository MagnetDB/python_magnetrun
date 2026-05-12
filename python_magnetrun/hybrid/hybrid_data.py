"""
HybridData - Unified interface for hybrid magnet data (kHz, RMS, Trigger)

This module provides a unified interface for reading and accessing hybrid magnet
data from FEPC acquisition systems. Data is organized in three categories:
- kHz: High-frequency (1 kHz) data from FEPC cards
- RMS: Root Mean Square data at lower frequency
- Trigger: Event-triggered data

Directory structure:
    base_dir/
        kHz/
            YYYY-MM-DD/
                FEPC-AUX-LNCMI/
                FEPC-LNCMI/
        rms/
            YYYY-MM-DD/
                FEPC-AUX-LNCMI/
                FEPC-LNCMI/
        trigger/
            TRIGGER__YYYY-MM-DD__HH-MM/
                FEPC-AUX-LNCMI/
                FEPC-LNCMI/
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from natsort import natsorted

from ..magnetdata_base import DataType
from ..outliers import OutlierConfig

# Local imports

# Setup logger
logger = logging.getLogger(__name__)

# Import FEPC readers
try:
    from .kHz.fepc_reader import (
        FEPCConfig,
        calibrate_channel,
        compute_hour_t0,
        parse_cfg_file,
        read_hour_file,
    )
except ImportError as e:
    logger.warning(f"Could not import fepc_reader: {e}")
    FEPCConfig = None  # type: ignore[assignment, misc]
    parse_cfg_file = None  # type: ignore[assignment]
    read_hour_file = None  # type: ignore[assignment]
    calibrate_channel = None  # type: ignore[assignment]
    compute_hour_t0 = None  # type: ignore[assignment]

try:
    from .rms.rms_reader import RMSFileReader
except ImportError as e:
    logger.warning(f"Could not import rms_reader: {e}")
    RMSFileReader = None  # type: ignore[assignment, misc]


# FEPC system names
FEPC_SYSTEMS = ["FEPC-LNCMI", "FEPC-AUX-LNCMI"]


@dataclass
class HybridDataInfo:
    """Information about available hybrid data for a given day"""

    date: date
    base_dir: Path
    khz_available: bool = False
    rms_available: bool = False
    trigger_available: bool = False
    fepc_systems: list[str] = field(default_factory=list)
    khz_files: dict[str, list[Path]] = field(default_factory=dict)
    rms_files: dict[str, list[Path]] = field(default_factory=dict)
    trigger_dirs: dict[str, list[Path]] = field(
        default_factory=dict
    )  # TRIGGER__YYYY-MM-DD__HH-MM dirs
    trigger_files: dict[str, list[Path]] = field(default_factory=dict)


class HybridData:
    """
    Unified interface for hybrid magnet data (kHz, RMS, Trigger)

    Similar interface to MagnetData class for consistency.

    Parameters
    ----------
    base_dir : str or Path
        Base directory containing kHz, rms, trigger subdirectories
    date_str : str
        Date string in YYYY-MM-DD format
    fepc_system : str, optional
        FEPC system name: 'FEPC-LNCMI' or 'FEPC-AUX-LNCMI' (default: both)
    endian : str, optional
        Endianness for binary data: 'big' or 'little' (default: 'big')

    Attributes
    ----------
    FileName : str
        Identifier string for this data (similar to MagnetData)
    Groups : dict
        Groups of data channels organized by type
    Keys : list
        List of available data keys
    Type : int
        Data type identifier (3 for HybridData)
    Data : dict
        Dictionary containing loaded data
    """

    def __init__(
        self,
        base_dir: str | Path,
        date_str: str,
        fepc_system: str | None = None,
        endian: str = "big",
        defs_file: str | None = None,
    ):
        self.base_dir = Path(base_dir)
        self.date_str = date_str
        self.date = datetime.strptime(date_str, "%Y-%m-%d").date()
        self.fepc_system = fepc_system
        self.endian = endian
        self.defs_file: str | None = defs_file

        # MagnetData-like attributes
        self.FileName = f"HybridData_{date_str}"
        self.Groups: dict[str, dict] = {}
        self.Keys: list[str] = []
        self.Type = DataType.HYBRID
        self.Data: dict[str, Any] = {}
        self.units: dict[str, tuple] = {}

        # Internal storage (use Any for type hints since modules might not be available)
        self._khz_configs: dict[str, Any] = {}  # FEPCConfig instances
        self._rms_readers: dict[str, Any] = {}  # RMSFileReader instances
        self._info: HybridDataInfo = HybridDataInfo(
            date=self.date,
            base_dir=self.base_dir,
        )

        # Discover available data
        self._discover_data()

    def _discover_data(self) -> None:
        """Discover available data files for the given date"""
        systems_to_check = [self.fepc_system] if self.fepc_system else FEPC_SYSTEMS

        for system in systems_to_check:
            # Check kHz data
            khz_dir = self.base_dir / "kHz" / self.date_str / system
            if khz_dir.exists():
                self._info.khz_available = True
                if system not in self._info.fepc_systems:
                    self._info.fepc_systems.append(system)
                self._info.khz_files[system] = list(khz_dir.glob("*.bin"))
                # Look for CFG file
                cfg_files = list(khz_dir.glob("HOST_*_DATA.CFG"))
                if cfg_files:
                    self._info.khz_files[f"{system}_cfg"] = cfg_files

            # Check RMS data
            rms_dir = self.base_dir / "rms" / self.date_str / system
            if rms_dir.exists():
                self._info.rms_available = True
                if system not in self._info.fepc_systems:
                    self._info.fepc_systems.append(system)
                # Sort RMS files by name to ensure chronological order
                # Format: FEPC-LNCMI_YYYY-MM-DD_HHMM—YYYY-MM-DD_HHMM.rms
                self._info.rms_files[system] = sorted(rms_dir.glob("*.rms"))

        # Check trigger data (format: TRIGGER__YYYY-MM-DD__HH-MM)
        trigger_base = self.base_dir / "trigger"
        if trigger_base.exists():
            # Find all trigger directories matching the date
            trigger_pattern = f"TRIGGER__{self.date_str}__*"
            trigger_dirs = sorted(trigger_base.glob(trigger_pattern))

            for trigger_dir in trigger_dirs:
                for system in systems_to_check:
                    system_dir = trigger_dir / system
                    if system_dir.exists():
                        self._info.trigger_available = True
                        if system not in self._info.fepc_systems:
                            self._info.fepc_systems.append(system)
                        # Store trigger directories
                        if system not in self._info.trigger_dirs:
                            self._info.trigger_dirs[system] = []
                        self._info.trigger_dirs[system].append(trigger_dir)
                        # Store trigger files
                        if system not in self._info.trigger_files:
                            self._info.trigger_files[system] = []
                        self._info.trigger_files[system].extend(
                            list(system_dir.glob("*"))
                        )

        # Build groups and keys
        self._build_groups()

    def _build_group_keys(self, group: str, system: str) -> dict:
        """Build data keys for a specific group"""

        if group == "kHz":
            return self.get_khz_variables(system)
        elif group == "rms":
            return self.get_rms_variables(system)
        elif group == "trigger":
            raise NotImplementedError("Trigger group keys not implemented yet")
        else:
            raise ValueError(f"Unknown group: {group}")

    def _build_groups(self) -> None:
        """Build Groups and Keys from discovered data"""
        logger.debug("Building groups and keys for HybridData on %s", self.date_str)
        self.Groups = {}
        self.Keys = []

        for system in self._info.fepc_systems:
            # kHz group
            if system in self._info.khz_files:
                group_name = f"kHz/{system}"
                self.Groups[group_name] = {
                    "type": "kHz",
                    "system": system,
                    "files": self._info.khz_files[system],
                }
                try:
                    keys = self._build_group_keys("kHz", system)["analog"]
                except (ImportError, FileNotFoundError, ValueError) as e:
                    logger.warning("Could not get kHz keys for %s: %s", system, e)
                    keys = []
                logger.debug("getKeys: kHz keys for system=%s: %s", system, keys)
                self.Keys += [f"kHz/{system}/{key}" for key in keys]

            # RMS group
            if system in self._info.rms_files:
                group_name = f"rms/{system}"
                self.Groups[group_name] = {
                    "type": "rms",
                    "system": system,
                    "files": self._info.rms_files[system],
                }
                try:
                    keys = self._build_group_keys("rms", system)["analog"]
                except (ImportError, FileNotFoundError, ValueError) as e:
                    logger.warning("Could not get rms keys for %s: %s", system, e)
                    keys = []
                logger.debug("getKeys: RMS keys for system=%s: %s", system, keys)
                self.Keys += [f"rms/{system}/{key}" for key in keys]

            # Trigger group
            if system in self._info.trigger_files:
                group_name = f"trigger/{system}"
                self.Groups[group_name] = {
                    "type": "trigger",
                    "system": system,
                    "files": self._info.trigger_files[system],
                }
                self.Keys.append(group_name)

    @classmethod
    def fromdir(cls, base_dir: str, date_str: str, **kwargs):
        """
        Create HybridData from a directory for a given date

        Parameters
        ----------
        base_dir : str
            Base directory containing kHz, rms, trigger subdirectories
        date_str : str
            Date string in YYYY-MM-DD format
        **kwargs : dict
            Additional arguments (fepc_system, endian)

        Returns
        -------
        HybridData
            HybridData instance
        """
        return cls(base_dir, date_str, **kwargs)

    def __repr__(self):
        return (
            f"HybridData(date={self.date_str}, "
            f"systems={self._info.fepc_systems}, "
            f"kHz={self._info.khz_available}, "
            f"rms={self._info.rms_available}, "
            f"trigger={self._info.trigger_available})"
        )

    def getType(self) -> int:
        """Return data type identifier"""
        return self.Type

    def getKeys(self) -> list[str]:
        logger.debug(f"HybridData/getKeys: keys={self.Keys}")
        return self.Keys

    def getInfo(self) -> HybridDataInfo:
        """Return information about available data"""
        return self._info

    def print_summary(self) -> None:
        """Print summary of available data"""
        print(f"HybridData Summary for {self.date_str}")
        print("=" * 60)
        print(f"Base directory: {self.base_dir}")
        print(f"FEPC Systems: {', '.join(self._info.fepc_systems)}")

        print("Data availability:")
        print(f"  kHz data:     {'yes' if self._info.khz_available else 'no'}")
        print(f"  RMS data:     {'yes' if self._info.rms_available else 'no'}")
        print(f"  Trigger data: {'yes' if self._info.trigger_available else 'no'}")

        if self._info.khz_available:
            print("kHz files:")
            for system, files in self._info.khz_files.items():
                if not system.endswith("_cfg"):
                    print(f"  {system}: {len(files)} files")

        if self._info.rms_available:
            print("RMS files:")
            for system, files in self._info.rms_files.items():
                print(f"  {system}: {len(files)} files")

        if self._info.trigger_available:
            print("Trigger directories:")
            for system, dirs in self._info.trigger_dirs.items():
                print(f"  {system}: {len(dirs)} trigger events")
                for d in dirs:
                    # Extract time from directory name (TRIGGER__YYYY-MM-DD__HH-MM)
                    time_part = d.name.split("__")[-1] if "__" in d.name else ""
                    print(f"    - {time_part}")
            print("Trigger files:")
            for system, files in self._info.trigger_files.items():
                print(f"  {system}: {len(files)} files")

        print(flush=True)

    # -------------------------------------------------------------------------
    # kHz Data Methods
    # -------------------------------------------------------------------------

    def load_khz_config(self, system: str) -> Any:
        """
        Load kHz configuration for a FEPC system

        Parameters
        ----------
        system : str
            FEPC system name

        Returns
        -------
        FEPCConfig
            Configuration object

        Raises
        ------
        ImportError
            If the fepc_reader module is not available
        FileNotFoundError
            If no CFG file is found for the given system
        """
        logger.debug(f"load_khz_config: system={system}")
        if FEPCConfig is None or parse_cfg_file is None:
            raise ImportError("fepc_reader module not available")

        if system in self._khz_configs:
            return self._khz_configs[system]

        cfg_key = f"{system}_cfg"
        if cfg_key not in self._info.khz_files or not self._info.khz_files[cfg_key]:
            raise FileNotFoundError(f"No CFG file found for {system}")

        config = parse_cfg_file(str(self._info.khz_files[cfg_key][0]))
        self._khz_configs[system] = config
        return config

    def get_khz_variables(self, system: str) -> dict[str, list[str]]:
        """
        Get available kHz variables for a FEPC system

        Parameters
        ----------
        system : str
            FEPC system name

        Returns
        -------
        dict
            Dictionary with 'analog' and 'digital' variable lists
        """
        logger.debug(f"get_khz_variables: system={system}")
        config = self.load_khz_config(system)

        analog_vars = []
        digital_vars = []

        for card in config.cards:
            if card.card_type == "ANA":
                for var in card.variable_names:
                    # analog_vars.append(f"slot{card.slot}/{var}")
                    analog_vars.append(var)
            else:
                for var in card.variable_names:
                    # digital_vars.append(f"slot{card.slot}/{var}")
                    digital_vars.append(var)

        return {"analog": natsorted(analog_vars), "digital": natsorted(digital_vars)}

    def read_khz_variable(
        self,
        system: str,
        variable: str,
        slot: int | None = None,
        hours: range | list[int] | None = None,
        apply_calib: bool = True,
        cnv_dir: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Read kHz data for a specific variable

        Parameters
        ----------
        system : str
            FEPC system name
        variable : str
            Variable name
        slot : int, optional
            Card slot number (auto-detected if not provided)
        hours : range or list of int, optional
            Hours to read (default: all available)
        apply_calib : bool, optional
            Apply calibration (default: True)
        cnv_dir : str, optional
            Directory for CNV calibration files

        Returns
        -------
        tuple
            (data_array, time_array)
        """
        logger.debug(
            f"read_kHz_variable: system={system}, variable={variable}, slot={slot}, hours={hours}, apply_calib={apply_calib}, cnv_dir={cnv_dir}"
        )
        if (
            read_hour_file is None
            or calibrate_channel is None
            or compute_hour_t0 is None
        ):
            raise ImportError("fepc_reader module not available")

        config = self.load_khz_config(system)

        # Find variable slot and channel
        var_slot = slot
        var_channel = None
        var_card = None

        for _i, card in enumerate(config.cards):
            logger.debug(
                f"Checking card slot {card.slot} with variables: {card.variable_names}"
            )
            if variable in card.variable_names:
                var_slot = card.slot
                var_channel = card.variable_names.index(variable)
                var_card = card
                break

        if var_slot is None or var_channel is None or var_card is None:
            raise ValueError(f"Variable '{variable}' not found in configuration")

        # Find bin files for the slot
        khz_dir = self.base_dir / "kHz" / self.date_str / system
        bin_pattern = f"*HOST_*_LIST_{var_slot}.bin"
        bin_files = sorted(khz_dir.glob(bin_pattern))

        if hours is not None:
            # Filter by hour
            filtered_files = []
            for f in bin_files:
                # Extract hour from filename (XXHOST_...)
                try:
                    hour = int(f.name[:2])
                    if hour in hours:
                        filtered_files.append(f)
                except ValueError:
                    pass
            logger.debug(f"filtered_files: {filtered_files}")
            bin_files = filtered_files

        if not bin_files:
            raise ValueError(f"No bin files found for slot {var_slot}")

        # Get card type (var_card is guaranteed to be non-None here)
        card_type = var_card.card_type

        # Global t0 = HH:00:00 local time of the first file.
        # Each file is read with its own file-local t0 (returning timestamps in 0..3600s),
        # then shifted by (file_t0 - global_t0) so all timestamps share the same origin.
        global_t0 = compute_hour_t0(str(bin_files[0]), self.date_str)

        all_data = []
        all_timestamps = []
        debug = logger.isEnabledFor(logging.DEBUG)
        for bin_file in bin_files:
            logger.debug(f"Reading {bin_file.name}...")
            file_t0 = compute_hour_t0(str(bin_file), self.date_str)
            logger.debug(
                "file_t0: %s, global_t0: %s, offset: %s seconds",
                file_t0,
                global_t0,
                file_t0 - global_t0,
            )
            hour_data, hour_timestamps = read_hour_file(
                str(bin_file), card_type, endian=self.endian, t0=file_t0, debug=debug
            )
            # Shift file-local timestamps to be relative to global_t0
            offset = file_t0 - global_t0
            hour_timestamps = np.where(
                np.isnan(hour_timestamps), np.nan, hour_timestamps + offset
            )
            all_data.append(hour_data[:, var_channel])
            all_timestamps.append(hour_timestamps)

        data = np.concatenate(all_data)
        time = np.concatenate(all_timestamps)  # elapsed seconds from global_t0

        # Apply calibration
        if apply_calib and card_type == "ANA":
            if cnv_dir is None:
                cnv_dir = str(khz_dir)
            data = calibrate_channel(data, var_card, var_channel, cnv_dir)

        if np.all(np.isnan(data)):
            logger.warning(
                f"read_khz_variable: all-NaN result for {system}/{variable} — "
                f"check calibration files in {cnv_dir or khz_dir}"
            )

        return data, time

    # -------------------------------------------------------------------------
    # RMS Data Methods
    # -------------------------------------------------------------------------

    def load_rms_data(self, system: str, file_idx: int = 0) -> pd.DataFrame:
        """
        Load RMS data for a FEPC system

        Parameters
        ----------
        system : str
            FEPC system name
        file_idx : int, optional
            Index of RMS file to load (default: 0)

        Returns
        -------
        pd.DataFrame
            RMS data as DataFrame
        """
        if RMSFileReader is None:
            raise ImportError("rms_reader module not available")

        if system not in self._info.rms_files:
            raise ValueError(f"No RMS files found for {system}")

        rms_files = self._info.rms_files[system]
        if file_idx >= len(rms_files):
            raise ValueError(
                f"File index {file_idx} out of range (max: {len(rms_files) - 1})"
            )

        rms_file = rms_files[file_idx]
        reader = RMSFileReader(str(rms_file), endian=self.endian)
        return reader.read()

    def get_rms_variables(self, system: str, file_idx: int = 0) -> dict[str, list[str]]:
        """
        Get available RMS variables for a FEPC system

        Parameters
        ----------
        system : str
            FEPC system name
        file_idx : int, optional
            Index of RMS file (default: 0)

        Returns
        -------
        dict
            Dictionary with 'analog' and 'digital' variable lists
        """
        logger.debug(f"get_rms_variables: system={system}")
        if RMSFileReader is None:
            raise ImportError("rms_reader module not available")

        if system not in self._info.rms_files:
            raise ValueError(f"No RMS files found for {system}")

        rms_files = self._info.rms_files[system]
        if file_idx >= len(rms_files):
            raise ValueError(f"File index {file_idx} out of range")

        reader = RMSFileReader(str(rms_files[file_idx]), endian=self.endian)
        var_info = reader.get_variable_info()

        analog_vars = []
        digital_vars = []

        for _, row in var_info.iterrows():
            if row["type"] == "float32":
                analog_vars.append(row["name"])
            else:
                digital_vars.append(row["name"])

        return {"analog": natsorted(analog_vars), "digital": natsorted(digital_vars)}

    def get_rms_variable_info(self, system: str, file_idx: int = 0) -> pd.DataFrame:
        """
        Get detailed RMS variable information as DataFrame

        Parameters
        ----------
        system : str
            FEPC system name
        file_idx : int, optional
            Index of RMS file (default: 0)

        Returns
        -------
        pd.DataFrame
            Variable information with columns: name, type, unit, min, max, byte_size, display_format
        """
        if RMSFileReader is None:
            raise ImportError("rms_reader module not available")

        if system not in self._info.rms_files:
            raise ValueError(f"No RMS files found for {system}")

        rms_files = self._info.rms_files[system]
        if file_idx >= len(rms_files):
            raise ValueError(f"File index {file_idx} out of range")

        reader = RMSFileReader(str(rms_files[file_idx]), endian=self.endian)
        return reader.get_variable_info()

    def _parse_rms_filename_hour(self, filepath: Path) -> int | None:
        """
        Parse the start hour from an RMS filename

        Filename format: {system}_YYYY-MM-DD_HHMM—YYYY-MM-DD_HHMM.rms
        Example: FEPC-LNCMI_2025-01-06_0000—2025-01-06_0100.rms -> returns 0

        Parameters
        ----------
        filepath : Path
            Path to RMS file

        Returns
        -------
        int or None
            Start hour (0-23) or None if parsing fails
        """
        import re

        name = filepath.stem  # Remove .rms extension
        # Pattern: system_YYYY-MM-DD_HHMM—YYYY-MM-DD_HHMM
        # We want the first HHMM after the date
        match = re.search(r"\d{4}-\d{2}-\d{2}_(\d{2})\d{2}[—-]", name)
        if match:
            return int(match.group(1))
        return None

    def read_rms_variable(
        self,
        system: str,
        variable: str,
        file_idx: int | None = None,
        hours: range | list[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Read RMS data for a specific variable

        Parameters
        ----------
        system : str
            FEPC system name
        variable : str
            Variable name
        file_idx : int, optional
            Index of a specific RMS file to load. If provided, only this file is loaded.
        hours : range or list of int, optional
            List of hours to load (0-23). Files are selected based on their filename start hour.

        If both file_idx and hours are None, all available RMS files are loaded.

        Returns
        -------
        tuple
            (data_array, time_array) where time is in seconds from start
        """
        logger.debug(
            f"read_rms_variable: system={system}, variable={variable}, hours={hours}"
        )
        if RMSFileReader is None:
            raise ImportError("rms_reader module not available")

        if system not in self._info.rms_files:
            raise ValueError(f"No RMS files found for {system}")

        rms_files = self._info.rms_files[system]
        if not rms_files:
            raise ValueError(f"No RMS files found for {system}")

        # Determine which files to load
        if file_idx is not None:
            # Load specific file by index
            if file_idx >= len(rms_files):
                raise ValueError(
                    f"File index {file_idx} out of range (max: {len(rms_files) - 1})"
                )
            files_to_load = [rms_files[file_idx]]
        elif hours is not None:
            # Filter files by hour
            files_to_load = []
            for f in rms_files:
                hour = self._parse_rms_filename_hour(f)
                if hour is not None and hour in hours:
                    files_to_load.append(f)
            if not files_to_load:
                raise ValueError(f"No RMS files found for hours {hours}")
            logger.debug(f"Loading {len(files_to_load)} RMS files for hours {hours}")
        else:
            # Load all files (default behavior, consistent with kHz)
            files_to_load = rms_files
            logger.debug(f"Loading all {len(files_to_load)} RMS files")

        # Load and concatenate data from all selected files
        all_data = []
        all_timestamps = []

        for rms_file in files_to_load:
            if not rms_file.exists():
                logger.warning(
                    f"read_rms_variable: file no longer exists, skipping: {rms_file}"
                )
                continue
            reader = RMSFileReader(str(rms_file), endian=self.endian)
            rms_df = reader.read()

            # Check variable exists
            if variable not in rms_df.columns:
                available = ", ".join(rms_df.columns[:10])
                raise ValueError(
                    f"Variable '{variable}' not found in {rms_file.name}. Available: {available}..."
                )

            all_data.append(rms_df[variable].values)
            all_timestamps.append(rms_df.index)

        if not all_data:
            raise FileNotFoundError(
                f"read_rms_variable: no readable RMS files remain for {system}"
            )

        # Concatenate arrays
        data = np.concatenate(all_data)

        # Handle timestamps - concatenate and convert to seconds from start
        import pandas as pd

        timestamps = pd.Index(np.concatenate([t.to_numpy() for t in all_timestamps]))

        if hasattr(timestamps, "to_numpy"):
            # Convert to numpy datetime64, then to seconds
            time_ns = timestamps.to_numpy().astype("datetime64[ns]").astype(np.int64)
            time = (time_ns - time_ns[0]) / 1e9  # Convert nanoseconds to seconds
        else:
            # Fallback: use sample index with assumed frequency
            time = np.arange(len(data)) / 10.0  # Assume 10 Hz default

        return data, time

    def list_rms_files(self, system: str) -> list[Path]:
        """
        List available RMS files for a FEPC system

        RMS files are located in: base_dir/rms/YYYY-MM-DD/system/
        Filename format: {system}_YYYY-MM-DD_HHMM—YYYY-MM-DD_HHMM.rms

        Parameters
        ----------
        system : str
            FEPC system name ('FEPC-LNCMI' or 'FEPC-AUX-LNCMI')

        Returns
        -------
        list of Path
            List of RMS file paths, sorted chronologically by filename
        """
        if system not in self._info.rms_files:
            return []
        return self._info.rms_files[system]

    # -------------------------------------------------------------------------
    # Trigger Data Methods
    # -------------------------------------------------------------------------

    def list_trigger_events(self, system: str) -> list[dict[str, Any]]:
        """
        List trigger events (directories) for a FEPC system

        Parameters
        ----------
        system : str
            FEPC system name

        Returns
        -------
        list of dict
            List of trigger events with 'time', 'path', and 'files' keys
        """
        if system not in self._info.trigger_dirs:
            return []

        events = []
        for trigger_dir in self._info.trigger_dirs[system]:
            # Parse time from directory name (TRIGGER__YYYY-MM-DD__HH-MM)
            parts = trigger_dir.name.split("__")
            time_str = parts[-1] if len(parts) >= 3 else ""

            system_dir = trigger_dir / system
            files = list(system_dir.glob("*")) if system_dir.exists() else []

            events.append(
                {
                    "time": time_str,
                    "path": trigger_dir,
                    "system_path": system_dir,
                    "files": files,
                }
            )

        return events

    def list_trigger_files(self, system: str) -> list[Path]:
        """
        List available trigger files for a FEPC system

        Parameters
        ----------
        system : str
            FEPC system name

        Returns
        -------
        list of Path
            List of trigger file paths
        """
        if system not in self._info.trigger_files:
            return []
        return self._info.trigger_files[system]

    # -------------------------------------------------------------------------
    # MagnetData-like Interface Methods
    # -------------------------------------------------------------------------

    def getData(
        self,
        key: str | None = None,
        hours: range | list[int] | None = None,
    ) -> Any:
        """
        Get data for a specific key (MagnetData-compatible interface)

        Parameters
        ----------
        key : str, optional
            Data key in format 'type/system' or 'type/system/variable'
        hours : range or list of int, optional
            Hours to read (default: all available)

        Returns
        -------
        Data (type depends on the requested data)
        """
        if key is None:
            return self.Data

        parts = key.split("/")
        logger.debug(f"hybrid_data.getData: key={key}, parts={parts}, hours={hours}")
        if len(parts) < 2:
            raise ValueError(f"Invalid key format: {key}")

        data_type = parts[0]
        system = parts[1]

        if data_type == "kHz":
            if len(parts) >= 3:
                variable = parts[2]
                return self.read_khz_variable(system, variable, hours=hours)
            else:
                return self.get_khz_variables(system)

        elif data_type == "rms":
            if len(parts) >= 3:
                # Could add variable selection here
                pass
            return self.load_rms_data(system)

        elif data_type == "trigger":
            return self.list_trigger_files(system)

        elif data_type == "vprocess":
            # Placeholder for future processed data
            raise NotImplementedError(
                "HyBridData.getData: vprocess data not implemented yet"
            )
        else:
            raise ValueError(f"Unknown data type: {data_type}")

    def load_units_from_json(self, json_file: str, debug: bool = False) -> None:
        """Populate ``self.units`` from a JSON field-definition file.

        Overrides the base-class implementation to handle the ``kHz/``,
        ``rms/``, and ``trigger/`` key prefixes used in :attr:`Keys`.
        JSON entries use the short form ``"SYSTEM/VARIABLE"``; this method
        matches them against all prefixed variants present in ``self.Keys``.
        """
        from ..field_defs import load_defs
        from ..magnetdata_base import FieldMeta, _make_ureg

        ureg = _make_ureg()
        field_defs: dict = load_defs(json_file)

        # Build short_key → [full_key, ...] map.
        # Full keys: "kHz/FEPC-AUX-LNCMI/ALIM1_J1"  →  short: "FEPC-AUX-LNCMI/ALIM1_J1"
        short_to_fulls: dict[str, list[str]] = {}
        _prefixes = {"kHz", "rms", "trigger"}
        for full_key in self.Keys:
            parts = full_key.split("/", 1)
            short_key = (
                parts[1] if len(parts) == 2 and parts[0] in _prefixes else full_key
            )
            short_to_fulls.setdefault(short_key, []).append(full_key)

        for json_key, defn in field_defs.items():
            if json_key.startswith("_"):
                continue
            # Accept a direct match (json_key already has a prefix) or a short match.
            full_keys = (
                [json_key]
                if json_key in self.Keys
                else short_to_fulls.get(json_key, [])
            )
            if not full_keys:
                logger.debug(
                    f"load_units_from_json: {json_key!r} not in Keys, skipping"
                )
                continue

            symbol: str = defn.get("symbol", "")
            unit_str: str | None = defn.get("unit")
            label: str = defn.get("label", "")
            description: str = defn.get("description", "")

            if unit_str is None:
                pint_unit = None
            else:
                try:
                    parsed = ureg.parse_expression(unit_str)
                    pint_unit = parsed.units if hasattr(parsed, "units") else parsed
                except (ValueError, AttributeError) as exc:
                    raise ValueError(
                        f"load_units_from_json: cannot parse unit {unit_str!r} for field {json_key!r}"
                    ) from exc

            meta = FieldMeta(
                symbol=symbol, unit=pint_unit, label=label, description=description
            )
            for full_key in full_keys:
                self.units[full_key] = (symbol, pint_unit)
                self.field_meta[full_key] = meta
                if debug:
                    logger.debug(
                        f"load_units_from_json: {json_key!r} → {full_key!r}  symbol={symbol}, unit={pint_unit}"
                    )

    def Units(self, debug: bool = False, json_file: str | None = None) -> None:
        """Populate ``self.units`` from a field-definition JSON file.

        Resolution order:
        1. *json_file* argument (explicit override)
        2. ``self.defs_file`` set at construction time

        If neither is set, ``self.units`` remains empty (units are unknown).
        """
        resolved = json_file or self.defs_file
        if resolved is not None:
            self.load_units_from_json(resolved, debug=debug)

    def getUnitKey(self, key: str) -> tuple:
        """Return ``(symbol, unit)`` for *key*, or ``()`` when not available."""
        return self.units.get(key, ())

    def getFieldMeta(self, key: str):  # type: ignore[override]
        """Return :class:`FieldMeta` for *key*, or ``None`` when not available."""
        return self.field_meta.get(key)

    def addData(  # noqa: N802
        self,
        key: str,
        formula: str,
        symbol: str,
        unit: Any,  # pint.Unit | str | None
        label: str,
        description: str,
        debug: bool = False,
    ) -> int:
        """Register a derived field lazily (stored for future use).

        HybridData does not hold a single in-memory DataFrame, so derived
        fields cannot be computed eagerly.  This method records the intent so
        that callers that inspect ``self.field_meta`` can still discover the
        field's metadata.
        """
        from pint.errors import UndefinedUnitError

        from ..magnetdata_base import FieldMeta, _make_ureg

        if isinstance(unit, str) and unit:
            try:
                ureg = _make_ureg()
                parsed = ureg.parse_expression(unit)
                pint_unit = parsed.units if hasattr(parsed, "units") else parsed
            except (ValueError, UndefinedUnitError):
                pint_unit = None
        else:
            pint_unit = unit if unit else None

        if key not in self.Keys:
            self.Keys.append(key)
        self.units[key] = (symbol, pint_unit)
        self.field_meta[key] = FieldMeta(
            symbol=symbol, unit=pint_unit, label=label, description=description
        )
        logger.debug(f"HybridData.addData: registered derived key {key!r} (lazy)")
        return 0

    # -------------------------------------------------------------------------
    # Plotting Methods (delegating to plotting module)
    # -------------------------------------------------------------------------

    def plot_khz_variable(
        self,
        system: str,
        variable: str,
        hours: range | list[int] | None = None,
        apply_calib: bool = True,
        cnv_dir: str | None = None,
        ax=None,
        show: bool = True,
        save: str | None = None,
        outlier_config: OutlierConfig | None = None,
        **plot_kwargs,
    ):
        """
        Plot kHz data for a specific variable.

        This method delegates to hybrid.plotting.plot_khz_variable().

        Parameters
        ----------
        system : str
            FEPC system name
        variable : str
            Variable name
        hours : range or list of int, optional
            Hours to read (default: all available)
        apply_calib : bool, optional
            Apply calibration (default: True)
        cnv_dir : str, optional
            Directory for CNV calibration files
        ax : matplotlib.axes.Axes, optional
            Axes to plot on (creates new figure if None)
        show : bool, optional
            Show plot (default: True)
        save : str, optional
            Save plot to file
        outlier_config : OutlierConfig, optional
            Outlier detection/handling configuration. ``None`` skips detection.
        **plot_kwargs : dict
            Additional arguments passed to plt.plot()

        Returns
        -------
        tuple
            (fig, ax) matplotlib figure and axes
        """
        from ..outliers import OutlierDetector
        from . import plotting

        # Perform outlier detection if config provided
        outlier_result = None
        if outlier_config is not None:
            data, _ = self.read_khz_variable(
                system, variable, hours=hours, apply_calib=apply_calib, cnv_dir=cnv_dir
            )
            outlier_result = OutlierDetector(config=outlier_config).detect(data)

        return plotting.plot_khz_variable(
            self,
            system,
            variable,
            hours=hours,
            apply_calib=apply_calib,
            cnv_dir=cnv_dir,
            ax=ax,
            show=show,
            save=save,
            outlier_result=outlier_result,
            **plot_kwargs,
        )

    def plot_khz_variables(
        self,
        system: str,
        variables: list[str],
        hours: range | list[int] | None = None,
        apply_calib: bool = True,
        cnv_dir: str | None = None,
        layout: str = "subplots",
        share_x: bool = True,
        show: bool = True,
        save: str | None = None,
        outlier_config: OutlierConfig | None = None,
        **plot_kwargs,
    ):
        """
        Plot multiple kHz variables.

        This method delegates to hybrid.plotting.plot_khz_variables().

        Parameters
        ----------
        system : str
            FEPC system name
        variables : list of str
            List of variable names to plot
        hours : range or list of int, optional
            Hours to read (default: all available)
        apply_calib : bool, optional
            Apply calibration (default: True)
        cnv_dir : str, optional
            Directory for CNV calibration files
        layout : str, optional
            Plot layout: 'subplots' (default) or 'overlay'
        share_x : bool, optional
            Share x-axis in subplots layout (default: True)
        show : bool, optional
            Show plot (default: True)
        save : str, optional
            Save plot to file
        outlier_config : OutlierConfig, optional
            Outlier detection/handling configuration. ``None`` skips detection.
        **plot_kwargs : dict
            Additional arguments passed to plt.plot()

        Returns
        -------
        tuple
            (fig, axes) matplotlib figure and axes
        """
        from ..outliers import OutlierDetector
        from . import plotting

        # Perform outlier detection for each variable if config provided
        outlier_results = None
        if outlier_config is not None:
            detector = OutlierDetector(config=outlier_config)
            outlier_results = {}
            for var in variables:
                data, _ = self.read_khz_variable(
                    system, var, hours=hours, apply_calib=apply_calib, cnv_dir=cnv_dir
                )
                outlier_results[var] = detector.detect(data)

        return plotting.plot_khz_variables(
            self,
            system,
            variables,
            hours=hours,
            apply_calib=apply_calib,
            cnv_dir=cnv_dir,
            layout=layout,
            share_x=share_x,
            show=show,
            save=save,
            outlier_results=outlier_results,
            **plot_kwargs,
        )

    def plot_rms_variable(
        self,
        system: str,
        variable: str,
        file_idx: int | None = None,
        hours: range | list[int] | None = None,
        ax=None,
        show: bool = True,
        save: str | None = None,
        outlier_config: OutlierConfig | None = None,
        **plot_kwargs,
    ):
        """
        Plot RMS data for a specific variable.

        This method delegates to hybrid.plotting.plot_rms_variable().

        Parameters
        ----------
        system : str
            FEPC system name
        variable : str
            Variable name
        file_idx : int, optional
            Index of RMS file to load (used if hours is None, default: 0)
        hours : range or list of int, optional
            List of hours to load (0-23). If provided, file_idx is ignored.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on (creates new figure if None)
        show : bool, optional
            Show plot (default: True)
        save : str, optional
            Save plot to file
        outlier_config : OutlierConfig, optional
            Outlier detection/handling configuration. ``None`` skips detection.
        **plot_kwargs : dict
            Additional arguments passed to plt.plot()

        Returns
        -------
        tuple
            (fig, ax) matplotlib figure and axes
        """
        from ..outliers import OutlierDetector
        from . import plotting

        # Perform outlier detection if config provided
        outlier_result = None
        if outlier_config is not None:
            data, _ = self.read_rms_variable(
                system, variable, file_idx=file_idx, hours=hours
            )
            outlier_result = OutlierDetector(config=outlier_config).detect(data)

        return plotting.plot_rms_variable(
            self,
            system,
            variable,
            file_idx=file_idx,
            hours=hours,
            ax=ax,
            show=show,
            save=save,
            outlier_result=outlier_result,
            **plot_kwargs,
        )

    def plot_rms_variables(
        self,
        system: str,
        variables: list[str],
        file_idx: int | None = None,
        hours: range | list[int] | None = None,
        layout: str = "subplots",
        share_x: bool = True,
        show: bool = True,
        save: str | None = None,
        outlier_config: OutlierConfig | None = None,
        **plot_kwargs,
    ):
        """
        Plot multiple RMS variables.

        This method delegates to hybrid.plotting.plot_rms_variables().

        Parameters
        ----------
        system : str
            FEPC system name
        variables : list of str
            List of variable names to plot
        file_idx : int, optional
            Index of RMS file to load
        hours : range or list of int, optional
            List of hours to load (0-23)
        layout : str, optional
            Plot layout: 'subplots' (default) or 'overlay'
        share_x : bool, optional
            Share x-axis in subplots layout (default: True)
        show : bool, optional
            Show plot (default: True)
        save : str, optional
            Save plot to file
        outlier_config : OutlierConfig, optional
            Outlier detection/handling configuration. ``None`` skips detection.
        **plot_kwargs : dict
            Additional arguments passed to plt.plot()

        Returns
        -------
        tuple
            (fig, axes) matplotlib figure and axes
        """
        from ..outliers import OutlierDetector
        from . import plotting

        # Perform outlier detection for each variable if config provided
        outlier_results = None
        if outlier_config is not None:
            detector = OutlierDetector(config=outlier_config)
            outlier_results = {}
            for var in variables:
                data, _ = self.read_rms_variable(
                    system, var, file_idx=file_idx, hours=hours
                )
                outlier_results[var] = detector.detect(data)

        return plotting.plot_rms_variables(
            self,
            system,
            variables,
            file_idx=file_idx,
            hours=hours,
            layout=layout,
            share_x=share_x,
            show=show,
            save=save,
            outlier_results=outlier_results,
            **plot_kwargs,
        )

    def plot_khz_with_rms(
        self,
        system: str,
        khz_variable: str,
        rms_variable: str | None = None,
        hours: range | list[int] | None = None,
        apply_calib: bool = True,
        rms_file_idx: int | None = None,
        rms_hours: range | list[int] | None = None,
        show: bool = True,
        save: str | None = None,
    ):
        """
        Plot kHz and RMS data together for comparison.

        This method delegates to hybrid.plotting.plot_khz_with_rms().

        Parameters
        ----------
        system : str
            FEPC system name
        khz_variable : str
            kHz variable name
        rms_variable : str, optional
            RMS variable name (defaults to khz_variable if None)
        hours : range or list of int, optional
            Hours to read for kHz data (also used for RMS if rms_hours is None)
        apply_calib : bool, optional
            Apply calibration to kHz data (default: True)
        rms_file_idx : int, optional
            Index of RMS file to load (ignored if rms_hours is provided)
        rms_hours : range or list of int, optional
            Hours to read for RMS data (defaults to hours if None)
        show : bool, optional
            Show plot (default: True)
        save : str, optional
            Save plot to file

        Returns
        -------
        tuple
            (fig, axes) matplotlib figure and axes array
        """
        from . import plotting

        return plotting.plot_khz_with_rms(
            self,
            system,
            khz_variable,
            rms_variable=rms_variable,
            hours=hours,
            apply_calib=apply_calib,
            rms_file_idx=rms_file_idx,
            rms_hours=rms_hours,
            show=show,
            save=save,
        )
