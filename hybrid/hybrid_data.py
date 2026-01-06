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

import os
import sys
from pathlib import Path
from datetime import datetime, date
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

# Import FEPC readers
try:
    from .kHz.fepc_reader import (
        FEPCConfig,
        CardInfo,
        CalibrationInfo,
        parse_cfg_file,
        read_hour_file,
        apply_calibration,
        load_calibration,
        calibrate_channel,
    )
except ImportError as e:
    print(f"Warning: Could not import fepc_reader: {e}")
    FEPCConfig = None

try:
    from .rms.rms_reader import RMSFileReader, read_rms_file, get_rms_info
except ImportError as e:
    print(f"Warning: Could not import rms_reader: {e}")
    RMSFileReader = None


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
    fepc_systems: List[str] = field(default_factory=list)
    khz_files: Dict[str, List[Path]] = field(default_factory=dict)
    rms_files: Dict[str, List[Path]] = field(default_factory=dict)
    trigger_dirs: Dict[str, List[Path]] = field(
        default_factory=dict
    )  # TRIGGER__YYYY-MM-DD__HH-MM dirs
    trigger_files: Dict[str, List[Path]] = field(default_factory=dict)


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
        base_dir: Union[str, Path],
        date_str: str,
        fepc_system: Optional[str] = None,
        endian: str = "big",
    ):
        self.base_dir = Path(base_dir)
        self.date_str = date_str
        self.date = datetime.strptime(date_str, "%Y-%m-%d").date()
        self.fepc_system = fepc_system
        self.endian = endian

        # MagnetData-like attributes
        self.FileName = f"HybridData_{date_str}"
        self.Groups: Dict[str, Dict] = {}
        self.Keys: List[str] = []
        self.Type = 3  # New type for hybrid data
        self.Data: Dict[str, Any] = {}
        self.units: Dict[str, Tuple] = {}

        # Internal storage
        self._khz_configs: Dict[str, FEPCConfig] = {}
        self._rms_readers: Dict[str, RMSFileReader] = {}
        self._info: Optional[HybridDataInfo] = None

        # Discover available data
        self._discover_data()

    def _discover_data(self) -> None:
        """Discover available data files for the given date"""
        self._info = HybridDataInfo(
            date=self.date,
            base_dir=self.base_dir,
        )

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
                self._info.rms_files[system] = list(rms_dir.glob("*.rms"))

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

    def _build_groups(self) -> None:
        """Build Groups and Keys from discovered data"""
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
                self.Keys.append(group_name)

            # RMS group
            if system in self._info.rms_files:
                group_name = f"rms/{system}"
                self.Groups[group_name] = {
                    "type": "rms",
                    "system": system,
                    "files": self._info.rms_files[system],
                }
                self.Keys.append(group_name)

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

    def getKeys(self) -> List[str]:
        """Return list of available data keys"""
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
        print()

        print("Data availability:")
        print(f"  kHz data:     {'✓' if self._info.khz_available else '✗'}")
        print(f"  RMS data:     {'✓' if self._info.rms_available else '✗'}")
        print(f"  Trigger data: {'✓' if self._info.trigger_available else '✗'}")
        print()

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

    # -------------------------------------------------------------------------
    # kHz Data Methods
    # -------------------------------------------------------------------------

    def load_khz_config(self, system: str) -> Optional[FEPCConfig]:
        """
        Load kHz configuration for a FEPC system

        Parameters
        ----------
        system : str
            FEPC system name

        Returns
        -------
        FEPCConfig or None
            Configuration object or None if not available
        """
        if FEPCConfig is None:
            raise ImportError("fepc_reader module not available")

        if system in self._khz_configs:
            return self._khz_configs[system]

        cfg_key = f"{system}_cfg"
        if cfg_key not in self._info.khz_files:
            print(f"No CFG file found for {system}")
            return None

        cfg_files = self._info.khz_files[cfg_key]
        if not cfg_files:
            return None

        config = parse_cfg_file(str(cfg_files[0]))
        self._khz_configs[system] = config
        return config

    def get_khz_variables(self, system: str) -> Dict[str, List[str]]:
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
        config = self.load_khz_config(system)
        if config is None:
            return {"analog": [], "digital": []}

        analog_vars = []
        digital_vars = []

        for card in config.cards:
            if card.card_type == "ANA":
                for var in card.variable_names:
                    analog_vars.append(f"slot{card.slot}/{var}")
            else:
                for var in card.variable_names:
                    digital_vars.append(f"slot{card.slot}/{var}")

        return {"analog": analog_vars, "digital": digital_vars}

    def read_khz_variable(
        self,
        system: str,
        variable: str,
        slot: Optional[int] = None,
        hours: Optional[List[int]] = None,
        apply_calib: bool = True,
        cnv_dir: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        hours : list of int, optional
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
        config = self.load_khz_config(system)
        if config is None:
            raise ValueError(f"No configuration found for {system}")

        # Find variable slot and channel
        var_slot = slot
        var_channel = None
        var_card = None

        for card in config.cards:
            if variable in card.variable_names:
                var_slot = card.slot
                var_channel = card.variable_names.index(variable)
                var_card = card
                break

        if var_slot is None or var_channel is None:
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
            bin_files = filtered_files

        if not bin_files:
            raise ValueError(f"No bin files found for slot {var_slot}")

        # Read data from all files
        all_data = []
        for bin_file in bin_files:
            print(f"Reading {bin_file.name}...", flush=True)
            hour_data = read_hour_file(
                str(bin_file),
                var_card.card_type,
                endian=self.endian,
            )
            all_data.append(hour_data[:, var_channel])

        data = np.concatenate(all_data)

        # Apply calibration
        if apply_calib and var_card.card_type == "ANA":
            if cnv_dir is None:
                cnv_dir = str(khz_dir)
            data = calibrate_channel(data, var_card, var_channel, cnv_dir)

        # Create time array (1 kHz sampling)
        sampling_freq = 1000  # 1 kHz
        time = np.arange(len(data)) / sampling_freq

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
                f"File index {file_idx} out of range (max: {len(rms_files)-1})"
            )

        rms_file = rms_files[file_idx]
        reader = RMSFileReader(str(rms_file), endian=self.endian)
        return reader.read()

    def get_rms_variables(self, system: str, file_idx: int = 0) -> pd.DataFrame:
        """
        Get RMS variable information

        Parameters
        ----------
        system : str
            FEPC system name
        file_idx : int, optional
            Index of RMS file (default: 0)

        Returns
        -------
        pd.DataFrame
            Variable information
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

    # -------------------------------------------------------------------------
    # Trigger Data Methods
    # -------------------------------------------------------------------------

    def list_trigger_events(self, system: str) -> List[Dict[str, Any]]:
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

    def list_trigger_files(self, system: str) -> List[Path]:
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

    def getData(self, key: Optional[str] = None) -> Any:
        """
        Get data for a specific key (MagnetData-compatible interface)

        Parameters
        ----------
        key : str, optional
            Data key in format 'type/system' or 'type/system/variable'

        Returns
        -------
        Data (type depends on the requested data)
        """
        if key is None:
            return self.Data

        parts = key.split("/")
        if len(parts) < 2:
            raise ValueError(f"Invalid key format: {key}")

        data_type = parts[0]
        system = parts[1]

        if data_type == "kHz":
            if len(parts) >= 3:
                variable = parts[2]
                return self.read_khz_variable(system, variable)
            else:
                return self.get_khz_variables(system)

        elif data_type == "rms":
            if len(parts) >= 3:
                # Could add variable selection here
                pass
            return self.load_rms_data(system)

        elif data_type == "trigger":
            return self.list_trigger_files(system)

        else:
            raise ValueError(f"Unknown data type: {data_type}")

    def Units(self, debug: bool = False) -> None:
        """Set units for data (placeholder for MagnetData compatibility)"""
        # TODO: Implement unit handling
        pass

    def getUnitKey(self, key: str) -> Tuple:
        """Get unit for a specific key"""
        return self.units.get(key, ())

    # -------------------------------------------------------------------------
    # Plotting Methods
    # -------------------------------------------------------------------------

    def plot_khz_variable(
        self,
        system: str,
        variable: str,
        hours: Optional[List[int]] = None,
        apply_calib: bool = True,
        cnv_dir: Optional[str] = None,
        ax=None,
        show: bool = True,
        save: Optional[str] = None,
        **plot_kwargs,
    ):
        """
        Plot kHz data for a specific variable

        Parameters
        ----------
        system : str
            FEPC system name
        variable : str
            Variable name
        hours : list of int, optional
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
        **plot_kwargs : dict
            Additional arguments passed to plt.plot()

        Returns
        -------
        tuple
            (fig, ax) matplotlib figure and axes
        """
        import matplotlib.pyplot as plt

        # Read data
        data, time = self.read_khz_variable(
            system, variable, hours=hours, apply_calib=apply_calib, cnv_dir=cnv_dir
        )

        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        else:
            fig = ax.get_figure()

        # Get unit if available
        config = self.load_khz_config(system)
        unit = ""
        if config:
            for card in config.cards:
                if variable in card.variable_names:
                    idx = card.variable_names.index(variable)
                    if idx < len(card.calibrations) and card.calibrations[idx]:
                        unit = card.calibrations[idx].unit or ""
                    break

        # Plot
        label = plot_kwargs.pop("label", f"{variable}")
        ax.plot(time, data, label=label, **plot_kwargs)
        ax.set_xlabel("Time (s)")
        ylabel = f"{variable}"
        if unit:
            ylabel += f" [{unit}]"
        ax.set_ylabel(ylabel)
        ax.set_title(f"{system} - {variable} ({self.date_str})")
        ax.grid(True, alpha=0.3)
        ax.legend()

        if save:
            fig.savefig(save, dpi=150, bbox_inches="tight")
            print(f"Saved plot to {save}")

        if show:
            plt.show()

        return fig, ax

    def plot_rms_variable(
        self,
        system: str,
        variable: str,
        file_idx: int = 0,
        ax=None,
        show: bool = True,
        save: Optional[str] = None,
        **plot_kwargs,
    ):
        """
        Plot RMS data for a specific variable

        Parameters
        ----------
        system : str
            FEPC system name
        variable : str
            Variable name (column in RMS DataFrame)
        file_idx : int, optional
            Index of RMS file to load (default: 0)
        ax : matplotlib.axes.Axes, optional
            Axes to plot on (creates new figure if None)
        show : bool, optional
            Show plot (default: True)
        save : str, optional
            Save plot to file
        **plot_kwargs : dict
            Additional arguments passed to plt.plot()

        Returns
        -------
        tuple
            (fig, ax) matplotlib figure and axes
        """
        import matplotlib.pyplot as plt

        # Load RMS data
        rms_df = self.load_rms_data(system, file_idx=file_idx)

        if variable not in rms_df.columns:
            available = ", ".join(rms_df.columns[:10])
            raise ValueError(
                f"Variable '{variable}' not found. Available: {available}..."
            )

        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        else:
            fig = ax.get_figure()

        # Use index as time if available, otherwise create time array
        if "Time" in rms_df.columns:
            time = rms_df["Time"].values
        else:
            time = np.arange(len(rms_df))

        # Plot
        label = plot_kwargs.pop("label", f"{variable} (RMS)")
        ax.plot(time, rms_df[variable].values, label=label, **plot_kwargs)
        ax.set_xlabel("Time (s)" if "Time" in rms_df.columns else "Sample")
        ax.set_ylabel(variable)
        ax.set_title(f"{system} RMS - {variable} ({self.date_str})")
        ax.grid(True, alpha=0.3)
        ax.legend()

        if save:
            fig.savefig(save, dpi=150, bbox_inches="tight")
            print(f"Saved plot to {save}")

        if show:
            plt.show()

        return fig, ax

    def plot_khz_with_rms(
        self,
        system: str,
        khz_variable: str,
        rms_variable: Optional[str] = None,
        hours: Optional[List[int]] = None,
        apply_calib: bool = True,
        rms_file_idx: int = 0,
        show: bool = True,
        save: Optional[str] = None,
    ):
        """
        Plot kHz and RMS data together for comparison

        Parameters
        ----------
        system : str
            FEPC system name
        khz_variable : str
            kHz variable name
        rms_variable : str, optional
            RMS variable name (defaults to khz_variable if None)
        hours : list of int, optional
            Hours to read for kHz data
        apply_calib : bool, optional
            Apply calibration to kHz data (default: True)
        rms_file_idx : int, optional
            Index of RMS file to load (default: 0)
        show : bool, optional
            Show plot (default: True)
        save : str, optional
            Save plot to file

        Returns
        -------
        tuple
            (fig, axes) matplotlib figure and axes array
        """
        import matplotlib.pyplot as plt

        if rms_variable is None:
            rms_variable = khz_variable

        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)

        # Plot kHz data
        try:
            khz_data, khz_time = self.read_khz_variable(
                system, khz_variable, hours=hours, apply_calib=apply_calib
            )
            axes[0].plot(
                khz_time, khz_data, "b-", linewidth=0.5, label=f"{khz_variable} (kHz)"
            )
            axes[0].set_xlabel("Time (s)")
            axes[0].set_ylabel(khz_variable)
            axes[0].set_title(f"kHz Data: {khz_variable}")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()
        except Exception as e:
            axes[0].text(
                0.5,
                0.5,
                f"Error loading kHz data:\n{e}",
                ha="center",
                va="center",
                transform=axes[0].transAxes,
            )
            axes[0].set_title(f"kHz Data: {khz_variable} (ERROR)")

        # Plot RMS data
        try:
            rms_df = self.load_rms_data(system, file_idx=rms_file_idx)
            if rms_variable in rms_df.columns:
                if "Time" in rms_df.columns:
                    rms_time = rms_df["Time"].values
                else:
                    rms_time = np.arange(len(rms_df))
                axes[1].plot(
                    rms_time,
                    rms_df[rms_variable].values,
                    "r-",
                    marker=".",
                    markersize=2,
                    label=f"{rms_variable} (RMS)",
                )
                axes[1].set_xlabel("Time (s)" if "Time" in rms_df.columns else "Sample")
                axes[1].set_ylabel(rms_variable)
            else:
                axes[1].text(
                    0.5,
                    0.5,
                    f"Variable '{rms_variable}' not found in RMS data",
                    ha="center",
                    va="center",
                    transform=axes[1].transAxes,
                )
            axes[1].set_title(f"RMS Data: {rms_variable}")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()
        except Exception as e:
            axes[1].text(
                0.5,
                0.5,
                f"Error loading RMS data:\n{e}",
                ha="center",
                va="center",
                transform=axes[1].transAxes,
            )
            axes[1].set_title(f"RMS Data: {rms_variable} (ERROR)")

        fig.suptitle(f"{system} - {self.date_str}", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save:
            fig.savefig(save, dpi=150, bbox_inches="tight")
            print(f"Saved plot to {save}")

        if show:
            plt.show()

        return fig, axes


def list_available_dates(
    base_dir: Union[str, Path], data_type: str = "kHz"
) -> List[str]:
    """
    List available dates for a given data type

    Parameters
    ----------
    base_dir : str or Path
        Base directory
    data_type : str
        Data type: 'kHz', 'rms', or 'trigger'

    Returns
    -------
    list of str
        List of date strings in YYYY-MM-DD format
    """
    base_path = Path(base_dir) / data_type
    if not base_path.exists():
        return []

    dates = set()
    for item in base_path.iterdir():
        if item.is_dir():
            if data_type == "trigger":
                # Trigger directories are named TRIGGER__YYYY-MM-DD__HH-MM
                if item.name.startswith("TRIGGER__"):
                    parts = item.name.split("__")
                    if len(parts) >= 2:
                        date_str = parts[1]
                        try:
                            datetime.strptime(date_str, "%Y-%m-%d")
                            dates.add(date_str)
                        except ValueError:
                            pass
            else:
                # kHz and rms directories are named YYYY-MM-DD
                try:
                    datetime.strptime(item.name, "%Y-%m-%d")
                    dates.add(item.name)
                except ValueError:
                    pass

    return sorted(dates)


def main():
    """Main function for CLI usage"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Read hybrid magnet data (kHz, RMS, Trigger)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List available dates
    python hybrid_data.py --base-dir /data/hybrid --list-dates

    # Show summary for a specific date
    python hybrid_data.py --base-dir /data/hybrid --date 2025-01-06

    # Show kHz variables
    python hybrid_data.py --base-dir /data/hybrid --date 2025-01-06 --khz-vars FEPC-LNCMI

    # Show RMS variables
    python hybrid_data.py --base-dir /data/hybrid --date 2025-01-06 --rms-vars FEPC-LNCMI

    # Plot a kHz variable
    python hybrid_data.py -d 2025-01-06 -s FEPC-LNCMI --plot-khz ALIM1_J1

    # Plot a kHz variable for specific hours without calibration
    python hybrid_data.py -d 2025-01-06 -s FEPC-LNCMI --plot-khz ALIM1_J1 --hours 0,1,2 --no-calib

    # Plot an RMS variable
    python hybrid_data.py -d 2025-01-06 -s FEPC-LNCMI --plot-rms ALIM1_J1

    # Plot kHz and RMS together
    python hybrid_data.py -d 2025-01-06 -s FEPC-LNCMI --plot-both ALIM1_J1

    # Save plot to file
    python hybrid_data.py -d 2025-01-06 -s FEPC-LNCMI --plot-khz ALIM1_J1 --save output.png
        """,
    )

    parser.add_argument(
        "--base-dir",
        "-b",
        type=str,
        default="/home/LNCMI-G/christophe.trophime/LNCMIG-Data/CEA/",
        help="Base directory containing kHz, rms, trigger subdirectories",
    )

    parser.add_argument(
        "--date",
        "-d",
        type=str,
        help="Date in YYYY-MM-DD format",
    )

    parser.add_argument(
        "--fepc-system",
        "-s",
        type=str,
        choices=FEPC_SYSTEMS,
        help="FEPC system to use",
    )

    parser.add_argument(
        "--endian",
        "-e",
        type=str,
        choices=["big", "little"],
        default="big",
        help="Endianness of binary data (default: big)",
    )

    parser.add_argument(
        "--list-dates",
        action="store_true",
        help="List available dates",
    )

    parser.add_argument(
        "--khz-vars",
        type=str,
        metavar="SYSTEM",
        help="Show kHz variables for a FEPC system",
    )

    parser.add_argument(
        "--rms-vars",
        type=str,
        metavar="SYSTEM",
        help="Show RMS variables for a FEPC system",
    )

    # Plotting arguments
    parser.add_argument(
        "--plot-khz",
        type=str,
        metavar="VARIABLE",
        help="Plot a kHz variable (requires --fepc-system)",
    )

    parser.add_argument(
        "--plot-rms",
        type=str,
        metavar="VARIABLE",
        help="Plot an RMS variable (requires --fepc-system)",
    )

    parser.add_argument(
        "--plot-both",
        type=str,
        metavar="VARIABLE",
        help="Plot kHz and RMS data together (requires --fepc-system)",
    )

    parser.add_argument(
        "--rms-var",
        type=str,
        metavar="VARIABLE",
        help="RMS variable name for --plot-both (defaults to kHz variable name)",
    )

    parser.add_argument(
        "--hours",
        type=str,
        metavar="HOURS",
        help="Hours to plot for kHz data (comma-separated, e.g., '0,1,2')",
    )

    parser.add_argument(
        "--no-calib",
        action="store_true",
        help="Do not apply calibration to kHz data",
    )

    parser.add_argument(
        "--save",
        type=str,
        metavar="FILE",
        help="Save plot to file",
    )

    args = parser.parse_args()

    # List dates
    if args.list_dates:
        print("Available dates:")
        for data_type in ["kHz", "rms", "trigger"]:
            dates = list_available_dates(args.base_dir, data_type)
            if dates:
                print(
                    f"  {data_type}: {', '.join(dates[:5])}"
                    + (f" ... ({len(dates)} total)" if len(dates) > 5 else "")
                )
        return

    # Require date for other operations
    if not args.date:
        parser.print_help()
        return

    # Create HybridData instance
    try:
        data = HybridData(
            args.base_dir,
            args.date,
            fepc_system=args.fepc_system,
            endian=args.endian,
        )
    except Exception as e:
        print(f"Error creating HybridData: {e}")
        return

    # Show summary
    data.print_summary()

    # Show kHz variables
    if args.khz_vars:
        print(f"\nkHz Variables for {args.khz_vars}:")
        vars_info = data.get_khz_variables(args.khz_vars)
        print(f"  Analog ({len(vars_info['analog'])}):")
        for var in vars_info["analog"][:10]:
            print(f"    {var}")
        if len(vars_info["analog"]) > 10:
            print(f"    ... and {len(vars_info['analog']) - 10} more")
        print(f"  Digital ({len(vars_info['digital'])}):")
        for var in vars_info["digital"][:10]:
            print(f"    {var}")
        if len(vars_info["digital"]) > 10:
            print(f"    ... and {len(vars_info['digital']) - 10} more")

    # Show RMS variables
    if args.rms_vars:
        print(f"\nRMS Variables for {args.rms_vars}:")
        try:
            vars_df = data.get_rms_variables(args.rms_vars)
            print(vars_df.to_string())
        except Exception as e:
            print(f"  Error: {e}")

    # Parse hours if provided
    hours = None
    if args.hours:
        try:
            hours = [int(h.strip()) for h in args.hours.split(",")]
        except ValueError:
            print(
                f"Error: Invalid hours format '{args.hours}'. Use comma-separated integers."
            )
            return

    # Plot kHz variable
    if args.plot_khz:
        if not args.fepc_system:
            print("Error: --fepc-system is required for plotting")
            return
        try:
            print(f"\nPlotting kHz variable: {args.plot_khz}")
            data.plot_khz_variable(
                args.fepc_system,
                args.plot_khz,
                hours=hours,
                apply_calib=not args.no_calib,
                save=args.save,
            )
        except Exception as e:
            print(f"Error plotting kHz variable: {e}")

    # Plot RMS variable
    if args.plot_rms:
        if not args.fepc_system:
            print("Error: --fepc-system is required for plotting")
            return
        try:
            print(f"\nPlotting RMS variable: {args.plot_rms}")
            data.plot_rms_variable(
                args.fepc_system,
                args.plot_rms,
                save=args.save,
            )
        except Exception as e:
            print(f"Error plotting RMS variable: {e}")

    # Plot both kHz and RMS
    if args.plot_both:
        if not args.fepc_system:
            print("Error: --fepc-system is required for plotting")
            return
        try:
            rms_var = args.rms_var if args.rms_var else args.plot_both
            print(f"\nPlotting kHz ({args.plot_both}) and RMS ({rms_var})")
            data.plot_khz_with_rms(
                args.fepc_system,
                args.plot_both,
                rms_variable=rms_var,
                hours=hours,
                apply_calib=not args.no_calib,
                save=args.save,
            )
        except Exception as e:
            print(f"Error plotting: {e}")


if __name__ == "__main__":
    main()
