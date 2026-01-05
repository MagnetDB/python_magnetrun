"""
Configuration constants and dataclasses for magnetrun analysis.

This module provides structured configuration for the magnetrun analysis pipeline,
replacing the original tuple-based setup() function with typed dataclasses.

The configuration hierarchy is:
- AnalysisConfig: Top-level configuration combining all settings
  - SiteConfig: Site-specific channel mappings (M8, M9, M10)
  - ChannelMapping: Reference to current channel mappings
  - VoltageChannelMapping: Voltage probe channels per reference
  - ThresholdConfig: Detection thresholds for each channel
  - ColorConfig: Visualization color mappings

Example usage::

    from python_magnetrun.analysis.config import AnalysisConfig, SITE_CONFIGS

    # Get complete configuration for M9
    config = AnalysisConfig.for_site("M9")

    # Access site-specific mappings
    current_channel = config.site.reference_gr1_current  # "IH"

    # Get threshold for signal detection
    threshold = config.thresholds.get("Courant_GR1")  # 0.5
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# =============================================================================
# Sampling rates (Hz)
# =============================================================================
SAMPLING_RATE_OVERVIEW: float = 1.0
"""Sampling rate for overview TDMS files (1 Hz, 1 sample/second)."""

SAMPLING_RATE_ARCHIVE: float = 120.0
"""Sampling rate for archive TDMS files (120 Hz)."""

SAMPLING_RATE_INCIDENTS: float = 4800.0
"""Sampling rate for incident files: default, trigger, spike (4800 Hz)."""


# =============================================================================
# Time offset calculations
# =============================================================================
def get_time_offset(sampling_rate: float) -> float:
    """
    Calculate time offset for downsampled data.

    TDMS downsampled data represents the center of sampling windows,
    so we add half the sampling period as offset.

    Parameters
    ----------
    sampling_rate : float
        Sampling rate in Hz

    Returns
    -------
    float
        Time offset in seconds
    """
    return (1.0 / sampling_rate) / 2.0


TIME_OFFSET_OVERVIEW: float = get_time_offset(SAMPLING_RATE_OVERVIEW)
"""Time offset for overview data (0.5 seconds)."""

TIME_OFFSET_ARCHIVE: float = get_time_offset(SAMPLING_RATE_ARCHIVE)
"""Time offset for archive data (~0.00417 seconds)."""

TIME_OFFSET_INCIDENTS: float = get_time_offset(SAMPLING_RATE_INCIDENTS)
"""Time offset for incident data (~0.000104 seconds)."""


# =============================================================================
# Analysis thresholds and defaults
# =============================================================================
LAG_THRESHOLD_RATIO: float = 0.2
"""
Threshold ratio for lag detection.

If |lag| / duration >= this ratio, the lag is considered significant
and flagged as potentially unreliable.
"""

DEFAULT_WINDOW_SIZE: int = 50
"""Default rolling window size for signal processing."""

DEFAULT_BINS: int = 10
"""Default number of bins for histogram analysis."""

DEFAULT_LEVELS: int = 4
"""Default number of levels for piecewise linear fitting."""

CURRENT_THRESHOLD_FOR_MODE: float = 300.0
"""Minimum current (A) to consider for mode detection (normal vs eco)."""

MODE_INTERCEPT_TOLERANCE: float = 10.0
"""Maximum acceptable intercept value for mode detection fit."""


# =============================================================================
# Default paths (can be overridden via environment variables)
# =============================================================================
DEFAULT_DATA_DIR: str = os.environ.get(
    "MAGNETRUN_DATA_DIR",
    "/home/LNCMI-G/christophe.trophime/LNCMIG-Data/srv-data-install",
)
"""
Default directory for pupitre and log data files.

Can be overridden by setting the MAGNETRUN_DATA_DIR environment variable.
"""

DEFAULT_PIGBROTHER_DATA_DIR: str = os.environ.get(
    "MAGNETRUN_PIGBROTHER_DATA_DIR",
    "/home/LNCMI-G/christophe.trophime/github/python_magnetrun/pigbrotherdata/Fichiers_Data",
)
"""
Default directory for pupitre and log data files.

Can be overridden by setting the MAGNETRUN_DATA_DIR environment variable.
"""


# =============================================================================
# TDMS group names
# =============================================================================
DEFAULT_GROUP: str = "Courants_Alimentations"
"""Default TDMS group for current/reference channels."""


# =============================================================================
# Color configuration for visualization
# =============================================================================
@dataclass(frozen=True)
class ColorConfig:
    """
    Color mapping for visualization.

    Maps signal type prefixes to matplotlib colors.

    Attributes
    ----------
    up_color : str
        Color for 'U' (up/voltage) signals
    plateau_color : str
        Color for 'P' (plateau) signals
    down_color : str
        Color for 'D' (down) signals
    """

    up_color: str = "red"
    plateau_color: str = "green"
    down_color: str = "blue"

    def get(self, signal_type: str) -> str:
        """
        Get color for a signal type.

        Parameters
        ----------
        signal_type : str
            Signal type prefix: 'U', 'P', or 'D'

        Returns
        -------
        str
            Matplotlib color string
        """
        mapping = {
            "U": self.up_color,
            "P": self.plateau_color,
            "D": self.down_color,
        }
        return mapping.get(signal_type, "black")

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary format (backward compatibility)."""
        return {"U": self.up_color, "P": self.plateau_color, "D": self.down_color}


# =============================================================================
# Channel mapping configuration
# =============================================================================
@dataclass(frozen=True)
class ChannelMapping:
    """
    Mapping between reference and current channels.

    Maps TDMS reference channel names to their corresponding current channels.

    Attributes
    ----------
    reference_gr1 : str
        Current channel name for Référence_GR1
    reference_gr2 : str
        Current channel name for Référence_GR2
    """

    reference_gr1: str = "Courant_GR1"
    reference_gr2: str = "Courant_GR2"

    def get_current_channel(self, reference_key: str) -> str:
        """
        Get the current channel name for a reference key.

        Parameters
        ----------
        reference_key : str
            Reference channel name (e.g., "Référence_GR1")

        Returns
        -------
        str
            Corresponding current channel name

        Raises
        ------
        KeyError
            If reference_key is not a valid reference channel
        """
        mapping = {
            "Référence_GR1": self.reference_gr1,
            "Référence_GR2": self.reference_gr2,
        }
        if reference_key not in mapping:
            raise KeyError(
                f"Unknown reference key: {reference_key}. "
                f"Valid keys: {list(mapping.keys())}"
            )
        return mapping[reference_key]

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary format (backward compatibility)."""
        return {
            "Référence_GR1": self.reference_gr1,
            "Référence_GR2": self.reference_gr2,
        }


@dataclass(frozen=True)
class VoltageChannelMapping:
    """
    Mapping of voltage probe channels per reference group.

    Maps reference channels to their associated voltage measurement channels.
    This includes internal (Interne) and external (Externe) probes.

    Attributes
    ----------
    reference_gr1_channels : tuple[str, ...]
        Voltage channels associated with GR1 (typically internal probes)
    reference_gr2_channels : tuple[str, ...]
        Voltage channels associated with GR2 (typically external probes)
    """

    reference_gr1_channels: tuple = (
        "ALL_internes",
        "Interne1",
        "Interne2",
        "Interne3",
        "Interne4",
        "Interne5",
        "Interne6",
        "Interne7",
    )
    reference_gr2_channels: tuple = (
        "ALL_externes",
        "Externe1",
        "Externe2",
    )

    def get_channels(self, reference_key: str) -> tuple:
        """
        Get voltage channels for a reference key.

        Parameters
        ----------
        reference_key : str
            Reference channel name

        Returns
        -------
        tuple
            Associated voltage channel names
        """
        mapping = {
            "Référence_GR1": self.reference_gr1_channels,
            "Référence_GR2": self.reference_gr2_channels,
        }
        return mapping.get(reference_key, ())

    def to_dict(self) -> Dict[str, List[str]]:
        """Convert to dictionary format (backward compatibility)."""
        return {
            "Référence_GR1": list(self.reference_gr1_channels),
            "Référence_GR2": list(self.reference_gr2_channels),
        }


# =============================================================================
# Site-specific configuration
# =============================================================================
@dataclass(frozen=True)
class SiteConfig:
    """
    Configuration specific to a measurement site (M8, M9, M10).

    Different sites have different channel naming conventions and
    probe configurations. This dataclass captures those differences.

    Attributes
    ----------
    name : str
        Site identifier (e.g., "M9")
    reference_gr1_current : str
        Pupitre current channel for GR1 reference
    reference_gr2_current : str
        Pupitre current channel for GR2 reference
    reference_gr1_flow : str
        Flow measurement channel for GR1
    reference_gr2_flow : str
        Flow measurement channel for GR2
    reference_gr1_rpm : str
        Pump RPM channel for GR1
    reference_gr2_rpm : str
        Pump RPM channel for GR2
    reference_gr1_pin : str
        Inlet pressure channel for GR1
    reference_gr2_pin : str
        Inlet pressure channel for GR2
    voltage_channels_gr1 : tuple[str, ...]
        Voltage probe channels for GR1 in pupitre data
    voltage_channels_gr2 : tuple[str, ...]
        Voltage probe channels for GR2 in pupitre data

    Notes
    -----
    M9 uses IH/IB for GR1/GR2 respectively (H=High, B=Bas/Bottom).
    M8 and M10 swap this: IB/IH for GR1/GR2.
    """

    name: str
    reference_gr1_current: str
    reference_gr2_current: str
    reference_gr1_flow: str
    reference_gr2_flow: str
    reference_gr1_rpm: str
    reference_gr2_rpm: str
    reference_gr1_pin: str
    reference_gr2_pin: str
    voltage_channels_gr1: tuple = field(default_factory=tuple)
    voltage_channels_gr2: tuple = field(default_factory=tuple)

    def get_pupitre_channel(self, reference_key: str) -> str:
        """
        Get pupitre current channel for a reference key.

        Parameters
        ----------
        reference_key : str
            Reference channel name (e.g., "Référence_GR1")

        Returns
        -------
        str
            Corresponding pupitre channel name (e.g., "IH")
        """
        mapping = {
            "Référence_GR1": self.reference_gr1_current,
            "Référence_GR2": self.reference_gr2_current,
        }
        return mapping.get(reference_key, "")

    def get_flow_channel(self, reference_key: str) -> str:
        """Get flow channel for a reference key."""
        mapping = {
            "Référence_GR1": self.reference_gr1_flow,
            "Référence_GR2": self.reference_gr2_flow,
        }
        return mapping.get(reference_key, "")

    def get_rpm_channel(self, reference_key: str) -> str:
        """Get RPM channel for a reference key."""
        mapping = {
            "Référence_GR1": self.reference_gr1_rpm,
            "Référence_GR2": self.reference_gr2_rpm,
        }
        return mapping.get(reference_key, "")

    def get_pin_channel(self, reference_key: str) -> str:
        """Get inlet pressure channel for a reference key."""
        mapping = {
            "Référence_GR1": self.reference_gr1_pin,
            "Référence_GR2": self.reference_gr2_pin,
        }
        return mapping.get(reference_key, "")

    def get_voltage_channels(self, reference_key: str) -> tuple:
        """Get voltage channels for a reference key."""
        mapping = {
            "Référence_GR1": self.voltage_channels_gr1,
            "Référence_GR2": self.voltage_channels_gr2,
        }
        return mapping.get(reference_key, ())

    def to_pupitre_dict(self) -> Dict[str, str]:
        """
        Convert to pupitre_dict format (backward compatibility).

        Returns the dictionary format used by the original setup() function.
        """
        return {
            "Référence_GR1": self.reference_gr1_current,
            "Référence_GR2": self.reference_gr2_current,
            "Référence_GR1_Q": self.reference_gr1_flow,
            "Référence_GR2_Q": self.reference_gr2_flow,
            "Référence_GR1_Rpm": self.reference_gr1_rpm,
            "Référence_GR2_Rpm": self.reference_gr2_rpm,
            "Référence_GR1_Pin": self.reference_gr1_pin,
            "Référence_GR2_Pin": self.reference_gr2_pin,
        }

    def to_upupitre_dict(self) -> Dict[str, List[str]]:
        """Convert to upupitre_dict format (backward compatibility)."""
        return {
            "Référence_GR1": list(self.voltage_channels_gr1),
            "Référence_GR2": list(self.voltage_channels_gr2),
        }


# =============================================================================
# Pre-defined site configurations
# =============================================================================
SITE_CONFIGS: Dict[str, SiteConfig] = {
    "M9": SiteConfig(
        name="M9",
        reference_gr1_current="IH",
        reference_gr2_current="IB",
        reference_gr1_flow="FlowH",
        reference_gr2_flow="FlowB",
        reference_gr1_rpm="RpmH",
        reference_gr2_rpm="RpmB",
        reference_gr1_pin="HPH",
        reference_gr2_pin="HPB",
        voltage_channels_gr1=("UH",),
        voltage_channels_gr2=("UB", "Ucoil15", "Ucoil16"),
    ),
    "M8": SiteConfig(
        name="M8",
        reference_gr1_current="IB",  # Note: swapped from M9
        reference_gr2_current="IH",
        reference_gr1_flow="FlowB",
        reference_gr2_flow="FlowH",
        reference_gr1_rpm="RpmB",
        reference_gr2_rpm="RpmH",
        reference_gr1_pin="HPB",
        reference_gr2_pin="HPH",
        voltage_channels_gr1=("UB", "Ucoil15", "Ucoil16"),  # Note: swapped
        voltage_channels_gr2=("UH",),
    ),
    "M10": SiteConfig(
        name="M10",
        reference_gr1_current="IB",  # Same as M8
        reference_gr2_current="IH",
        reference_gr1_flow="FlowB",
        reference_gr2_flow="FlowH",
        reference_gr1_rpm="RpmB",
        reference_gr2_rpm="RpmH",
        reference_gr1_pin="HPB",
        reference_gr2_pin="HPH",
        voltage_channels_gr1=("UB", "Ucoil15", "Ucoil16"),
        voltage_channels_gr2=("UH",),
    ),
}
"""Pre-defined configurations for measurement sites M8, M9, and M10."""


def get_site_config(site: str) -> SiteConfig:
    """
    Get site configuration by name.

    Parameters
    ----------
    site : str
        Site identifier (M8, M9, M10)

    Returns
    -------
    SiteConfig
        Site configuration

    Raises
    ------
    ValueError
        If site is unknown

    Examples
    --------
    >>> config = get_site_config("M9")
    >>> config.reference_gr1_current
    'IH'
    """
    if site not in SITE_CONFIGS:
        raise ValueError(
            f"Unknown site: {site}. Available: {list(SITE_CONFIGS.keys())}"
        )
    return SITE_CONFIGS[site]


# =============================================================================
# Threshold configuration
# =============================================================================
@dataclass
class ThresholdConfig:
    """
    Threshold values for signal detection.

    Defines detection thresholds for each measurement channel.
    These thresholds are used for:
    - Signature detection (identifying regime changes)
    - Noise filtering
    - Signal validation

    Attributes
    ----------
    thresholds : Dict[str, float]
        Mapping of channel names to threshold values
    """

    thresholds: Dict[str, float] = field(default_factory=dict)

    def get(self, key: str, default: float = 0.1) -> float:
        """
        Get threshold for a given channel.

        Parameters
        ----------
        key : str
            Channel name
        default : float, optional
            Default value if channel not found (default: 0.1)

        Returns
        -------
        float
            Threshold value for the channel
        """
        return self.thresholds.get(key, default)

    def __getitem__(self, key: str) -> float:
        """Allow dictionary-style access: config.thresholds['IH']."""
        return self.thresholds[key]

    def __contains__(self, key: str) -> bool:
        """Allow 'in' operator: 'IH' in config.thresholds."""
        return key in self.thresholds

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format (backward compatibility)."""
        return dict(self.thresholds)

    @classmethod
    def default(cls) -> "ThresholdConfig":
        """
        Create default threshold configuration.

        Returns the same threshold values as the original setup() function.

        Returns
        -------
        ThresholdConfig
            Configuration with default threshold values
        """
        thresholds = {
            # Reference and current channels (TDMS)
            "Référence_GR1": 0.5,
            "Courant_GR1": 0.5,
            "Référence_GR2": 0.5,
            "Courant_GR2": 0.5,
            # Aggregate voltage channels
            "ALL_internes": 0.1,
            "ALL_externes": 0.1,
            # Internal voltage probes (Interne1-7)
            **{f"Interne{i}": 1.0e-2 for i in range(1, 8)},
            # External voltage probes
            "Externe1": 0.1,
            "Externe2": 0.1,
            # Pupitre current channels
            "IH": 1.0,
            "IB": 1.0,
            # Pupitre voltage channels
            "UH": 0.1,
            "UB": 0.1,
            # Individual coil voltages (Ucoil1-14)
            **{f"Ucoil{i}": 1.0e-2 for i in range(1, 15)},
            # Bitter coil voltages
            "Ucoil15": 0.1,
            "Ucoil16": 0.1,
            # Water flow and pressure
            "debitbrut": 25.0,
            "Pmagnet": 0.1,
        }
        return cls(thresholds=thresholds)


# =============================================================================
# Complete analysis configuration
# =============================================================================
@dataclass
class AnalysisConfig:
    """
    Complete analysis configuration.

    This is the main configuration class that combines all settings
    needed for the analysis pipeline.

    Attributes
    ----------
    site : SiteConfig
        Site-specific configuration
    channels : ChannelMapping
        Reference to current channel mapping
    voltage_channels : VoltageChannelMapping
        Voltage probe channel mapping
    thresholds : ThresholdConfig
        Detection thresholds
    colors : ColorConfig
        Visualization colors

    Examples
    --------
    >>> config = AnalysisConfig.for_site("M9")
    >>> print(config.site.reference_gr1_current)
    IH
    >>> print(config.thresholds.get("Courant_GR1"))
    0.5
    """

    site: SiteConfig
    channels: ChannelMapping = field(default_factory=ChannelMapping)
    voltage_channels: VoltageChannelMapping = field(
        default_factory=VoltageChannelMapping
    )
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig.default)
    colors: ColorConfig = field(default_factory=ColorConfig)

    @classmethod
    def for_site(cls, site_name: str) -> "AnalysisConfig":
        """
        Create configuration for a specific site.

        Parameters
        ----------
        site_name : str
            Site identifier (e.g., "M8", "M9", "M10")

        Returns
        -------
        AnalysisConfig
            Complete configuration for the specified site

        Raises
        ------
        ValueError
            If site_name is not a known site

        Examples
        --------
        >>> config = AnalysisConfig.for_site("M9")
        >>> config.site.name
        'M9'
        """
        if site_name not in SITE_CONFIGS:
            available = ", ".join(sorted(SITE_CONFIGS.keys()))
            raise ValueError(f"Unknown site: {site_name}. Available sites: {available}")
        return cls(site=SITE_CONFIGS[site_name])

    def get_pupitre_channel(self, reference_key: str) -> str:
        """
        Get pupitre channel for a reference key.

        Convenience method that delegates to site configuration.

        Parameters
        ----------
        reference_key : str
            Reference channel name

        Returns
        -------
        str
            Pupitre channel name
        """
        return self.site.get_pupitre_channel(reference_key)

    def get_current_channel(self, reference_key: str) -> str:
        """
        Get TDMS current channel for a reference key.

        Parameters
        ----------
        reference_key : str
            Reference channel name

        Returns
        -------
        str
            TDMS current channel name
        """
        return self.channels.get_current_channel(reference_key)


# =============================================================================
# Backward compatibility: setup() function replacement
# =============================================================================
def get_legacy_config(site_name: Optional[str] = None) -> tuple:
    """
    Get configuration in the legacy tuple format.

    This function provides backward compatibility with code that
    expects the original setup() function return format.

    Parameters
    ----------
    site_name : str, optional
        If provided, include site-specific pupitre dicts.
        If None, return all sites' configurations.

    Returns
    -------
    tuple
        (color_dict, channels_dict, uchannels_dict,
         pupitre_dict, upupitre_dict, threshold_dict)

    Examples
    --------
    >>> color_dict, channels_dict, uchannels_dict, \\
    ...     pupitre_dict, upupitre_dict, threshold_dict = get_legacy_config()
    """
    colors = ColorConfig()
    channels = ChannelMapping()
    voltage_channels = VoltageChannelMapping()
    thresholds = ThresholdConfig.default()

    # Build pupitre dicts for all sites
    pupitre_dict = {
        site_name: config.to_pupitre_dict()
        for site_name, config in SITE_CONFIGS.items()
    }
    upupitre_dict = {
        site_name: config.to_upupitre_dict()
        for site_name, config in SITE_CONFIGS.items()
    }

    return (
        colors.to_dict(),
        channels.to_dict(),
        voltage_channels.to_dict(),
        pupitre_dict,
        upupitre_dict,
        thresholds.to_dict(),
    )


# Alias for complete backward compatibility
def setup() -> tuple:
    """
    Original setup() function for backward compatibility.

    .. deprecated::
        Use AnalysisConfig.for_site() instead.

    Returns
    -------
    tuple
        Configuration dictionaries in legacy format
    """
    return get_legacy_config()
