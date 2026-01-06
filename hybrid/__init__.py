"""
Hybrid Data Module

Unified interface for hybrid magnet data from FEPC acquisition systems.

Supports three data types:
- kHz: High-frequency (1 kHz) data from FEPC analog and digital cards
- RMS: Root Mean Square data at lower frequency (typically 10 Hz)
- Trigger: Event-triggered data

Example usage:
    from hybrid import HybridData

    data = HybridData("/data/hybrid", "2025-01-06")
    data.print_summary()

    # Read kHz variable
    values, time = data.read_khz_variable("FEPC_LNCMI", "ALIM1_J1")
"""


# Lazy imports to avoid RuntimeWarning when running as `python -m hybrid.hybrid_data`
def __getattr__(name):
    if name in ("HybridData", "HybridDataInfo", "list_available_dates"):
        from .hybrid_data import HybridData, HybridDataInfo, list_available_dates

        globals()["HybridData"] = HybridData
        globals()["HybridDataInfo"] = HybridDataInfo
        globals()["list_available_dates"] = list_available_dates
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "HybridData",
    "HybridDataInfo",
    "list_available_dates",
]

# Get version from parent package
try:
    from importlib.metadata import version, PackageNotFoundError

    __version__ = version("python_magnetrun")
except PackageNotFoundError:
    __version__ = "0.0.0"
