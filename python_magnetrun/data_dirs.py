"""data_dirs — Canonical data directory constants for python_magnetrun.

Each constant is resolved at import time from a prioritised chain:

1. Environment variable(s)
2. ``~/.config/python_magnetrun/data_dirs.json``
3. Hard-coded default path

Env var resolution order
------------------------
MAGNETRUN_PUPITRE_DATA_DIR → PUPITRE_DATADIR → json["pupitre"]
MAGNETRUN_PIGBROTHER_DATA_DIR → PIGBROTHER_DATADIR → json["pigbrother"]
MAGNETRUN_HYBRID_DATA_DIR → HYBRID_DATADIR → json["hybrid"]
MAGNETRUN_BPROFILE_DATA_DIR → json["bprofile"]
MAGNETRUN_FEELPP_DATA_DIR → json["feelpp"]
MAGNETRUN_MAGNETTOOLS_DATA_DIR → json["magnettools"]
MAGNETRUN_PIGBROTHER_RUNLOG_DIR → json["pigbrother_runlog"]
MAGNETRUN_PUPITRE_RUNLOG_DIR → json["pupitre_runlog"]
MAGNETRUN_HYBRID_RUNLOG_DIR → json["hybrid_runlog"]

data_dirs.json example
-----------------------
Create ``~/.config/python_magnetrun/data_dirs.json``::

    {
        "pupitre":    "/mnt/nas/Pupitre",
        "pigbrother": "/mnt/nas/PigBrother",
        "hybrid":     "/mnt/nas/FEPC"
    }

Usage
-----
::

    from python_magnetrun.data_dirs import (
        PUPITRE_DATA_DIR,
        PIGBROTHER_DATA_DIR,
        HYBRID_DATA_DIR,
        BPROFILE_DATA_DIR,
        FEELPP_DATA_DIR,
        MAGNETTOOLS_DATA_DIR,
    )
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# User config file
# ---------------------------------------------------------------------------

USER_CONFIG_DIR: Path = Path.home() / ".config" / "python_magnetrun"
"""User configuration directory (``~/.config/python_magnetrun/``)."""

_DATA_DIRS_JSON_PATH: Path = USER_CONFIG_DIR / "data_dirs.json"
"""Default path to the user data-directory config file."""


def load_data_dirs_json(path: Path | None = None) -> dict[str, str]:
    """Load data-directory overrides from a JSON config file.

    Parameters
    ----------
    path : Path, optional
        Path to the JSON file.  Defaults to
        ``~/.config/python_magnetrun/data_dirs.json``.

    Returns
    -------
    dict[str, str]
        Mapping of key name to directory path string.
        Returns an empty dict when the file does not exist or is unreadable.

    Notes
    -----
    Recognised keys: ``pupitre``, ``pigbrother``, ``hybrid``, ``bprofile``,
    ``feelpp``, ``magnettools``, ``pigbrother_runlog``, ``pupitre_runlog``,
    ``hybrid_runlog``.  Unknown keys are silently ignored.  Empty-string
    values are treated as absent.
    """
    p = path if path is not None else _DATA_DIRS_JSON_PATH
    if not p.exists():
        return {}
    try:
        with p.open() as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            logger.warning("data_dirs.json: expected a JSON object, got %s", type(data).__name__)
            return {}
        return {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, str) and v}
    except json.JSONDecodeError as exc:
        logger.warning("data_dirs.json: JSON parse error: %s", exc)
        return {}
    except OSError as exc:
        logger.warning("data_dirs.json: cannot read: %s", exc)
        return {}


def _first_env(*names: str, default: str = "") -> str:
    """Return the value of the first set (non-empty) env var in *names*.

    Falls back to *default* when none of the named variables are set.
    """
    for name in names:
        val = os.environ.get(name, "")
        if val:
            return val
    return default


# Load the JSON config once at import time.
_JSON_DIRS: dict[str, str] = load_data_dirs_json()


def _resolve(*env_names: str, json_key: str = "", default: str = "") -> str:
    """Resolve a directory: env var > JSON config > hard-coded default.

    Parameters
    ----------
    *env_names : str
        Environment variable names to check, in priority order.
    json_key : str
        Key to look up in ``_JSON_DIRS`` (loaded from ``data_dirs.json``).
    default : str
        Hard-coded fallback when neither env vars nor JSON provide a value.

    Returns
    -------
    str
        Resolved directory path, or *default* if nothing else matched.
    """
    for name in env_names:
        val = os.environ.get(name, "")
        if val:
            return val
    if json_key:
        val = _JSON_DIRS.get(json_key, "")
        if val:
            return val
    return default


# ---------------------------------------------------------------------------
# Pupitre (.txt) data files
# ---------------------------------------------------------------------------

PUPITRE_DATA_DIR: str = _resolve(
    "MAGNETRUN_PUPITRE_DATA_DIR",
    "PUPITRE_DATADIR",
    json_key="pupitre",
    default="/mnt/LNCMIG-Data/records/srv-data-install",
)
"""Root directory for pupitre ``.txt`` data files.

Override priority: ``MAGNETRUN_PUPITRE_DATA_DIR`` >
``PUPITRE_DATADIR`` > ``data_dirs.json["pupitre"]`` > hard-coded default.
"""

# ---------------------------------------------------------------------------
# Pigbrother (.tdms) data files
# ---------------------------------------------------------------------------

PIGBROTHER_DATA_DIR: str = _resolve(
    "MAGNETRUN_PIGBROTHER_DATA_DIR",
    "PIGBROTHER_DATADIR",
    json_key="pigbrother",
    default="/mnt/LNCMIG-Data/records/pbsurv",
)
"""Root directory for pigbrother ``.tdms`` data files.

Override priority: ``MAGNETRUN_PIGBROTHER_DATA_DIR`` >
``PIGBROTHER_DATADIR`` > ``data_dirs.json["pigbrother"]`` > hard-coded default.
"""

# ---------------------------------------------------------------------------
# Hybrid (kHz / RMS / trigger) data files
# ---------------------------------------------------------------------------

HYBRID_DATA_DIR: str = _resolve(
    "MAGNETRUN_HYBRID_DATA_DIR",
    "HYBRID_DATADIR",
    json_key="hybrid",
    default="/mnt/LNCMIG-Data/records/CEA",
)
"""Root directory for hybrid (kHz/RMS/trigger) data files.

Override priority: ``MAGNETRUN_HYBRID_DATA_DIR`` >
``HYBRID_DATADIR`` > ``data_dirs.json["hybrid"]`` > hard-coded default.
"""

# ---------------------------------------------------------------------------
# BProfile data files
# ---------------------------------------------------------------------------

BPROFILE_DATA_DIR: str = _resolve(
    "MAGNETRUN_BPROFILE_DATA_DIR",
    json_key="bprofile",
    default="",
)
"""Root directory for BProfile data files.

Override via ``MAGNETRUN_BPROFILE_DATA_DIR`` or ``data_dirs.json["bprofile"]``.
No hard-coded default.
"""

# ---------------------------------------------------------------------------
# Feel++ simulation data files
# ---------------------------------------------------------------------------

FEELPP_DATA_DIR: str = _resolve(
    "MAGNETRUN_FEELPP_DATA_DIR",
    json_key="feelpp",
    default="",
)
"""Root directory for Feel++ simulation data files.

Override via ``MAGNETRUN_FEELPP_DATA_DIR`` or ``data_dirs.json["feelpp"]``.
No hard-coded default.
"""

# ---------------------------------------------------------------------------
# MagnetTools data files
# ---------------------------------------------------------------------------

MAGNETTOOLS_DATA_DIR: str = _resolve(
    "MAGNETRUN_MAGNETTOOLS_DATA_DIR",
    json_key="magnettools",
    default="",
)
"""Root directory for MagnetTools data files.

Override via ``MAGNETRUN_MAGNETTOOLS_DATA_DIR`` or ``data_dirs.json["magnettools"]``.
No hard-coded default.
"""

# ---------------------------------------------------------------------------
# Run-log (ACQ_ENET) data files — per data source
# ---------------------------------------------------------------------------

PIGBROTHER_RUNLOG_DIR: str = _resolve(
    "MAGNETRUN_PIGBROTHER_RUNLOG_DIR",
    json_key="pigbrother_runlog",
    default="",
)
"""Root directory for pigbrother run-log files (co-located with .tdms files).

Override via ``MAGNETRUN_PIGBROTHER_RUNLOG_DIR`` or
``data_dirs.json["pigbrother_runlog"]``.  No hard-coded default.
"""

PUPITRE_RUNLOG_DIR: str = _resolve(
    "MAGNETRUN_PUPITRE_RUNLOG_DIR",
    json_key="pupitre_runlog",
    default="",
)
"""Root directory for pupitre run-log files.

Override via ``MAGNETRUN_PUPITRE_RUNLOG_DIR`` or
``data_dirs.json["pupitre_runlog"]``.  No hard-coded default.
"""

HYBRID_RUNLOG_DIR: str = _resolve(
    "MAGNETRUN_HYBRID_RUNLOG_DIR",
    json_key="hybrid_runlog",
    default="",
)
"""Root directory for hybrid run-log files.

Override via ``MAGNETRUN_HYBRID_RUNLOG_DIR`` or
``data_dirs.json["hybrid_runlog"]``.  No hard-coded default.
"""
