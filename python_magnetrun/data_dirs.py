"""data_dirs — Canonical data directory constants for python_magnetrun.

Each constant is resolved at import time from a prioritised chain of
environment variables, falling back to a hard-coded default path.

Env var resolution order
------------------------
MAGNETRUN_PUPITRE_DATA_DIR → PUPITRE_DATADIR → MAGNETRUN_DATA_DIR
MAGNETRUN_PIGBROTHER_DATA_DIR → PIGBROTHER_DATADIR → PIGBROTHER
MAGNETRUN_HYBRID_DATA_DIR → HYBRID_DATADIR
MAGNETRUN_BPROFILE_DATA_DIR
MAGNETRUN_FEELPP_DATA_DIR
MAGNETRUN_MAGNETTOOLS_DATA_DIR

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

import os


def _first_env(*names: str, default: str = "") -> str:
    """Return the value of the first set (non-empty) env var in *names*.

    Falls back to *default* when none of the named variables are set.
    """
    for name in names:
        val = os.environ.get(name, "")
        if val:
            return val
    return default


# ---------------------------------------------------------------------------
# Pupitre (.txt) data files
# ---------------------------------------------------------------------------

PUPITRE_DATA_DIR: str = _first_env(
    "MAGNETRUN_PUPITRE_DATA_DIR",
    "PUPITRE_DATADIR",
    default="/mnt/LNCMIG-Data/records/srv-data-install",
)
"""Root directory for pupitre ``.txt`` data files.

Override priority: ``MAGNETRUN_PUPITRE_DATA_DIR`` >
``PUPITRE_DATADIR`` > hard-coded default.
"""

# ---------------------------------------------------------------------------
# Pigbrother (.tdms) data files
# ---------------------------------------------------------------------------

PIGBROTHER_DATA_DIR: str = _first_env(
    "MAGNETRUN_PIGBROTHER_DATA_DIR",
    "PIGBROTHER_DATADIR",
    default="/mnt/LNCMIG-Data/records/pbsurv",
)
"""Root directory for pigbrother ``.tdms`` data files.

Override priority: ``MAGNETRUN_PIGBROTHER_DATA_DIR`` >
``PIGBROTHER_DATADIR`` > hard-coded default.
"""

# ---------------------------------------------------------------------------
# Hybrid (kHz / RMS / trigger) data files
# ---------------------------------------------------------------------------

HYBRID_DATA_DIR: str = _first_env(
    "MAGNETRUN_HYBRID_DATA_DIR",
    "HYBRID_DATADIR",
    default="/mnt/LNCMIG-Data/records/CEA",
)
"""Root directory for hybrid (kHz/RMS/trigger) data files.

Override priority: ``MAGNETRUN_HYBRID_DATA_DIR`` >
``HYBRID_DATADIR`` > hard-coded default.
"""

# ---------------------------------------------------------------------------
# BProfile data files
# ---------------------------------------------------------------------------

BPROFILE_DATA_DIR: str = _first_env(
    "MAGNETRUN_BPROFILE_DATA_DIR",
    default="",
)
"""Root directory for BProfile data files.

Override via ``MAGNETRUN_BPROFILE_DATA_DIR``.  No hard-coded default.
"""

# ---------------------------------------------------------------------------
# Feel++ simulation data files
# ---------------------------------------------------------------------------

FEELPP_DATA_DIR: str = _first_env(
    "MAGNETRUN_FEELPP_DATA_DIR",
    default="",
)
"""Root directory for Feel++ simulation data files.

Override via ``MAGNETRUN_FEELPP_DATA_DIR``.  No hard-coded default.
"""

# ---------------------------------------------------------------------------
# MagnetTools data files
# ---------------------------------------------------------------------------

MAGNETTOOLS_DATA_DIR: str = _first_env(
    "MAGNETRUN_MAGNETTOOLS_DATA_DIR",
    default="",
)
"""Root directory for MagnetTools data files.

Override via ``MAGNETRUN_MAGNETTOOLS_DATA_DIR``.  No hard-coded default.
"""

# ---------------------------------------------------------------------------
# Run-log (ACQ_ENET) data files — per data source
# ---------------------------------------------------------------------------

PIGBROTHER_RUNLOG_DIR: str = _first_env(
    "MAGNETRUN_PIGBROTHER_RUNLOG_DIR",
    default="",
)
"""Root directory for pigbrother run-log files (co-located with .tdms files).

Override via ``MAGNETRUN_PIGBROTHER_RUNLOG_DIR``.  No hard-coded default.
"""

PUPITRE_RUNLOG_DIR: str = _first_env(
    "MAGNETRUN_PUPITRE_RUNLOG_DIR",
    default="",
)
"""Root directory for pupitre run-log files.

Override via ``MAGNETRUN_PUPITRE_RUNLOG_DIR``.  No hard-coded default.
"""

HYBRID_RUNLOG_DIR: str = _first_env(
    "MAGNETRUN_HYBRID_RUNLOG_DIR",
    default="",
)
"""Root directory for hybrid run-log files.

Override via ``MAGNETRUN_HYBRID_RUNLOG_DIR``.  No hard-coded default.
"""
