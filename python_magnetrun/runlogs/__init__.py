"""runlogs — Run-log readers for all MagnetRun data sources.

Sub-modules
-----------
pigbrother
    Parser for the LabVIEW/DAQmx acquisition system log
    (``LOG_ACQ_ENET.txt`` co-located with pigbrother .tdms files).

pupitre
    Reader/stub for Cirrus power-supply run-log files
    (``cirrus/A[1-4]/YYYY-MM-DD_cirrus_out.log``).
"""

from .pigbrother import LogParser
from .pupitre import CirrusRunlogLoader, discover_pupitre_runlogs

__all__ = [
    "LogParser",
    "CirrusRunlogLoader",
    "discover_pupitre_runlogs",
]
