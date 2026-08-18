# runlogs — Run-log readers for MagnetRun data sources

This sub-package provides readers for the operator run-log files produced by
the LNCMI acquisition systems.  It replaces the former `tdms/log_parser.py`
and adds a stub for the pupitre Cirrus logs.

## Sub-modules

| Module | Source system | Log file(s) |
|---|---|---|
| `pigbrother.py` | LabVIEW/DAQmx (pigbrother) | `LOG_ACQ_ENET.txt` |
| `pupitre.py` | Cirrus PSU control (pupitre) | `cirrus/A[1-4]/YYYY-MM-DD_cirrus_out.log` |

---

## pigbrother — `LOG_ACQ_ENET.txt`

### File location

A single file co-located with the pigbrother `.tdms` data files:

```
/mnt/LNCMIG-Data/records/pbsurv/LOG_ACQ_ENET.txt
```

Override the root via the env var `MAGNETRUN_PIGBROTHER_RUNLOG_DIR`.

### Log file format

Records follow the pattern:

```
<day_fr> <date> <time>: <message>
```

Example:

```
jeu. 19-09-2019 16:14:33: Test de présence des boitiers ENET
Test A1 : OK
Test M3 : KO
```

- Day abbreviations are French: `lun. mar. mer. jeu. ven. sam. dim.`
- Messages are a mix of French and English
- Errors and ENET tests span multiple lines

### Parsed event types

- **ENET box presence tests** — connectivity checks (OK/KO per device)
- **Acquisition events** — start/stop on each magnet group
- **File creation events** — Archive, Overview, Stats, Spike, Default, ManuelTrig
- **DAQmx/TDMS errors** — error codes with descriptions
- **Fault detection** — SpikeAimant, DefautNums, Courants50Hz

### Usage

```python
from python_magnetrun.runlogs.pigbrother import LogParser

parser = LogParser("/mnt/LNCMIG-Data/records/pbsurv/LOG_ACQ_ENET.txt").parse()

summary = parser.summary()
print(f"Total entries : {summary['total_entries']}")
print(f"Files with errors : {summary['files_with_errors']}")

for acq in parser.acquisitions:
    print(f"{acq.action} on {acq.magnet} at {acq.timestamp}")

files_with_errors = parser.get_files_with_errors_dict()
defaut_files      = parser.get_defaut_files_dict()
```

Parse from a string:

```python
parser = LogParser("/dev/null")
parser.parse_string(log_content)
```

### Output dictionaries

**`files_with_errors`** — Archive/Overview files that had DAQ errors:

```json
{
  "M9_Archive_191002-1445.tdms": {
    "filepath": "F:\\...\\M9_Archive_191002-1445.tdms",
    "file_type": "Archive",
    "magnet": "M9",
    "file_timestamp": "2019-10-02T14:46:22",
    "errors": [
      {
        "error_code": -200279,
        "error_short": "Buffer overflow",
        "error_timestamp": "2019-10-03T09:02:49"
      }
    ]
  }
}
```

**`defaut_files`** — Fault TDMS files with detection details:

```json
{
  "M9_Spike_191002-153000.tdms": {
    "type": "SpikeAimant",
    "description": "Spike de courant anormal détecté...",
    "timestamp": "2019-10-02T15:30:00",
    "details": { "sensors": ["Interne1"] }
  }
}
```

### `LogParser` API

| Method / attribute | Description |
|---|---|
| `parse()` | Parse the file; returns `self` for chaining |
| `parse_string(content)` | Parse from a string |
| `summary()` | Statistical summary dict |
| `get_defaut_files_dict()` | Fault TDMS files |
| `get_files_with_errors_dict()` | Archive/Overview files with errors |
| `entries` | All `LogEntry` objects |
| `enet_tests` | `ENETTestResult` list |
| `acquisitions` | `AcquisitionEvent` list |
| `files_created` | `FileCreatedEvent` list |
| `errors` | `ErrorEvent` list |
| `defauts` | `DefautEvent` list |

The parser automatically tries UTF-8, Latin-1, CP1252, and ISO-8859-1 before
falling back to UTF-8 with replacement, so broken French characters are
handled gracefully.

For the detailed ACQ_ENET message format see
[`Documentation_LogAcqNet.md`](../../docs/Documentation_LogAcqNet.md).

---

## pupitre — Cirrus PSU run-logs

### File location

One log file per power-supply unit (A1–A4) per calendar day:

```
<runlog_root>/cirrus/A1/2026-04-16_cirrus_out.log
<runlog_root>/cirrus/A2/2026-04-16_cirrus_out.log
...
```

Override the root via the env var `MAGNETRUN_PUPITRE_RUNLOG_DIR`.

### Log file format

Each line has three whitespace-separated fields:

```
<Type>  <RelativeTime>  <Message>
```

`RelativeTime` is seconds since an unspecified origin.  Absolute timestamps
are anchored from embedded messages such as:

```
Info  1234.5  Arret redresseur date  2026-4-15 16:0:9.111719
```

A processed CSV export (`Logs Cirrus.csv`) with pre-computed absolute
timestamps is also available from the Cirrus application.

### Usage

```python
from python_magnetrun.runlogs.pupitre import CirrusRunlogLoader

loader = CirrusRunlogLoader("/mnt/LNCMIG-Data/records/srv-data-install/M9")
files = loader.find_files("2026-04-15", "2026-04-16")
# entries = loader.load(files)   # not yet implemented
```

Or as a convenience function:

```python
from python_magnetrun.runlogs.pupitre import discover_pupitre_runlogs

paths = discover_pupitre_runlogs(
    "/mnt/LNCMIG-Data/records/srv-data-install/M9",
    start_date="2026-04-15",
    end_date="2026-04-16",
)
```

> **Note**: `CirrusRunlogLoader.load()` is not yet implemented — it returns
> an empty list with a warning.  The discovery helper (`find_files`) is
> fully functional.
