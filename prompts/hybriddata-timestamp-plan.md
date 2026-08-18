# Plan: HybridData timestamp support (`start_timestamp`, `end_timestamp`, `addTime()`)

## Context

`MagnetDataBase` defines the UTC timestamp convention:
- `start_timestamp` / `end_timestamp` — naive UTC `datetime`
- `t` column — elapsed seconds from first sample
- `timestamp` column — naive UTC `pd.Timestamp`
- `addTime()` is *eager*: computes `t` + `timestamp` for all data at once

`HybridData` **inherits from `MagnetDataBase`** (added in Stream 3.6 R4). The base-class
`__init__` already sets `self.start_timestamp = None` and `self.end_timestamp = None`,
so no attribute injection is needed — only the methods that populate them are missing.

`runetl.py:157` calls `data.addTime()` and `runetl.py:158` calls `data.getDuration()`
on whatever data object is passed in (including `HybridData`). Both currently fall
through to the `MagnetDataBase` no-op defaults. These callers will work correctly once
this plan is implemented.

`processing/cli.py:176` calls `mdata.getStartDate()` — currently harmless (result is
stored but never consumed), but must return the correct 4-string-tuple format for
interface consistency.

---

## Convention (same as MagnetDataBase)

| Attribute / column             | Type                    | Value                          |
|-------------------------------|-------------------------|--------------------------------|
| `start_timestamp`             | naive `datetime`        | UTC                            |
| `end_timestamp`               | naive `datetime`        | UTC                            |
| `t` column (in read arrays)   | float                   | elapsed seconds from first sample |
| `timestamp` column (future)   | naive `pd.Timestamp`    | UTC                            |

---

## Files to change

### 1. `python_magnetrun/hybrid/hybrid_data.py`

#### 1a. `__init__` — call `_infer_timestamps()` after discovery

`start_timestamp` and `end_timestamp` are already set to `None` by
`MagnetDataBase.__init__()` (called via `super().__init__(...)`). No new attribute
assignments are needed. Only add one line at the end of `__init__`, after
`self._discover_data()`:

```python
self._infer_timestamps()
```

#### 1b. Add `_infer_timestamps(time_zone="Europe/Paris")`

Derives `start_timestamp` / `end_timestamp` from the file structure already
discovered in `self._info`. Converts local → naive UTC via pytz.

**Source priority** (most precise first):

| Priority | Source | Format | Time zone |
|---|---|---|---|
| 1 | RMS filenames | `{sys}_YYYY-MM-DD_HHMM—YYYY-MM-DD_HHMM.rms` | local → UTC |
| 2 | kHz bin filenames via `compute_hour_t0` | Unix UTC float; `XX` = hour in `XXHOST_*_LIST_{slot}.bin` | already UTC |
| 3 | Trigger directory names | `TRIGGER__YYYY-MM-DD__HH-MM` | local → UTC |

**Critical invariant**: each source must store its timestamps as naive UTC *before*
appending to the shared accumulation lists. The kHz path uses `compute_hour_t0` which
returns Unix UTC floats — these must **not** go through `_local_to_utc`. Mixing raw
local timestamps and UTC-naive timestamps in the same list and then calling
`_local_to_utc(min(starts))` would double-convert the kHz values. Use separate
conversion per source:

```python
def _infer_timestamps(self, time_zone: str = "Europe/Paris") -> None:
    """Infer start_timestamp / end_timestamp from discovered files (naive UTC)."""
    import re
    import pytz

    tz = pytz.timezone(time_zone)

    def _local_to_utc(local_ts: pd.Timestamp) -> datetime:
        return (
            local_ts
            .tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
            .tz_convert(pytz.utc)
            .to_pydatetime()
            .replace(tzinfo=None)
        )

    utc_starts: list[datetime] = []
    utc_ends: list[datetime] = []

    # --- 1. RMS filenames (local time, both start and end encoded) ---
    # Format: {system}_YYYY-MM-DD_HHMM—YYYY-MM-DD_HHMM.rms
    # Separator may be em-dash (—, U+2014) or hyphen-minus (-).
    pattern = r"(\d{4}-\d{2}-\d{2})_(\d{2})(\d{2})[—\-](\d{4}-\d{2}-\d{2})_(\d{2})(\d{2})"
    for files in self._info.rms_files.values():
        for f in files:
            m = re.search(pattern, f.stem)
            if m:
                d1, h1, mn1, d2, h2, mn2 = m.groups()
                utc_starts.append(_local_to_utc(pd.Timestamp(f"{d1} {h1}:{mn1}")))
                utc_ends.append(_local_to_utc(pd.Timestamp(f"{d2} {h2}:{mn2}")))

    # --- 2. kHz bin filenames (already UTC — do NOT apply _local_to_utc) ---
    if not utc_starts and compute_hour_t0 is not None:
        for system, files in self._info.khz_files.items():
            if system.endswith("_cfg") or not files:
                continue
            bin_files = [f for f in files if f.suffix == ".bin"]
            if not bin_files:
                continue
            try:
                t0_start = compute_hour_t0(str(bin_files[0]), self.date_str)
                t0_end = compute_hour_t0(str(bin_files[-1]), self.date_str)
                utc_starts.append(
                    pd.Timestamp(t0_start, unit="s", tz="UTC").tz_localize(None).to_pydatetime()
                )
                utc_ends.append(
                    pd.Timestamp(t0_end + 3600, unit="s", tz="UTC").tz_localize(None).to_pydatetime()
                )
            except Exception as exc:
                logger.debug(f"_infer_timestamps: kHz fallback failed for {system}: {exc}")

    # --- 3. Trigger directories (last resort: start only, local time) ---
    if not utc_starts:
        for dirs in self._info.trigger_dirs.values():
            for d in dirs:
                parts = d.name.split("__")   # TRIGGER__YYYY-MM-DD__HH-MM
                if len(parts) >= 3:
                    try:
                        utc_starts.append(
                            _local_to_utc(
                                pd.Timestamp(f"{parts[1]} {parts[2].replace('-', ':')}")
                            )
                        )
                    except Exception:
                        pass

    if not utc_starts:
        logger.debug(f"_infer_timestamps: no timestamps found for {self.date_str}")
        return

    self.start_timestamp = min(utc_starts)
    self.end_timestamp = max(utc_ends) if utc_ends else None
```

#### 1c. Add `addTime(time_zone="Europe/Paris")`

Unlike `PandasMagnetData`/`TdmsMagnetData`, `HybridData` doesn't hold pre-loaded
DataFrames — data is read on demand. `addTime()` is therefore **metadata-only**: it
ensures `start_timestamp`/`end_timestamp` are set.

The time arrays returned by `read_khz_variable()` and `read_rms_variable()` remain
relative (elapsed seconds from first sample). Adding absolute `timestamp` arrays is
a follow-up item (see "Out of scope").

```python
def addTime(self, time_zone: str = "Europe/Paris") -> int:
    """Compute and store start_timestamp / end_timestamp (naive UTC).

    Must be called before getDuration(), getStartDate(), or any
    timestamp-based comparison with other data sources.

    Unlike PandasMagnetData/TdmsMagnetData, HybridData does not hold
    pre-loaded DataFrames, so addTime() only sets the metadata timestamps.
    Time arrays from read_khz_variable() / read_rms_variable() remain
    relative (elapsed seconds from first sample).

    Parameters
    ----------
    time_zone : str
        IANA timezone of the source date/time data.

    Returns
    -------
    int
        ``0`` on success.
    """
    self._infer_timestamps(time_zone=time_zone)
    return 0
```

#### 1d. Override `getStartDate()` and `getDuration()`

**`getStartDate` must return a 4-string-tuple**, matching the format used by
`PandasMagnetData` and expected by `MagnetRun.from_txt:258`:

```python
start_date, start_time, end_date, end_time = res
```

The strings use `"%Y.%m.%d"` and `"%H:%M:%S"` format (dot-separated date, colon-
separated time), consistent with `PandasMagnetData.getStartDate()`.

```python
def getStartDate(self, group: str | None = None) -> tuple:  # noqa: N802
    """Return start/end date and time strings derived from file metadata.

    Parameters
    ----------
    group : str, optional
        Unused; accepted for interface compatibility.

    Returns
    -------
    tuple
        ``(start_date, start_time, end_date, end_time)`` strings in
        ``"%Y.%m.%d"`` / ``"%H:%M:%S"`` format, or empty tuple when
        timestamps are unavailable.
    """
    if self.start_timestamp is None:
        self._infer_timestamps()
    if self.start_timestamp is None:
        return ()
    fmt_date = "%Y.%m.%d"
    fmt_time = "%H:%M:%S"
    return (
        self.start_timestamp.strftime(fmt_date),
        self.start_timestamp.strftime(fmt_time),
        self.end_timestamp.strftime(fmt_date) if self.end_timestamp else "",
        self.end_timestamp.strftime(fmt_time) if self.end_timestamp else "",
    )

def getDuration(self, group: str | None = None) -> float:  # noqa: N802
    """Return the duration of the dataset in seconds.

    Parameters
    ----------
    group : str, optional
        Unused; accepted for interface compatibility.

    Returns
    -------
    float
        ``(end_timestamp - start_timestamp).total_seconds()`` [s], or
        ``0.0`` when either timestamp is unavailable.
    """
    if self.start_timestamp is None or self.end_timestamp is None:
        return 0.0
    return (self.end_timestamp - self.start_timestamp).total_seconds()
```

---

### 2. `python_magnetrun/hybrid/hybrid_run.py`

#### 2a. Fix `get_time_range()` to mirror `MagnetRun`

`MagnetRun.get_time_range()` delegates to `self.MagnetData.get_time_range()`, which
returns `(start_timestamp, end_timestamp)` from the already-set base-class attributes.

`HybridRun.get_time_range()` currently calls the module-level function
`_khz_first_last_utc(self.HybridData)` directly. After task 4.1, `HybridData.__init__`
populates `start_timestamp`/`end_timestamp` via `_infer_timestamps()` (which absorbs
the kHz-hour logic as its priority-2 source). `HybridRun.get_time_range()` should then
delegate to `self.HybridData.get_time_range()` — identical to the `MagnetRun` pattern:

```python
def get_time_range(self) -> tuple[datetime, datetime]:
    """Get time range of available data.

    Delegates to :meth:`~python_magnetrun.hybrid.hybrid_data.HybridData.get_time_range`,
    mirroring the pattern used by :meth:`~python_magnetrun.MagnetRun.MagnetRun.get_time_range`.

    Returns
    -------
    tuple[datetime, datetime]
        ``(start_timestamp, end_timestamp)`` as naive UTC datetimes.

    Raises
    ------
    RuntimeError
        If no :class:`~python_magnetrun.hybrid.hybrid_data.HybridData` is associated.
    NotImplementedError
        If no timestamps could be inferred from the data files.
    """
    if self.HybridData is None:
        raise RuntimeError("No HybridData associated")
    return self.HybridData.get_time_range()
```

The `_khz_first_last_utc` module-level function in `hybrid_data.py` is now superseded
by `_infer_timestamps` (which uses it internally as its priority-2 fallback). It should
be kept until `TestHybridRunGetTimeRange` tests are updated to use the new path; removal
is a follow-up.

---

### 3. `tests/test_hybrid_api.py`

Add `TestHybridTimestamps` class. Fixtures use `tmp_path` with a synthetic `.rms` file
named e.g. `SUPRA_2025-01-06_1000—2025-01-06_1200.rms` placed under
`tmp_path/rms/SUPRA/`. No real data files required.

| Test | What it verifies |
|---|---|
| `test_no_files_returns_none` | Empty `_info` leaves both timestamps as `None` without raising |
| `test_start_timestamp_from_rms_filename` | `_infer_timestamps()` picks up start from RMS filename; result is naive UTC `datetime` |
| `test_end_timestamp_from_rms_filename` | `end_timestamp > start_timestamp` |
| `test_utc_conversion_winter` | Known local `2025-01-06 10:00` CET (+1 h) → UTC `09:00` |
| `test_addTime_sets_both` | `addTime()` sets both attributes; returns `0` |
| `test_getDuration_positive` | `getDuration()` > 0 |
| `test_getStartDate_format` | Returns 4-tuple; all elements are non-empty strings; dates are dot-separated |

---

## Key implementation detail: `compute_hour_t0` return type

`compute_hour_t0(filepath, date_str)` returns a **Unix UTC float** (seconds since
1970-01-01 UTC). Convert to naive UTC datetime as:

```python
pd.Timestamp(t0_float, unit="s", tz="UTC").tz_localize(None).to_pydatetime()
```

Do **not** pass this through `_local_to_utc` — it is already UTC.

---

## What is NOT changed in this plan

- Absolute `timestamp` arrays from `read_khz_variable()` / `read_rms_variable()` —
  these still return relative elapsed seconds. A follow-up can add an optional
  `return_timestamps=True` parameter that returns absolute naive UTC timestamps using
  `start_timestamp + t_elapsed`.
- `extractTimeData()` for `HybridData` — timestamp-based time-range filtering of
  kHz/RMS data; blocked on having absolute timestamps first.
- `DataProvider` / `DataLoader` protocol unification — tracked separately in
  `prompts/cross-domain-comparison.prompt.md` Phase A0.
- `_khz_first_last_utc` removal — kept until `TestHybridRunGetTimeRange` is updated;
  tracked as a follow-up cleanup.

---

## Relation to other plans

- **Prerequisite**: none — this plan is self-contained.
- **Enables**: `ComparisonSession` / `analysis/loaders.py` integration with
  `HybridData` (needs `start_timestamp` to align time axes across data sources).
- **Parallel with**: `prompts/timestamp-utc-refactoring.plan.md` (which covers
  `PandasMagnetData` and `TdmsMagnetData` only; explicitly defers `HybridData`).

---

## File changes summary

| File | Change |
|---|---|
| `python_magnetrun/hybrid/hybrid_data.py` | Call `_infer_timestamps()` at end of `__init__`; add `_infer_timestamps()`, `addTime()`, `getStartDate()` (4-string-tuple), `getDuration()` |
| `python_magnetrun/hybrid/hybrid_run.py` | Fix `get_time_range()` to delegate to `self.HybridData.get_time_range()` |
| `tests/test_hybrid_api.py` | Add `TestHybridTimestamps` class (7 test cases) |
