# Plan: HybridData timestamp support (`start_timestamp`, `end_timestamp`, `addTime()`)

## Context

`MagnetDataBase` defines the UTC timestamp convention:
- `start_timestamp` / `end_timestamp` — naive UTC `datetime`
- `t` column — elapsed seconds from first sample
- `timestamp` column — naive UTC `pd.Timestamp`
- `addTime()` is *eager*: computes `t` + `timestamp` for all data at once

`HybridData` does **not** inherit from `MagnetDataBase` (it's a standalone class) and
currently has none of these attributes. This plan brings it to parity, following the
same convention established in `prompts/timestamp-utc-refactoring.plan.md`.

`runetl.py:115` already calls `data.addTime()` on whatever data object is passed in;
`processing/cli.py:171` calls `mdata.getStartDate()`. Both currently fall through to
the `MagnetDataBase` no-op defaults. These callers will work correctly once this plan
is implemented.

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

#### 1a. `__init__` — add attributes

After `self.units` (line ~141), add:

```python
self.start_timestamp: datetime | None = None   # naive UTC
self.end_timestamp: datetime | None = None     # naive UTC
```

Then at the end of `__init__`, after `self._discover_data()`, call:

```python
self._infer_timestamps()
```

#### 1b. Add `_infer_timestamps(time_zone="Europe/Paris")`

Derives `start_timestamp` / `end_timestamp` from the file structure already
discovered in `self._info`. Converts local → naive UTC via pytz.

**Source priority** (most precise first):

| Priority | Source | Format |
|---|---|---|
| 1 | RMS filenames | `{sys}_YYYY-MM-DD_HHMM—YYYY-MM-DD_HHMM.rms` (local time) |
| 2 | kHz bin filenames via `compute_hour_t0` | returns Unix UTC float; XX = hour encoded in `XXHOST_*_LIST_{slot}.bin` |
| 3 | Trigger directory names | `TRIGGER__YYYY-MM-DD__HH-MM` (local time) |

```python
def _infer_timestamps(self, time_zone: str = "Europe/Paris") -> None:
    """Infer start_timestamp / end_timestamp from discovered files (naive UTC)."""
    import re
    import pytz
    from datetime import datetime as dt

    tz = pytz.timezone(time_zone)

    def _local_to_utc(local_ts: pd.Timestamp) -> datetime:
        return (
            local_ts
            .tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
            .tz_convert(pytz.utc)
            .to_pydatetime()
            .replace(tzinfo=None)
        )

    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []

    # --- 1. RMS filenames (local time, both start and end encoded) ---
    # Format: {system}_YYYY-MM-DD_HHMM—YYYY-MM-DD_HHMM.rms
    # Note: separator may be em-dash (—) or regular dash (-)
    pattern = r"(\d{4}-\d{2}-\d{2})_(\d{2})(\d{2})[—-](\d{4}-\d{2}-\d{2})_(\d{2})(\d{2})"
    for files in self._info.rms_files.values():
        for f in files:
            m = re.search(pattern, f.stem)
            if m:
                d1, h1, mn1, d2, h2, mn2 = m.groups()
                starts.append(pd.Timestamp(f"{d1} {h1}:{mn1}"))
                ends.append(pd.Timestamp(f"{d2} {h2}:{mn2}"))

    # --- 2. kHz bin filenames (fallback: start only, no end) ---
    # compute_hour_t0 returns a Unix UTC float for HH:00:00 local time
    if not starts and compute_hour_t0 is not None:
        for system, files in self._info.khz_files.items():
            if system.endswith("_cfg") or not files:
                continue
            bin_files = [f for f in files if f.suffix == ".bin"]
            if not bin_files:
                continue
            try:
                # First file → start, last file → approximate end
                t0_start = compute_hour_t0(str(bin_files[0]), self.date_str)
                t0_end = compute_hour_t0(str(bin_files[-1]), self.date_str)
                # Convert Unix UTC float → naive UTC datetime
                starts.append(pd.Timestamp(t0_start, unit="s", tz="UTC").tz_localize(None))
                # End is at most 1 hour after the last file's t0
                ends.append(pd.Timestamp(t0_end + 3600, unit="s", tz="UTC").tz_localize(None))
            except Exception as exc:
                logger.debug(f"_infer_timestamps: kHz fallback failed for {system}: {exc}")

    # --- 3. Trigger directories (last resort: start only) ---
    if not starts:
        for dirs in self._info.trigger_dirs.values():
            for d in dirs:
                parts = d.name.split("__")   # TRIGGER__YYYY-MM-DD__HH-MM
                if len(parts) >= 3:
                    try:
                        starts.append(
                            pd.Timestamp(f"{parts[1]} {parts[2].replace('-', ':')}")
                        )
                    except Exception:
                        pass

    if not starts:
        logger.debug(f"_infer_timestamps: no timestamps found for {self.date_str}")
        return

    # For RMS, starts/ends are local time — convert to UTC.
    # For kHz fallback, they are already naive UTC — _local_to_utc would double-convert.
    # Solution: kHz path stores pd.Timestamp objects that are already UTC-naive,
    # marked via a separate list; RMS path uses the conversion above.
    # Simpler approach: let each path store UTC-naive directly (see kHz path above).
    # Only RMS/trigger paths need the local→UTC conversion.

    # NOTE: the kHz fallback path already stores naive UTC directly (see above).
    # The RMS and trigger paths store local timestamps — convert them here.
    # To distinguish, kHz path timestamps have no tz (added as UTC-naive directly),
    # RMS/trigger path timestamps also have no tz but represent local time.
    # Therefore, separate the two paths clearly:

    # Re-implement cleanly with explicit conversion per source:
    self.start_timestamp = _local_to_utc(min(starts)) if starts else None
    self.end_timestamp = _local_to_utc(max(ends)) if ends else None
```

> **Implementation note**: the kHz fallback path computes Unix UTC floats via
> `compute_hour_t0`, so those must NOT go through `_local_to_utc`. Keep the kHz
> fallback in a separate branch with direct UTC conversion, and only merge into
> `starts`/`ends` after conversion, so the final `_local_to_utc(min(starts))` call
> only sees local-time values from RMS/trigger paths.
>
> Clean structure:
> 1. Collect `(utc_start, utc_end)` pairs directly from each source.
> 2. `self.start_timestamp = min(utc_starts)`
> 3. `self.end_timestamp = max(utc_ends)` if any.

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
    """
    self._infer_timestamps(time_zone=time_zone)
    return 0
```

#### 1d. Override `getStartDate()` and `getDuration()`

```python
def getStartDate(self, group: str | None = None) -> tuple:
    if self.start_timestamp is None:
        self._infer_timestamps()
    return (self.start_timestamp,) if self.start_timestamp is not None else ()

def getDuration(self, group: str | None = None) -> float:
    if self.start_timestamp is None or self.end_timestamp is None:
        return 0.0
    return (self.end_timestamp - self.start_timestamp).total_seconds()
```

---

### 2. `tests/test_hybrid_api.py`

Add `TestHybridTimestamps` class. Use either:
- Real test data if available (existing fixtures)
- Synthetic `HybridDataInfo` patched directly onto a `HybridData` instance

| Test | What it verifies |
|---|---|
| `test_start_timestamp_from_rms_filename` | `_infer_timestamps()` picks up start from RMS filename; result is naive UTC |
| `test_end_timestamp_from_rms_filename` | `end_timestamp > start_timestamp` |
| `test_addTime_sets_both` | `addTime()` sets both attributes; returns `0` |
| `test_getDuration_positive` | `getDuration()` returns positive float when both timestamps set |
| `test_no_files_returns_none` | Empty `_info` leaves both timestamps as `None` without raising |
| `test_utc_conversion` | A known local time (e.g. `2025-01-06 10:00` CET = UTC+1) converts to correct UTC |

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
| `python_magnetrun/hybrid/hybrid_data.py` | Add `start_timestamp`, `end_timestamp` to `__init__`; add `_infer_timestamps()`, `addTime()`, `getStartDate()`, `getDuration()` |
| `tests/test_hybrid_api.py` | Add `TestHybridTimestamps` class (5–6 test cases) |
