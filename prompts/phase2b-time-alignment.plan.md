# Phase 2B — Time Alignment Layer: Detailed Implementation Plan

*Created: 2026-06-04 — based on analysis of `examples/plot_hybrid_with_pupitre_tdms.py`
and `docs/hybrid_refactoring_notes.md` items 2, 3, 4.*

---

## Background

Phase 2B goal: given multiple data sources (pupitre `.txt`, pigbrother `.tdms`,
hybrid kHz `.bin`, hybrid RMS `.rms`) recorded on the same day, compute a
per-source offset in seconds so all traces can be overlaid on a shared time axis.

A working prototype exists in `examples/plot_hybrid_with_pupitre_tdms.py`
(`plot_comparison`).  It uses "seconds since midnight" as common x-axis but has
two latent bugs that must be fixed before the approach can be formalised.

---

## Root-Cause Bugs in the Current Prototype

### Bug 1 — `hours` parameter semantics are inconsistent

`HybridData.read_khz_variable` and `read_rms_variable` both have the explicit
comment *"user-supplied hours are French local time"* and convert UTC filename
hours → Europe/Paris before comparing.  But `analysis/loaders.py` treats
`hours` as UTC hours directly, and `compute_hour_t0` (which drives the kHz time
axis) is unambiguously UTC.

The demonstrator passes `--hours 10:13` → `t0 = hours[0] * 3600 = 36000` s,
intending this as a UTC offset.  But `read_khz_variable` treats `hours[0]=10`
as a **local** hour and loads the file at UTC 08.  So `global_t0 =
compute_hour_t0(first_bin) = 08:00:00 UTC = 28800 s`, while `t0 = 36000 s` —
the x-axis reference is wrong by 2 h (CEST offset).

### Bug 2 — per-source offsets use filename-parsed local time

```python
pupitre_t0 = seconds_since_midnight(parse_txt_filename(mdata.FileName))  # LOCAL
t0         = hours[0] * 3600                                              # treated as UTC
offset     = pupitre_t0 - t0                                              # wrong by DST offset
```

`parse_txt_filename` returns a naive datetime from the filename (local French
time); `hours[0]` is treated as UTC.  In summer (CEST = UTC+2) a session
starting at 10:05 local / 08:05 UTC computes an offset of 7500 s instead of 300 s.

The fix for both bugs is the same: use `get_time_range()[0]` (naive UTC
`datetime`) for every source and compute offsets in UTC space.

---

## Task Breakdown

### B0 — kHz HH = UTC ✅ Already done

`fepc_reader.py:compute_hour_t0` already builds `ZoneInfo("UTC")`; `tz_name` is
a backward-compat no-op.  No action needed.  Remove the "pending FEPC designer
input" note from the roadmap.

---

### B0.5 — Resolve `hours` parameter semantics  *(~3 h)*

**Decision:** `hours` = **UTC** everywhere — consistent with bin filenames,
`analysis/loaders.py`, and `compute_hour_t0`.

**Files to change:**

1. **`python_magnetrun/hybrid/utils.py`** — add utility (kept for display/UI use):

   ```python
   def utc_hour_to_local(utc_h: int, date_str: str, tz: str = "Europe/Paris") -> int:
       """Convert a UTC integer hour to the equivalent local hour for the given date."""
       import datetime as _dt
       from zoneinfo import ZoneInfo
       d = _dt.date.fromisoformat(date_str)
       return (
           _dt.datetime(d.year, d.month, d.day, utc_h, 0, 0, tzinfo=ZoneInfo("UTC"))
           .astimezone(ZoneInfo(tz))
           .hour
       )
   ```

2. **`python_magnetrun/hybrid/hybrid_data.py:read_khz_variable`** (lines 489–512):

   Remove the 21-line UTC→local block.  Replace with:
   ```python
   if hours is not None:
       bin_files = [f for f in bin_files
                    if _safe_int(f.name[:2]) in hours]
   ```
   where `_safe_int` is a module-level helper (or inline `try/except`).

3. **`python_magnetrun/hybrid/hybrid_data.py:read_rms_variable`** (lines 747–770):

   Remove the 24-line UTC→local block.  Replace with:
   ```python
   if hours is not None:
       files_to_load = [f for f in rms_files
                        if self._parse_rms_filename_hour(f) in hours]
   ```

4. **`python_magnetrun/analysis/processing.py:load_hybrid_data`** (~line 582):

   `_utc_hour_to_local` is used to build `hours_set` from file paths.  After
   B0.5 the filtering inside `HybridData` is UTC-based, so `hours_set` passed
   to `getData` must also be UTC hours — i.e., just `int(Path(f).name[:2])`.
   Remove the conversion closure and use UTC hours directly.

5. **`python_magnetrun/analysis/loaders.py`** — already UTC; no change needed.

**Tests:** `tests/test_hybrid_*.py` — check that `hours` filtering still selects
the correct files.  Add a test that explicitly verifies a UTC-10 file is
selected when `hours={10}`.

---

### B1 — Fix `HybridRun.get_time_range()`  *(~2 h, depends on B0.5)*

**File:** `python_magnetrun/hybrid/hybrid_run.py` (~line 918)

Current implementation returns `(start_of_day, start_of_day + 1 day)`.

**Add private helper** (bottom of `hybrid_data.py`, or in
`hybrid/kHz/fepc_reader.py` next to `compute_hour_t0`):

```python
def _khz_first_last_utc(hdata: "HybridData") -> tuple[float, float]:
    """Return (first_t0, last_t0_end) as Unix UTC timestamps across all FEPC systems."""
    import math
    all_utc_hours: list[int] = []
    for key, files in hdata._info.khz_files.items():
        if key.endswith("_cfg"):
            continue
        for f in files:
            try:
                all_utc_hours.append(int(Path(f).name[:2]))
            except ValueError:
                pass
    if not all_utc_hours:
        raise RuntimeError("No kHz bin files found — cannot determine time range")
    first_h = min(all_utc_hours)
    last_h  = max(all_utc_hours)
    date = hdata.date_str  # e.g. "2025-01-27"
    t0_first = compute_hour_t0(f"{first_h:02d}placeholder.bin_{date}", date)
    # Simpler: build directly
    import datetime as _dt
    from zoneinfo import ZoneInfo
    d = _dt.date.fromisoformat(date)
    t0_start = _dt.datetime(d.year, d.month, d.day, first_h, 0, 0,
                            tzinfo=ZoneInfo("UTC")).timestamp()
    t0_end   = _dt.datetime(d.year, d.month, d.day, last_h + 1, 0, 0,
                            tzinfo=ZoneInfo("UTC")).timestamp()
    return t0_start, t0_end
```

**Update `HybridRun.get_time_range()`:**

```python
def get_time_range(self) -> tuple[datetime, datetime]:
    if self.HybridData is None:
        raise RuntimeError("No HybridData associated")
    from datetime import timezone
    from .hybrid_data import _khz_first_last_utc
    t0_start, t0_end = _khz_first_last_utc(self.HybridData)
    to_naive = lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)
    return to_naive(t0_start), to_naive(t0_end)
```

Note: `_khz_first_last_utc` does not call `compute_hour_t0(filepath, …)` with a
real file path — it builds the UTC datetime directly.  The function signature of
`compute_hour_t0` requires a real filename to extract `HH`; instead, build the
datetime in-line as shown.  Keep `compute_hour_t0` as-is (called with real
paths inside `read_fepc_variable`).

**Tests:** assert `get_time_range()` returns naive UTC datetimes; assert the
hour matches the lowest UTC hour present in the bin files.

---

### B2 — Fix RMS time origin  *(~30 min, independent)*

**File:** `python_magnetrun/hybrid/hybrid_data.py:read_rms_variable` (~line 810)

Current:
```python
time_ns = timestamps.to_numpy().astype("datetime64[ns]").astype(np.int64)
time = (time_ns - time_ns[0]) / 1e9  # relative to first sample — origin lost
```

Change to seconds since UTC midnight of the recording date:

```python
import datetime as _dt
from zoneinfo import ZoneInfo
_date = _dt.date.fromisoformat(self.date_str)
midnight_utc = _dt.datetime(_date.year, _date.month, _date.day, 0, 0, 0,
                             tzinfo=ZoneInfo("UTC"))
midnight_ns = np.int64(midnight_utc.timestamp() * 1e9)
time = (time_ns - midnight_ns) / 1e9  # seconds since UTC midnight
```

This gives RMS and kHz (when not hours-filtered) a compatible origin.
For alignment purposes, `get_time_range()[0]` still provides the authoritative
UTC anchor — B2 is an internal consistency improvement.

---

### B2.5 — Fix `plot_rms_variable` double-read bug  *(~1 h, from item 3)*

**File:** `python_magnetrun/hybrid/plotting.py` (~lines 629–673)

In highlight mode `orig_data, orig_time` are stashed on line 629 but the two
highlight branches re-call `hybrid_data.read_rms_variable()` instead of using
the stash.  Fix: replace the two re-read calls with `orig_data`/`orig_time`.
Mirror the kHz version which already does this correctly.

This is an S-effort independent fix; do alongside B2.

---

### B3 — Add `align_to_common_time()`  *(~1 h, depends on B1)*

**File:** `python_magnetrun/utils/timestamps.py`

```python
from datetime import datetime
from typing import Protocol, runtime_checkable

@runtime_checkable
class _HasTimeRange(Protocol):
    def get_time_range(self) -> tuple[datetime, datetime]: ...


def align_to_common_time(
    sources: list[_HasTimeRange],
    reference: datetime | None = None,
) -> dict[int, float]:
    """
    Compute per-source time offsets (seconds) relative to a common UTC reference.

    Parameters
    ----------
    sources:
        Any objects implementing ``get_time_range() -> (naive_utc, naive_utc)``.
        Typically ``MagnetRun`` and ``HybridRun`` instances.
    reference:
        Explicit reference datetime (naive UTC).  Defaults to the earliest
        ``get_time_range()[0]`` across all sources.

    Returns
    -------
    dict mapping ``id(source)`` to offset in seconds.
    Caller adds ``offset`` to the time array returned by ``source.getData()``.

    Example
    -------
    offsets = align_to_common_time([hrun, pupitre_run, tdms_run])
    data, time = hrun.getData(key)
    aligned_time = time + offsets[id(hrun)]
    """
    t0s = {id(s): s.get_time_range()[0] for s in sources}
    ref = reference or min(t0s.values())
    return {k: (t0 - ref).total_seconds() for k, t0 in t0s.items()}
```

**Tests:** construct mock sources with known `get_time_range()` returns; verify
offsets sum to zero for the earliest source and are positive for later ones.

---

### B4 — Refactor demonstrator  *(~2 h, depends on B0.5 + B1 + B3)*

**File:** `examples/plot_hybrid_with_pupitre_tdms.py`

**Remove:**
- `t0_from_filename()` helper (lines 77–85)
- `t0_from_tdms_filename()` helper (lines 88–96)
- `t0 = 0.0 / hours[0] * 3600` block in `plot_comparison` (lines 158–162)
- `from python_magnetrun.utils.timestamps import ... seconds_since_midnight` (if unused after refactor)
- Remaining `print()` calls → `logger.*`

**In `plot_comparison`, add `hrun: HybridRun` parameter and compute offsets:**

```python
from python_magnetrun.utils.timestamps import align_to_common_time

hybrid_origin: datetime = hrun.get_time_range()[0]  # naive UTC

# hybrid kHz time already in seconds from hybrid_origin (after B1 fix)
# pupitre offset:
all_sources = [hrun] + pupitre_data + tdms_data
offsets = align_to_common_time(all_sources, reference=hybrid_origin)

# in pupitre loop:
offset = offsets[id(pdata)]
pupitre_time = df["t"].to_numpy() + offset

# in tdms loop:
offset = offsets[id(tdata)]
tdms_time = df["t"].to_numpy() + offset
```

X-axis label becomes `"Time (seconds from {hybrid_origin.strftime('%H:%M UTC')})"`.

---

## Sequencing

```
B0   ✅ done
B0.5 ──► B1 ──► B3 ──► B4
B2   (parallel with B0.5/B1)
B2.5 (parallel with B0.5/B1)
```

| Task | Effort | Risk | Depends on |
|------|--------|------|-----------|
| B0.5 hours=UTC | ~3 h | Low | — |
| B2 RMS origin | ~30 min | Low | — |
| B2.5 double-read bug | ~1 h | Low | — |
| B1 `get_time_range()` | ~2 h | Low | B0.5 |
| B3 `align_to_common_time` | ~1 h | Low | B1 |
| B4 refactor demonstrator | ~2 h | Low | B1, B3 |
| **Total** | **~1 day** | | |

---

## What Phase 2B Does NOT Cover

- Channel name mapping (`HYBRID_TO_PUPITRE_MAP` etc.) — that is Phase 2E.
- Extending `plot_data()` for hybrid — that is Phase 2C.
- Trigger / VProcess integration into `HybridData` — that is needed for Phase 2C
  completeness (refactoring notes item 10); track separately.
- `_resolve_backend` consolidation (item 6) — S-effort, do opportunistically.
- `_BinaryFileReaderBase` extraction (item 1) — L-effort, track in Stream 3.

---

## Related Items from `docs/hybrid_refactoring_notes.md`

| Item | Interaction | When |
|------|------------|------|
| Item 2 — UTC→local x4 | B0.5 consolidates these | **With B0.5** |
| Item 3 — double-read bug | Fix before Phase 2C | **B2.5** |
| Item 4 — RMS missing downsample | Fix while merging plot helpers | Phase 2C |
| Item 10 — trigger/vprocess not in HybridData | Phase 2C scope (kHz+RMS only for now) | Phase 2C |

---

## Success Criteria

- [ ] `HybridRun.get_time_range()` returns naive UTC pair derived from actual bin files
- [ ] `read_khz_variable(hours=[10, 11])` selects files `10*.bin`, `11*.bin` (UTC)
- [ ] `read_rms_variable(hours=[10, 11])` same convention
- [ ] `align_to_common_time([hrun, pupitre_run])` returns correct offsets in UTC
- [ ] Demonstrator `plot_comparison` produces correctly-aligned traces in both CET and CEST
- [ ] `plot_rms_variable` highlight mode uses stashed data (no double-read)
- [ ] All 866+ existing tests still pass
