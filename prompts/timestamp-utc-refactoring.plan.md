# Plan: Consistent UTC timestamp storage with local-time display

## Context

`PandasMagnetData.timestamp` is currently naive **local** time; `TdmsMagnetData.timestamp` is naive **UTC** by default. Both classes already store `start_timestamp`/`end_timestamp` as naive UTC. The goal is to unify: **all `timestamp` columns store naive UTC; conversion to local happens only at display/filter boundaries** (`plotData` and `extractTimeData`).

Secondary goal: make `addTime()` eager — it computes both `t` and `timestamp` for all groups at once, removing scattered lazy guards in `plotData` / `extractData` / `extractTimeData`.

---

## Convention (documented in `MagnetDataBase` docstring)

| attribute/column | type | value |
|---|---|---|
| `start_timestamp` / `end_timestamp` | naive `datetime` | UTC |
| `t` column | float | elapsed seconds from first sample |
| `timestamp` column | naive `pd.Timestamp` | UTC |

---

## Files to change

### 1. `python_magnetrun/magnetdata_base.py`

- Update class docstring: document UTC convention for `timestamp`.
- `addTime()` signature: add `time_zone: str = "Europe/Paris"`.
- `extractTimeData()` signature: add `time_zone: str = "Europe/Paris"`.
- `plotData()` signature: add `time_zone: str = "Europe/Paris"`.
- No logic changes in base no-ops — just signatures.

### 2. `python_magnetrun/magnetdata_pandas.py`

**`addTime(time_zone="Europe/Paris")`** — change timestamp storage to naive UTC:
```python
# Build naive local timestamp as before
_local_ts = self.Data["Date"] + self.Data["Time"]
t0 = _local_ts.iloc[0]
self.Data["t"] = (_local_ts - t0).dt.total_seconds()

# Convert local → UTC → naive
import pytz
tz = pytz.timezone(time_zone)
self.Data["timestamp"] = (
    _local_ts
    .dt.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
    .dt.tz_convert(pytz.utc)
    .dt.tz_localize(None)   # strip tzinfo → naive UTC
)
self.Data.drop(["Date", "Time"], axis=1, inplace=True)
```

**`extractTimeData(timerange, group=None, time_zone="Europe/Paris")`** — require `timestamp` column (UTC), accept local datetime strings:
```python
if "timestamp" not in self.Keys:
    raise RuntimeError("call addTime() before extractTimeData()")
trange = timerange.split(";")
import pytz
tz = pytz.timezone(time_zone)
t_start = pd.Timestamp(trange[0]).tz_localize(tz).tz_convert(pytz.utc).tz_localize(None)
t_end   = pd.Timestamp(trange[1]).tz_localize(tz).tz_convert(pytz.utc).tz_localize(None)
return self.Data[self.Data["timestamp"].between(t_start, t_end, inclusive="both")]
```

**`plotData(x, y, ..., time_zone="Europe/Paris")`** — when `x == "timestamp"`, convert UTC → naive local for display:
```python
if xcol == "timestamp":
    import pytz
    tz = pytz.timezone(time_zone)
    df["timestamp"] = (
        df["timestamp"]
        .dt.tz_localize(pytz.utc)
        .dt.tz_convert(tz)
        .dt.tz_localize(None)   # naive local for clean axis labels
    )
```

### 3. `python_magnetrun/magnetdata_tdms.py`

**`addTime(time_zone="Europe/Paris")`** — eager: compute both `t` AND `timestamp` for all groups:
```python
def addTime(self, time_zone: str = "Europe/Paris") -> int:
    self.addTdmsTime()        # adds 't' per group (wf_increment × index + offset)
    self.addTdmsTimestamp()   # adds 'timestamp' (naive UTC) per group
    return 0
```

**`addTdmsTimestamp()`** — no change needed; `timezone=None` (default) already stores naive UTC.

**Remove lazy guards** from:
- `plotData` — `if "t" not in ... addTdmsTime(...)` and `if "timestamp" not in ... addTdmsTimestamp(...)`
- `extractData` — `if "t" not in ... addTdmsTime(...)`
- `extractTimeData` — the `addTdmsTimestamp(...)` call

**`extractTimeData(timerange, group, time_zone="Europe/Paris")`** — assume `timestamp` present (require `addTime()` first), accept local strings:
```python
if group is None:
    raise RuntimeError("group is required for TDMS extractTimeData")
if "timestamp" not in self.Data[group].columns:
    raise RuntimeError("call addTime() before extractTimeData()")
trange = timerange.split(";")
import pytz
tz = pytz.timezone(time_zone)
t_start = pd.Timestamp(trange[0]).tz_localize(tz).tz_convert(pytz.utc).tz_localize(None)
t_end   = pd.Timestamp(trange[1]).tz_localize(tz).tz_convert(pytz.utc).tz_localize(None)
return self.Data[group][self.Data[group]["timestamp"].between(t_start, t_end, inclusive="both")]
```

**`plotData(..., time_zone="Europe/Paris")`** — same UTC→naive-local conversion as PandasMagnetData when `x == "timestamp"`.

---

## What is NOT changed in this plan

- `analysis/loaders.py` — already builds `timestamp` from UTC `start_timestamp` / `wf_start_time`; stays correct.
- `analysis/processing.py` — uses `timestamp` as UTC for synchronization; stays correct.
- `HybridData` — left for a separate plan (needs `start_timestamp`, `end_timestamp`, `addTime()`, and its own timestamp column strategy).
- `commands/select.py` — the timerange format changes from `"HH:MM:SS;HH:MM:SS"` to `"YYYY-MM-DD HH:MM:SS;YYYY-MM-DD HH:MM:SS"` (local time). This is a caller-side change to coordinate separately.

---

## Tests to update

### `tests/test_magnetdata.py`

**`TestAddTime`** — add UTC verification:
```python
def test_timestamp_is_utc_naive(self, simple_magnetdata):
    simple_magnetdata.addTime()
    ts = simple_magnetdata.Data["timestamp"]
    assert ts.dt.tz is None                             # no tzinfo → naive
    # 2022-03-30 21:55:17 CEST (UTC+2) → UTC = 19:55:17
    assert ts.iloc[0] == pd.Timestamp("2022-03-30 19:55:17")
```

**`TestExtractTimeData`** — rewrite to use `timestamp` column and full local datetime strings:
```python
def test_filters_by_local_time(self, simple_magnetdata):
    simple_magnetdata.addTime()
    df = simple_magnetdata.extractTimeData("2022-03-30 21:55:17;2022-03-30 21:55:18")
    assert len(df) == 2

def test_inclusive_boundaries(self, simple_magnetdata):
    simple_magnetdata.addTime()
    df = simple_magnetdata.extractTimeData("2022-03-30 21:55:17;2022-03-30 21:55:17")
    assert len(df) == 1

def test_raises_if_no_timestamp(self, simple_magnetdata):
    with pytest.raises(RuntimeError, match="call addTime"):
        simple_magnetdata.extractTimeData("2022-03-30 21:55:17;2022-03-30 21:55:18")
```

### `tests/test_magnetdata_tdms.py`

**New `TestAddTimeTdms`** — verify `addTime()` also populates `timestamp`:
```python
def test_addTime_adds_timestamp_column(self):
    # needs a fixture with wf_start_time in Groups
    tdms = _make_tdms_with_wf_start_time()
    tdms.addTime()
    assert "timestamp" in tdms.Data["GrpX"].columns
    assert tdms.Data["GrpX"]["timestamp"].dt.tz is None  # naive UTC

def test_addTime_no_lazy_guards_needed(self):
    # plotData / extractData should not call addTdmsTime internally
    tdms = _make_tdms_with_wf_start_time()
    tdms.addTime()
    # extractData must work without any internal addTdmsTime call
    df = tdms.extractData(["GrpX/ChA", "t"])
    assert "t" in df.columns
```

---

## Validation steps

1. **Run existing test suite** — all previously passing tests should still pass (updated tests replace old ones):
   ```bash
   source magnetrun-env/bin/activate
   pytest tests/test_magnetdata.py tests/test_magnetdata_tdms.py -v
   ```

2. **DST correctness check** — `2022-03-30 21:55:17` is CEST (UTC+2), stored UTC = `19:55:17`:
   ```python
   md = PandasMagnetData("test.txt", {}, [...], simple_df)
   md.addTime()
   assert md.Data["timestamp"].iloc[0] == pd.Timestamp("2022-03-30 19:55:17")
   ```

3. **Plot smoke test** — x-axis should display local time labels:
   ```python
   md = PandasMagnetData.fromtxt("data/sample.txt")
   md.addTime()
   fig, ax = plt.subplots()
   md.plotData(x="timestamp", y="Field", ax=ax)
   plt.savefig("/tmp/timestamp_plot.png")
   # Inspect: x-axis labels should be local time, not UTC
   ```

4. **TDMS eager addTime smoke** — all groups must have both `t` and `timestamp` after one `addTime()` call:
   ```python
   md = TdmsMagnetData.fromtdms("data/sample.tdms")
   md.addTime()
   for gname in md.Data:
       if gname == "Infos":
           continue
       assert "t" in md.Data[gname].columns
       assert "timestamp" in md.Data[gname].columns
   ```

5. **extractTimeData round-trip** — local input → UTC filter → correct rows returned:
   ```python
   df = md.extractTimeData("2025-11-05 09:53:00;2025-11-05 10:00:00", group="Courants_Alimentations")
   assert len(df) > 0
   ```
