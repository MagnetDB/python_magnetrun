> This file is superseded by [CHANGELOG.md](CHANGELOG.md).

# Breaking Changes — v0.3.0

## Timestamp convention unified: all `timestamp` columns now store naive UTC

### What changed

| Class | Before (≤ 0.2.x) | After (≥ 0.3.0) |
|---|---|---|
| `PandasMagnetData.Data["timestamp"]` | naive **local** time | naive **UTC** |
| `TdmsMagnetData.Data[group]["timestamp"]` | naive UTC (unchanged) | naive UTC (unchanged) |
| `start_timestamp` / `end_timestamp` | naive UTC (unchanged) | naive UTC (unchanged) |

### Affected methods

#### `addTime(time_zone="Europe/Paris")`

Added `time_zone` parameter to all three classes (`MagnetDataBase`, `PandasMagnetData`,
`TdmsMagnetData`).

**`PandasMagnetData.addTime`** now stores `timestamp` as naive UTC instead of naive local time.
The `time_zone` argument tells it which timezone the source `Date`/`Time` columns are in
(default: `"Europe/Paris"`).

**`TdmsMagnetData.addTime`** is now *eager*: it calls both `addTdmsTime()` and
`addTdmsTimestamp()` for all non-`Infos` groups in one call.  Previously only `addTdmsTime()`
was called and callers had to invoke `addTdmsTimestamp()` separately.

#### `extractTimeData(timerange, group=None, time_zone="Europe/Paris")`

**Timerange format changed** — was `"HH:MM:SS;HH:MM:SS"`, now
`"YYYY-MM-DD HH:MM:SS;YYYY-MM-DD HH:MM:SS"`.  The strings are interpreted as **local time**
in the `time_zone` timezone and converted to UTC internally before filtering.

Both boundaries remain inclusive.

`PandasMagnetData.extractTimeData` now raises `RuntimeError` if `addTime()` has not been
called yet (previously it would silently read the raw `Time` string column, often returning
wrong results).

`TdmsMagnetData.extractTimeData` now requires `group` to be non-`None` and raises
`RuntimeError` if `addTime()` has not been called yet.  Previously it called
`addTdmsTimestamp()` lazily and returned a boolean Series instead of a DataFrame.

#### `plotData(..., time_zone="Europe/Paris")`

Added `time_zone` parameter.  When `x == "timestamp"` the UTC-stored column is converted to
naive local time before being passed to matplotlib, so axis labels display local time.

Previously `PandasMagnetData.plotData` displayed local time incidentally (because the column
was stored as local).  The visual output is unchanged; the storage convention is now explicit.

### Migration guide

```python
# Old (≤ 0.2.x)
md.addTime()
df = md.extractTimeData("09:53:00;10:00:00")

# New (≥ 0.3.0)
md.addTime()                          # time_zone="Europe/Paris" is the default
df = md.extractTimeData("2025-11-05 09:53:00;2025-11-05 10:00:00")
```

If you read `timestamp` values directly and compare them to other timestamps, ensure both
sides are naive UTC.  Previously `PandasMagnetData` timestamps were naive local — code that
compared them to TDMS timestamps or `start_timestamp` would have been silently wrong.

### What is NOT changed

- `start_timestamp` / `end_timestamp` attributes — already naive UTC, no change.
- `t` column — elapsed seconds, no change.
- `TdmsMagnetData.addTdmsTime()` / `addTdmsTimestamp()` — still available as lower-level
  helpers; their signatures and behaviour are unchanged.
- `analysis/loaders.py`, `analysis/processing.py` — use `timestamp` as UTC for
  synchronization; already correct, no changes needed.
- `HybridData` — out of scope for this release; needs its own follow-up.
