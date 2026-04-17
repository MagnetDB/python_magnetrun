# Downsampling Refactoring Plan

Date: 2026-04-17

Effort key: **S** = ~1 h, **M** = half-day, **L** = 1–2 days.

---

## Motivation

`downsample_data()` currently lives inside `hybrid/hybrid_run.py` and is unreachable by
`PandasMagnetData`, `TdmsMagnetData`, or the `analysis/` pipeline.  Two separate concepts
(`downsample: int` target-points and `downsample_method: str`) travel as loose fields in
`LoadOptions`, and the `DownsamplingLoader` protocol accepts only the integer — no method
selection.  `analysis/processing.py` uses a third model (`downsample_percent: float`).

The goal is a single, shared downsampling module usable by every data type, with
method-specific parameters cleanly encapsulated.

---

## Target design

### New file: `python_magnetrun/utils/downsampling.py`

```python
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

@dataclass
class DownsampleConfig:
    n_out: int                          # target number of output points
    method: str = "stride"              # 'minmax_lttb' | 'lttb' | 'minmax' | 'stride'
    # method-specific knobs (add more as new algorithms are introduced)
    bucket_size: int | None = None      # 'minmax' only — auto-computed when None

    @classmethod
    def from_percent(cls, data_len: int, percent: float, method: str = "stride") -> "DownsampleConfig":
        """Build a config from a percentage of dataset length (bridges analysis/ model)."""
        n_out = max(1, int(data_len * percent / 100.0))
        return cls(n_out=n_out, method=method)


def downsample_arrays(
    data: np.ndarray,
    time: np.ndarray,
    config: DownsampleConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample a (data, time) pair according to config."""
    ...  # move body from hybrid_run.py:319-408


def downsample_dataframe(
    df: pd.DataFrame,
    time_col: str,
    value_cols: list[str],
    config: DownsampleConfig,
) -> pd.DataFrame:
    """Downsample a multi-column DataFrame; applies per-column and re-assembles."""
    ...
```

`DownsampleConfig.from_percent()` is the bridge between the `analysis/` percentage model and
the new config-based model, so no breaking changes are needed in `analysis/`.

---

## Steps

### Step 1 — Create `python_magnetrun/utils/downsampling.py` *(effort: S)*

- Move `downsample_data()` body from `hybrid_run.py:319-408` into `downsample_arrays()`.
- Keep the NaN-strip guard, all four methods (`minmax_lttb`, `lttb`, `minmax`, `stride`),
  and the `HAS_TSDOWNSAMPLE` availability check (import it locally in this module).
- Add `DownsampleConfig` dataclass.
- Add `downsample_dataframe()` — iterates `value_cols`, calls `downsample_arrays()` per column,
  re-assembles into a single DataFrame with the downsampled time column.
- Add `__all__` = `["DownsampleConfig", "downsample_arrays", "downsample_dataframe"]`.
- Export from `python_magnetrun/utils/__init__.py`.

### Step 2 — Update `hybrid_run.py` *(effort: S)*

- Delete `downsample_data()` (lines 319–408) and the local `HAS_TSDOWNSAMPLE` / import block
  (lines 46–53).
- Change `LoadOptions` fields:

  ```python
  # before
  downsample: int | None = None
  downsample_method: str = "stride"

  # after
  downsample: DownsampleConfig | None = None
  ```

- Update `HybridRun.getData()` call site (line ~690):

  ```python
  # before
  data, time = downsample_data(data, time, opts.downsample, opts.downsample_method)

  # after
  data, time = downsample_arrays(data, time, opts.downsample)
  ```

- Update `HybridRun.getData()` signature: `downsample: DownsampleConfig | None = None`
  and the override block (lines 619–628) that builds a new `LoadOptions` from a bare int
  — callers that still pass an int should be converted or a deprecation shim added.

- Update `HybridRun.compare_with_magnetrun()` (line ~877): accept `DownsampleConfig` instead
  of bare `int`.

- Update the cache key (line ~638): `cache_key = f"{key}:{opts.downsample}:{opts.hours}"`
  — `DownsampleConfig` must be hashable or the key must be derived from its fields.
  Either add `__hash__` or use `f"{key}:{opts.downsample.n_out}:{opts.downsample.method}:{opts.hours}"`.

### Step 3 — Update `data_protocol.py` *(effort: S)*

- Import `DownsampleConfig` from `utils.downsampling`.
- Update `DownsamplingLoader.getData()` signature:

  ```python
  def getData(
      self,
      key: str | None = None,
      downsample: DownsampleConfig | None = None,
  ) -> Any: ...
  ```

- The existing `get_downsampled_data()` utility at the bottom of `data_protocol.py` (line ~257)
  also needs updating: replace `target_points: int` with `config: DownsampleConfig`.

### Step 4 — Add downsampling to `PandasMagnetData` *(effort: S)*

- `PandasMagnetData.getData(key, downsample: DownsampleConfig | None = None)`:

  ```python
  if downsample is not None and len(df) > downsample.n_out:
      df = downsample_dataframe(df, time_col="t", value_cols=[key], config=downsample)
  ```

  When `key` is `None` (return full DataFrame), downsample all numeric columns.

- `PandasMagnetData` now satisfies the `DownsamplingLoader` protocol structurally.

### Step 5 — Add downsampling to `TdmsMagnetData` *(effort: S)*

- Same pattern as Step 4.
- TDMS data is returned as `(data_array, time_array)` tuples in places — use
  `downsample_arrays()` directly there; use `downsample_dataframe()` for DataFrame paths.

### Step 6 — Reconcile `analysis/processing.py` *(effort: S)*

- `PlotOptions.downsample_percent: float = 100.0` can stay as-is for the CLI surface.
- Inside the analysis pipeline, convert before calling any `getData()`:

  ```python
  ds_config = DownsampleConfig.from_percent(len(df), opts.downsample_percent, method="stride")
  ```

- No public API change required.

### Step 7 — Fix `pyproject.toml` *(effort: S)*

This step directly resolves REVIEW.md issue #12.

```toml
[project.optional-dependencies]
hybrid = ["tsdownsample>=1.0"]
```

Add `tsdownsample` to the `hybrid` extras group.  The `try/except ImportError` guard in
`utils/downsampling.py` is kept so the rest of the package remains importable without it.
Document the soft requirement in the module docstring.

---

## File change summary

| File | Change |
|---|---|
| `python_magnetrun/utils/downsampling.py` | **New** — `DownsampleConfig`, `downsample_arrays`, `downsample_dataframe` |
| `python_magnetrun/utils/__init__.py` | Export the three new names |
| `python_magnetrun/hybrid/hybrid_run.py` | Delete `downsample_data()`, update `LoadOptions`, `getData()`, `compare_with_magnetrun()` |
| `python_magnetrun/hybrid/data_protocol.py` | Update `DownsamplingLoader.getData()`, `get_downsampled_data()` |
| `python_magnetrun/magnetdata_pandas.py` | Add `downsample` param to `getData()` |
| `python_magnetrun/magnetdata_tdms.py` | Add `downsample` param to `getData()` |
| `python_magnetrun/analysis/processing.py` | Convert `downsample_percent` → `DownsampleConfig` internally |
| `pyproject.toml` | Add `tsdownsample` to `hybrid` extras |

---

## Interaction with `REVIEW.md` overall plan

### Directly resolves

- **Issue #12** (`tsdownsample` undeclared dependency) — Step 7 adds it to `pyproject.toml`
  as a `hybrid` extras group entry.

### Enables / unblocks

- **Cross-domain comparison Phase D (`CHANNEL_ALIASES` + `KeyMapping`)** and
  **Phase E (`ComparisonSession`)** — both will need to downsample data from heterogeneous
  sources for aligned plotting.  Having `DownsampleConfig` as a shared type means
  `ComparisonSession` can accept one config and pass it to `MagnetRun.getData()`,
  `HybridRun.getData()`, and future `SimulationRun.getData()` uniformly.

- **Phase A3 protocol compliance tests** — once `PandasMagnetData` and `TdmsMagnetData`
  implement `DownsamplingLoader`, `tests/test_protocol.py` can include downsampling
  compliance checks for all three concrete types.

### Sequencing constraint

This plan should be executed **after** Phase A1–A3 (add `getDomain()` to protocol, add
`MagnetRun.get_time_range()`, write `tests/test_protocol.py`) because:

1. Phase A3 writes `tests/test_protocol.py` — the downsampling plan adds new protocol
   requirements to that test file, so doing A3 first avoids a two-touch edit.
2. The `DownsamplingLoader` protocol change in Step 3 is an API change; it is cleaner to
   land it in one shot alongside other protocol work.

Alternatively, Steps 1–2 (create the module, update `hybrid_run.py`) are fully independent
and can be done at any time without touching the protocol.

### Does not conflict with

- Timestamp convention work (`hybriddata-timestamp-plan.md`) — orthogonal concern.
- `HousingConfig` consolidation — already done, unrelated.
- `MagnetData` factory replacement — already done, unrelated.
- CLI consolidation (issues #9, #10) — orthogonal.

---

## Recommended execution order

1. **Now (independent):** Steps 1–2 — create `utils/downsampling.py`, update `hybrid_run.py`.
   Zero protocol surface change; safe to land immediately.
2. **With Phase A3:** Steps 3 + write/extend `tests/test_protocol.py` for downsampling compliance.
3. **After A3:** Steps 4–6 — add downsampling to `PandasMagnetData`, `TdmsMagnetData`,
   `analysis/processing.py`.
4. **Anytime:** Step 7 — `pyproject.toml` extras entry (trivial, no code change).
