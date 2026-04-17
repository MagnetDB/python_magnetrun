# Cross-Domain Comparison — Architecture Evolution Plan

*Created: 2026-04-08 — Updated: 2026-04-16*

## Goal

Enable unified comparison of operational data (pupitre, pigbrother, hybrid/FEPC),
simulation data (feelpp, ensight, magnettools), and magnetic field measurements
(bprofile) — including overlaid plots, side-by-side views, and quantitative metrics
— through a single, consistent API.

---

## Prerequisites

| Prerequisite | File | Status |
|---|---|---|
| `load_magnetdata()` standalone factory | `magnetdata.py` | **Done** — shim replaced in commit `274d6bd` |
| `get_time_range()` on data objects | `magnetdata_pandas.py:418`, `magnetdata_tdms.py:409` | **Done** — concrete implementations exist |
| `DataProvider`/`DataLoader` duplication resolved | `hybrid/hybrid_run.py`, `hybrid/data_protocol.py` | **Done** — `DataProvider` deleted (Phase A0) |
| `get_time_range()` added to `DataLoader` protocol | `hybrid/data_protocol.py:177` | **Done** — `HybridRun.get_time_range()` also exists (`hybrid_run.py:822`) |
| `getDomain()` added to `DataLoader` protocol | `hybrid/data_protocol.py` | **Blocking** — not yet in protocol |
| `MagnetRun.get_time_range()` delegation wrapper | `MagnetRun.py` | **Blocking** — wrapper missing |
| `getDomain()` on `MagnetRun` and `HybridRun` | `MagnetRun.py`, `hybrid/hybrid_run.py` | **Blocking** — not yet implemented |

---

## Architecture Target

```
DataLoader protocol (extended — all sources)
       ↑                  ↑                  ↑
  MagnetRun          SimulationRun        BFieldRun
  HybridRun          (new adapter)        (new adapter)
                      wraps:               wraps:
                      EnsightMagnetData    BProfileMagnetData
                      FeelppMagnetData
                      MagnetToolsData (new)
                            ↓
              ComparisonSession(sources, key_map)
                 ├── plot_comparison()        → matplotlib Figure
                 ├── compute_metrics()        → dict[str, DistanceResult]
                 └── report()                → str / dict
```

---

## Phase A — Consolidate and extend the `DataLoader` protocol

**Files:** `python_magnetrun/hybrid/data_protocol.py`, `python_magnetrun/hybrid/hybrid_run.py`,
`python_magnetrun/MagnetRun.py`

### A0 — Remove `DataProvider` duplication *(done)*

`DataProvider` has been deleted from `hybrid_run.py`. All type annotations now use `DataLoader`
from `hybrid/data_protocol.py`.

### A1 — Add `getDomain()` and `get_time_range()` to the protocol

`get_time_range()` has been added to `DataLoader` (`data_protocol.py:177`). `getDomain()` is
still missing. The current `DataLoader` protocol requires `getData`, `getKeys`, `getType`,
`getSite`, `getHousing`, `get_time_range`. Still needed — add `getDomain()`:

```python
@runtime_checkable
class DataLoader(Protocol):
    # ... existing methods unchanged ...

    def getDomain(self) -> str:
        """Return 'operational', 'simulation', or 'bfield'."""
        ...

    def get_time_range(self) -> tuple[datetime, datetime]:
        """Return (start, end) of the dataset as UTC datetimes."""
        ...
```

### A2 — Add missing methods to existing implementations

`get_time_range()` is **already concrete** in the data layer:
- `PandasMagnetData.get_time_range()` — `magnetdata_pandas.py:418` (uses Date/Time columns)
- `TdmsMagnetData.get_time_range()` — `magnetdata_tdms.py:409` (uses wf_start_time)
- `HybridRun.get_time_range()` — `hybrid_run.py:822` (already implemented)

What is still missing (`getDomain()` and `MagnetRun` delegation):

**`MagnetRun.py`** — add two methods:

```python
def get_time_range(self) -> tuple:
    """Delegate to the underlying MagnetData object."""
    return self.MagnetData.get_time_range()

def getDomain(self) -> str:
    return "operational"
```

**`hybrid_run.py`** — add one method (after A0 removes `DataProvider`):

```python
def getDomain(self) -> str:
    return "operational"
```

### A3 — Write a protocol compliance test

Add `tests/test_protocol.py`:

```python
from python_magnetrun.hybrid.data_protocol import DataLoader
from python_magnetrun.MagnetRun import MagnetRun
from python_magnetrun.hybrid.hybrid_run import HybridRun

def test_magnetrun_satisfies_protocol():
    assert isinstance(MagnetRun(), DataLoader)

def test_hybridrun_satisfies_protocol():
    # construct minimal HybridRun without loading files
    ...
    assert isinstance(hrun, DataLoader)
```

---

## Phase B — `SimulationRun` adapter

**New file:** `python_magnetrun/simulation/simulation_run.py`

### B1 — Create the adapter class

`EnsightMagnetData`, `FeelppMagnetData`, and the new `MagnetToolsData` all inherit
from `PandasMagnetData` but none implement `DataLoader`. Wrap them:

```python
class SimulationRun:
    """DataLoader-compatible wrapper for simulation data sources.

    Wraps EnsightMagnetData, FeelppMagnetData, or MagnetToolsData and
    exposes them through the unified DataLoader protocol.
    """

    def __init__(
        self,
        data: PandasMagnetData,
        housing: str = "unknown",
        site: str = "",
        time_column: str | None = None,  # column name for time axis, if any
    ) -> None:
        self._data = data
        self.Housing = housing
        self.Site = site
        self._time_column = time_column

    @classmethod
    def from_ensight(cls, filename: str, housing: str = "", site: str = "") -> "SimulationRun":
        from ..magnetdata_pandas import EnsightMagnetData
        data = EnsightMagnetData.fromcsv(filename)
        return cls(data, housing=housing, site=site)

    @classmethod
    def from_feelpp(cls, filename: str, housing: str = "", site: str = "") -> "SimulationRun":
        from ..magnetdata_pandas import FeelppMagnetData
        data = FeelppMagnetData.fromcsv(filename)
        return cls(data, housing=housing, site=site)

    @classmethod
    def from_magnettools(cls, filename: str, housing: str = "", site: str = "") -> "SimulationRun":
        from .magnettools_reader import load_magnettools
        data = load_magnettools(filename)
        return cls(data, housing=housing, site=site)

    # DataLoader protocol implementation
    def getData(self, key: str | None = None) -> pd.DataFrame:
        return self._data.getData(key)

    def getKeys(self) -> list[str]:
        return self._data.getKeys()

    def getType(self) -> int:
        return int(self._data.Type)

    def getSite(self) -> str:
        return self.Site

    def getHousing(self) -> str:
        return self.Housing

    def getDomain(self) -> str:
        return "simulation"

    def get_time_range(self) -> tuple[datetime, datetime]:
        # Simulation data may not have wall-clock time; derive from time column or index
        df = self._data.getData()
        if self._time_column and self._time_column in df.columns:
            t = pd.to_datetime(df[self._time_column])
            return t.min().to_pydatetime(), t.max().to_pydatetime()
        raise NotImplementedError("No time column specified for this simulation dataset")
```

### B2 — `magnettools` reader

**New file:** `python_magnetrun/simulation/magnettools_reader.py`

Implement `load_magnettools(filename: str) -> PandasMagnetData`.

The magnettools format needs to be confirmed with the user.
Placeholder:

```python
def load_magnettools(filename: str) -> PandasMagnetData:
    """Load a magnettools output file.

    TODO: confirm exact format (CSV columns, delimiter, header structure).
    For now, attempt generic CSV loading via load_magnetdata().
    """
    from ..magnetdata import load_magnetdata
    return load_magnetdata(filename)
```

### B3 — Register in `__init__.py`

**File:** `python_magnetrun/simulation/__init__.py` (create)

```python
from .simulation_run import SimulationRun

__all__ = ["SimulationRun"]
```

---

## Phase C — `BFieldRun` adapter

**New file:** `python_magnetrun/bfield/bfield_run.py`

### C1 — Create the adapter class

`BProfileMagnetData` inherits from `PandasMagnetData` and stores profile data with
columns `(Index, Position, Profile, ...)`. It is spatial rather than temporal, but
a time label (measurement date/run ID) can be attached.

```python
class BFieldRun:
    """DataLoader-compatible wrapper for magnetic field profile data.

    Wraps BProfileMagnetData and exposes it through the unified DataLoader protocol.
    """

    def __init__(
        self,
        data: BProfileMagnetData,
        housing: str = "unknown",
        site: str = "",
        measurement_time: datetime | None = None,
    ) -> None:
        self._data = data
        self.Housing = housing
        self.Site = site
        self._measurement_time = measurement_time

    @classmethod
    def from_csv(
        cls,
        filename: str,
        housing: str = "",
        site: str = "",
        measurement_time: datetime | None = None,
    ) -> "BFieldRun":
        from ..magnetdata_pandas import BProfileMagnetData
        data = BProfileMagnetData.fromcsv(filename)
        return cls(data, housing=housing, site=site, measurement_time=measurement_time)

    # DataLoader protocol
    def getData(self, key: str | None = None) -> pd.DataFrame:
        return self._data.getData(key)

    def getKeys(self) -> list[str]:
        return self._data.getKeys()

    def getType(self) -> int:
        return int(self._data.Type)

    def getSite(self) -> str:
        return self.Site

    def getHousing(self) -> str:
        return self.Housing

    def getDomain(self) -> str:
        return "bfield"

    def get_time_range(self) -> tuple[datetime, datetime]:
        if self._measurement_time is None:
            raise NotImplementedError("BFieldRun has no time range — spatial data only")
        return self._measurement_time, self._measurement_time
```

### C2 — Register in `__init__.py`

**New file:** `python_magnetrun/bfield/__init__.py`

```python
from .bfield_run import BFieldRun

__all__ = ["BFieldRun"]
```

---

## Phase D — Channel name normalization (`KeyMapping`)

> **Design revision** (2026-04-16): the original plan proposed a hardcoded
> `CHANNEL_ALIASES` dict in `analysis/config.py`. This is superseded by
> `field_defs.py`, which already stores cross-format aliases in the `*-defs.json`
> files under the `"aliases"` key, and provides `build_crossref()` to build an
> O(1) lookup index. Phase D should reuse that infrastructure rather than
> duplicating it.

### D0 — Extend the `*-defs.json` files with `simulation` and `bfield` aliases

The existing alias entries in `pupitre-defs.json` / `pigbrother-defs.json` already
cover `"pigbrother"` and `"hybrid"` format names. Add `"simulation"` and `"bfield"`
entries where a cross-domain correspondence exists, using the `magnetrun-field-defs`
CLI:

```bash
magnetrun-field-defs pupitre-defs.json alias-add IH simulation Icoil_helix
magnetrun-field-defs pupitre-defs.json alias-add IB simulation Icoil_bitter
magnetrun-field-defs pupitre-defs.json alias-add Field simulation B0
magnetrun-field-defs pupitre-defs.json alias-add Field bfield Profile
# ... add further entries as the simulation/bfield formats are confirmed
```

> The exact simulation field names (`Icoil_helix`, `B0`, etc.) need confirmation
> with the user — see Open Question #4.

### D1 — Add `KeyMapping` resolver

**File:** `python_magnetrun/comparison/key_mapping.py` (new, not `analysis/config.py`)

`KeyMapping` is a thin resolver on top of `field_defs.build_crossref()`. It does
**not** own alias data — that lives exclusively in the JSON files.

```python
from python_magnetrun.field_defs import build_crossref, load_defs

class KeyMapping:
    """Resolve logical channel names to source-specific key names.

    Alias data is read from *-defs.json files via field_defs.build_crossref().
    Logical keys are the canonical field names in the defs files (e.g. "IH",
    "Field"); domains are alias format names (e.g. "simulation", "bfield",
    "pigbrother", "hybrid").
    """

    def __init__(self, defs_files: dict[str, str | Path]) -> None:
        """
        Parameters
        ----------
        defs_files:
            ``{format_name: path_to_defs_json}`` — e.g.
            ``{"pupitre": "pupitre-defs.json", "simulation": "feelpp-defs.json"}``.
            Bare filenames are resolved via ``field_defs.resolve_defs_file()``.
        """
        self._index = build_crossref(defs_files)

    @classmethod
    def default(cls) -> "KeyMapping":
        """Construct with the standard bundled defs files."""
        return cls({
            "pupitre":    "pupitre-defs.json",
            "pigbrother": "pigbrother-defs.json",
            "hybrid":     "hybrid-defs.json",
        })

    def resolve(self, logical_key: str, domain: str, available_keys: list[str]) -> str | None:
        """Return the alias of *logical_key* in *domain* that is in *available_keys*.

        Checks the logical key itself first (the key may exist unchanged in the
        target domain), then falls back to the aliases recorded in the defs file.
        Returns None if no match is found.
        """
        if logical_key in available_keys:
            return logical_key
        candidates = self._index.get(domain, {}).get(logical_key, {})
        for alias in candidates.values():
            if alias in available_keys:
                return alias
        return None

    def resolve_or_raise(self, logical_key: str, domain: str, available_keys: list[str]) -> str:
        key = self.resolve(logical_key, domain, available_keys)
        if key is None:
            raise KeyError(
                f"Logical key {logical_key!r} not found in domain {domain!r}. "
                f"Available keys: {available_keys}"
            )
        return key
```

### D2 — No change to `analysis/config.py` or `analysis/processing.py`

`CHANNEL_ALIASES` is **not** added to `analysis/config.py`. The defs JSON files
are the single source of truth for alias data. `KeyMapping` is placed in
`python_magnetrun/comparison/key_mapping.py` alongside `ComparisonSession`.

The hardcoded mappings in the analysis submodule are **not** candidates for
field_defs replacement:

- `ChannelMapping` (`config.py`) maps `Référence_GR1` → `Courant_GR1` — a fixed
  TDMS-internal relationship, not a cross-format alias.
- `_get_pupitre_channel()` (`processing.py`) maps `Courant_GR1` → `IH`/`IB`
  depending on housing — housing-dependent, rightly in `HousingConfig`.

The duplicate in `analysis/cli.py:162-165` has been removed. `cli.py` now uses
`channel_map.to_dict()` and the new group-centric `ChannelMapping` API
(`get_setpoint_channel(group)`, `get_actual_channel(group)`, `groups()`).

**Optional future improvement:** `analysis/plotting.py` hardcodes axis labels.
Using `load_defs()` for symbol/unit lookup would make plots self-documenting,
but this is a quality-of-life improvement independent of Phase D.

---

## Phase E — `ComparisonSession`

**New file:** `python_magnetrun/comparison/session.py`

### E1 — Core class

```python
@dataclass
class ComparisonResult:
    logical_key: str
    sources: list[str]           # source labels
    metrics: dict[str, DistanceResult]  # keyed by "source_a vs source_b"
    time_offset: dict[str, float]       # seconds shift per source


class ComparisonSession:
    """Orchestrate cross-domain comparison of multiple DataLoader sources.

    Usage::

        session = ComparisonSession(key_map=KeyMapping())
        session.add_source(mrun,  label="pupitre")
        session.add_source(hrun,  label="hybrid-kHz")
        session.add_source(simrun, label="feelpp")

        fig = session.plot_comparison(channels=["IH", "Field"])
        results = session.compute_metrics(channels=["IH", "Field"])
    """

    def __init__(
        self,
        key_map: KeyMapping | None = None,
        resample_freq: str = "1s",           # pandas offset alias for common time axis
    ) -> None:
        self._sources: list[tuple[DataLoader, str]] = []   # (loader, label)
        self._key_map = key_map or KeyMapping()
        self._resample_freq = resample_freq
        self._aligned: dict[str, pd.DataFrame] | None = None  # cache

    def add_source(self, loader: DataLoader, label: str) -> None:
        self._sources.append((loader, label))
        self._aligned = None  # invalidate cache

    def _align(self, logical_keys: list[str]) -> dict[str, pd.DataFrame]:
        """Resample all sources to common time axis and map channel names.

        Returns dict: logical_key → DataFrame with one column per source label.
        Use existing analysis.synchronization.synchronize_data() internally.
        """
        ...

    def plot_comparison(
        self,
        channels: list[str],
        title: str = "",
        figsize: tuple | None = None,
        save_path: str | None = None,
    ) -> "plt.Figure":
        """Produce overlaid time-series plots for each logical channel.

        Delegates to analysis.plotting.plot_comparison() after alignment.
        """
        ...

    def compute_metrics(
        self,
        channels: list[str],
        reference_label: str | None = None,   # if None, use first source as reference
    ) -> dict[str, ComparisonResult]:
        """Compute DistanceResult (MAE, MAPE, Pearson, DTW) for each channel.

        Delegates to analysis.metrics.compute_all_distances().
        """
        ...

    def report(self, channels: list[str]) -> str:
        """Return a formatted text summary of metrics."""
        ...
```

### E2 — `__init__.py`

**New file:** `python_magnetrun/comparison/__init__.py`

```python
from .session import ComparisonSession, ComparisonResult

__all__ = ["ComparisonSession", "ComparisonResult"]
```

---

## Phase F — CLI entry point

**New file:** `python_magnetrun/comparison/cli.py`

### F1 — Argument structure

```
magnetrun-compare
  --pupitre FILE              # operational pupitre .txt
  --tdms FILE                 # operational pigbrother .tdms
  --hybrid-dir DIR --hybrid-date YYYY-MM-DD  # hybrid FEPC
  --feelpp FILE               # simulation Feel++
  --ensight FILE              # simulation Ensight
  --magnettools FILE          # simulation magnettools
  --bprofile FILE             # B-field measurement
  --housing HOUSING
  --site SITE
  --channels IH Field FlowH   # logical channel names (default: all common)
  --resample-freq 1s          # common time axis resolution
  --metrics                   # print metrics table
  --plot                      # show plots
  --save-dir DIR
  --reference pupitre         # which source is the reference for metrics
```

### F2 — Register in `pyproject.toml`

```toml
[project.scripts]
magnetrun-compare = "python_magnetrun.comparison.cli:main"
```

---

## Phase G — Tests

**New file:** `tests/test_comparison.py`

| Test | What it checks |
|---|---|
| `test_simulation_run_protocol` | `isinstance(SimulationRun(...), DataLoader)` |
| `test_bfield_run_protocol` | `isinstance(BFieldRun(...), DataLoader)` |
| `test_key_mapping_resolve` | Resolve "IH" in "operational" → "IH" |
| `test_key_mapping_resolve_alias` | Resolve "IH" in "simulation" → "Icoil_helix" |
| `test_key_mapping_not_found` | Returns `None` for unknown key |
| `test_comparison_session_add_source` | Session accepts 2 sources without error |
| `test_comparison_session_align` | `_align(["IH"])` returns DataFrame with 2 columns |
| `test_comparison_session_metrics` | `compute_metrics(["IH"])` returns `DistanceResult` with finite values |

Use synthetic DataFrames (no real files) for all tests except CLI smoke tests.

---

## Files to create

| File | Purpose |
|---|---|
| `python_magnetrun/simulation/__init__.py` | Package init |
| `python_magnetrun/simulation/simulation_run.py` | SimulationRun adapter |
| `python_magnetrun/simulation/magnettools_reader.py` | magnettools format reader (stub) |
| `python_magnetrun/bfield/__init__.py` | Package init |
| `python_magnetrun/bfield/bfield_run.py` | BFieldRun adapter |
| `python_magnetrun/comparison/__init__.py` | Package init |
| `python_magnetrun/comparison/key_mapping.py` | `KeyMapping` resolver (Phase D) |
| `python_magnetrun/comparison/session.py` | ComparisonSession |
| `python_magnetrun/comparison/cli.py` | `magnetrun-compare` CLI |
| `tests/test_protocol.py` | Protocol compliance tests |
| `tests/test_comparison.py` | ComparisonSession unit tests |

## Files to modify

| File | Change |
|---|---|
| `python_magnetrun/hybrid/data_protocol.py` | Add `getDomain()` to `DataLoader` protocol (Phase A1 — `get_time_range()` already done) |
| `python_magnetrun/hybrid/hybrid_run.py` | Add `getDomain() → "operational"` (Phase A2 — `DataProvider` already deleted) |
| `python_magnetrun/MagnetRun.py` | Add `get_time_range()` delegation + `getDomain() → "operational"` (Phase A2) |
| `python_magnetrun/magnetdata_base.py` | No change needed — `get_time_range()` already concrete in subclasses |
| `python_magnetrun/analysis/config.py` | **No change** — `CHANNEL_ALIASES` is not added here (see Phase D revision) |
| `pupitre-defs.json`, `pigbrother-defs.json`, `hybrid-defs.json` | Add `"simulation"` and `"bfield"` alias entries (Phase D0) |
| `pyproject.toml` | Add `magnetrun-compare` entry point; add new packages to `find_packages` (Phase F) |

---

## Implementation order

```
Phase A0 (remove DataProvider duplication)   ← do first; unblocks A1
Phase A1 (extend DataLoader protocol)        ← requires A0
Phase A2 (add getDomain/get_time_range to    ← requires A1
          MagnetRun and HybridRun)
Phase A3 (protocol compliance tests)         ← requires A2
Phase D  (KeyMapping)                        ← independent; can run in parallel with B/C
Phase B  (SimulationRun)                     ← requires A1
Phase C  (BFieldRun)                         ← requires A1
Phase E  (ComparisonSession)                 ← requires A, B, C, D
Phase F  (CLI)                               ← requires E
Phase G  (tests)                             ← run throughout; finalize with F
```

---

## Open questions (confirm with user before implementing)

1. **magnettools format**: What is the exact file format — CSV delimiter, column
   names, header lines? Does it have a time axis or is it always spatial?

2. **B-field spatial vs. temporal**: `BProfileMagnetData` stores `(Position, Profile)`.
   For comparison against simulation field maps, should `ComparisonSession` compare
   along the spatial axis (z-position) rather than time? If yes, Phase E needs a
   `spatial_comparison()` path in addition to `time_comparison()`.

3. **Simulation time axis**: Feel++ and Ensight outputs may be steady-state
   (no time axis) or time-dependent. Should `SimulationRun` expose a
   `is_transient() -> bool` property to let `ComparisonSession` choose the
   right alignment strategy?

4. **Channel alias completeness**: The `CHANNEL_ALIASES` stub above covers IH, IB,
   Field, FlowH. What other channels need cross-domain mapping (voltage probes,
   temperatures, pressures)?

5. **Reference source**: When calling `compute_metrics()`, is the pupitre data always
   the reference, or should the reference be configurable per comparison?
