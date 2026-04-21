# Plotting Refactoring Plan

Date: 2026-04-20

Effort key: **S** = ~1 h, **M** = half-day, **L** = 1–2 days.

> **Likely feature flag** — items marked *(likely)* are planned but not yet
> committed.  They are included here so the core design does not foreclose them.

---

## Motivation

Plotting logic is currently scattered across at least five locations:

| File | Content |
|------|---------|
| `analysis/plotting.py` (933 lines) | `PlotStyle`, `PlotColors`, downsampling, annotations, `plot_data()` |
| `utils/plots.py` (216 lines) | Simple scatter / time-series utilities |
| `hybrid/plotting.py` | kHz / RMS plots — duplicates downsampling logic |
| `magnetdata_pandas.py:577–645` | `plotData()` — thin wrapper |
| `commands/plot.py` (318 lines) | CLI mixing I/O with plot business logic |

Problems this causes:
- `PlotStyle` / `PlotColors` live in `analysis/` but are needed everywhere.
- Downsampling is implemented three different ways in three modules (addressed
  separately in `downsampling-refactoring.plan.md`, which this plan depends on).
- No backend abstraction — matplotlib is hardwired throughout.
- No path to interactive web output (Plotly, Vega-Lite) or marimo / voilà dashboards
  without a full rewrite.
- `analysis/__init__.py` exports 80+ names because plotting is mixed in with
  loaders, config, and metrics (REVIEW.md issue #9).

---

## Goals

1. **Backend abstraction** — swap matplotlib ↔ plotly at call-site without
   changing the rest of the code.
2. **Feature 1** — `plot_subplots()`: N fields as stacked subplots sharing a time axis.
3. **Feature 2** — `plot_overlay()`: N fields on one axes, optional normalization
   with max value shown in legend.
4. **Feature 3** — clean `AnnotationManager` API usable on any `Axes` or Plotly figure.
5. **Feature 4** — downsampling toggle via `DownsampleConfig` (from
   `downsampling-refactoring.plan.md`).
6. **JS frontend path** — `backend.to_json()` serialises a Plotly figure to JSON
   that a JS frontend can render with `plotly.js` without duplicating any Python logic.
7. **Dashboard path** — Plotly backend works natively in marimo (`mo.ui.plotly()`),
   voilà, and voici without extra adaptation.
8. **File output** — both backends expose `save(fig, path, dpi)`.
9. **Feature 5 — unit selection and cross-source consistency** — each plotting
   function accepts a `display_units: dict[str, str] | None` mapping field names to
   target pint-parseable unit strings.  Stored units are read from `df.attrs["units"]`
   (populated by `getData()`), values are converted via the package `ureg`, and labels
   reflect the chosen unit.  Rules per plot type:
   - `plot_subplots()` — unit goes in the **ylabel** of each individual subplot.
   - `plot_overlay()` — if all curves share the **same display unit**, set ylabel
     `[unit]` (e.g. `[T]`); if units differ, **no ylabel** and each legend entry
     carries the unit instead: `"Field_B [T]"`, `"Courant_GR1 [A]"`.
   A dimension-consistency check warns (does not raise) when fields from different
   sources have incompatible physical dimensions, and suggests `normalize=True`.
10. *(likely)* **Dynamic resampling path** — a `"plotly-resampler"` backend wraps the
   Plotly figure in `FigureResampler` / `FigureWidgetResampler`, enabling view-dependent
   on-the-fly aggregation in Jupyter, voilà, marimo, and Dash apps.  Pre-computed
   `DownsampleConfig` is skipped in this mode (redundant).  Requires a live Python
   kernel — not available for static export or the REST/JS path.

---

## Target design

### New subpackage: `python_magnetrun/plotting/`

```
python_magnetrun/plotting/
├── __init__.py                    # public re-exports (PlotStyle, PlotColors, get_backend,
│                                  #   plot_subplots, plot_overlay, AnnotationManager)
├── backend.py                     # PlottingBackend Protocol + get_backend(name) factory
├── matplotlib_backend.py          # MatplotlibBackend
├── plotly_backend.py              # PlotlyBackend (static + to_json)
├── plotly_resampler_backend.py    # (likely) PlotlyResamplerBackend — live kernel only
├── style.py                       # PlotStyle, PlotColors (moved from analysis/plotting.py)
├── timeseries.py                  # plot_subplots(), plot_overlay()
└── annotations.py                 # AnnotationManager (refactored from analysis/plotting.py)
```

`analysis/plotting.py`, `utils/plots.py`, and `hybrid/plotting.py` become thin
compatibility wrappers that import from `python_magnetrun.plotting` — no public
API breakage.

---

### `plotting/backend.py` — the protocol

```python
from typing import Any, Protocol
from pathlib import Path

class PlottingBackend(Protocol):
    """Minimal contract every backend must satisfy."""

    def subplots(
        self,
        n: int,
        *,
        share_x: bool = True,
        style: "PlotStyle | None" = None,
    ) -> Any:
        """Return an opaque figure handle with n sub-axes."""
        ...

    def add_series(
        self,
        fig: Any,
        ax_idx: int,
        t: "np.ndarray",
        y: "np.ndarray",
        label: str,
        *,
        normalize: bool = False,
        color: str | None = None,
    ) -> None:
        """Plot (t, y) on the ax_idx-th axes.
        When normalize=True the series is divided by its max and the legend
        label is amended with '(max=<value> <unit>)'."""
        ...

    def add_annotation(
        self,
        fig: Any,
        ax_idx: int,
        t: float,
        label: str,
        detail: "dict | None" = None,
    ) -> None:
        """Add a clickable annotation at time t on the ax_idx-th axes."""
        ...

    def save(self, fig: Any, path: Path, *, dpi: int = 300) -> None: ...
    def show(self, fig: Any) -> None: ...
    def to_json(self, fig: Any) -> str:
        """Serialise to a self-contained JSON string (Plotly spec or Vega-Lite spec).
        Used by REST API endpoints and JS frontends."""
        ...


def get_backend(name: str = "matplotlib") -> PlottingBackend:
    """Factory.  name ∈ {'matplotlib', 'plotly', 'plotly-resampler'}.

    'plotly-resampler' requires a live Python kernel (Jupyter / voilà / marimo / Dash).
    It wraps the figure in FigureResampler for on-the-fly view-dependent aggregation.
    Use 'plotly' instead for static export or the REST/JS path.
    """
    if name == "plotly":
        from .plotly_backend import PlotlyBackend
        return PlotlyBackend()
    if name == "plotly-resampler":          # likely feature
        from .plotly_resampler_backend import PlotlyResamplerBackend
        return PlotlyResamplerBackend()
    from .matplotlib_backend import MatplotlibBackend
    return MatplotlibBackend()
```

`MatplotlibBackend.to_json()` raises `NotImplementedError` with a helpful message
directing the user to switch to `plotly`.

---

### `plotting/timeseries.py` — the two core functions

```python
def plot_subplots(
    data: pd.DataFrame,
    fields: list[str],
    t_col: str = "t",
    *,
    display_units: "dict[str, str] | None" = None,
    downsample: "DownsampleConfig | None" = None,
    backend: "str | PlottingBackend" = "matplotlib",
    style: "PlotStyle | None" = None,
) -> Any:
    """One subplot per field, all sharing the same x (time) axis.

    Parameters
    ----------
    display_units
        Optional mapping of field name → target unit string (pint-parseable),
        e.g. ``{"Champ_magn": "tesla"}``.  Stored units are read from
        ``data.attrs["units"]`` and converted before plotting.  The target unit
        appears in the ylabel of each subplot.  Fields absent from this dict are
        plotted in their stored unit.
    downsample
        If given, applied to each field before plotting via
        ``downsample_dataframe()`` from ``utils.downsampling``.
    backend
        Either a backend name string or a pre-constructed ``PlottingBackend``.
    """

def plot_overlay(
    data: pd.DataFrame,
    fields: list[str],
    t_col: str = "t",
    *,
    display_units: "dict[str, str] | None" = None,
    normalize: bool = False,
    downsample: "DownsampleConfig | None" = None,
    backend: "str | PlottingBackend" = "matplotlib",
    style: "PlotStyle | None" = None,
) -> Any:
    """All fields on one axes.

    Unit handling
    -------------
    If all curves resolve to the **same display unit**, a shared ylabel ``[unit]``
    is set (e.g. ``[T]``).  If units differ across curves, no ylabel is set and
    the unit is embedded in each legend entry instead:
    ``"Field_B [T]"``, ``"Courant_GR1 [A]"``.
    ``display_units`` overrides the stored unit for any listed field.
    A warning is emitted when fields have incompatible physical dimensions
    (e.g. tesla vs ampere); ``normalize=True`` is suggested as a workaround.

    Normalization
    -------------
    When ``normalize=True``:
    - each series is divided by its absolute maximum before plotting;
    - the legend entry becomes ``"Field_B [T]  (max = 12.3)"``.
    ``normalize`` and ``display_units`` are independent and compose.
    """
```

Both functions return the opaque figure handle from the backend so the caller can
call `backend.save(fig, path)`, `backend.show(fig)`, or `backend.to_json(fig)`.

---

### `plotting/annotations.py` — `AnnotationManager`

Refactored from the procedural code in `analysis/plotting.py:458–606`.

```python
class AnnotationManager:
    def __init__(self, backend: PlottingBackend): ...

    def add(
        self,
        fig: Any,
        ax_idx: int,
        t: float,
        label: str,
        detail: dict | None = None,
    ) -> None:
        """Add one annotation; delegates to backend.add_annotation()."""

    def connect(self, fig: Any) -> None:
        """Wire up interactive pick events (matplotlib) or click callbacks (plotly)."""
```

The matplotlib implementation preserves the existing interactive pick-event / detail
subplot behaviour.  The plotly implementation uses `customdata` + `clickData` callbacks.

---

## Steps

### Step 1 — Create `plotting/style.py` *(done)*

- `PlotStyle`, `PlotColors`, `PlotConfig` moved from `analysis/plotting.py`. ✓
- `load_plot_config()`, `save_plot_config()`, `DEFAULT_STYLE`, `DEFAULT_COLORS` exported. ✓
- `plotting/cli.py` added: `magnetrun-plot-config` entry point with `init` / `show` /
  `validate` subcommands. ✓

**Validate:**
```
pytest tests/plotting/test_backend.py -k style   # PlotStyle reaches backend
magnetrun-plot-config show                        # no import error
```

### Step 2 — Create `plotting/backend.py` *(done)*

- `PlottingBackend` Protocol with all six methods (`subplots`, `add_series`,
  `add_annotation`, `save`, `show`, `to_json`). ✓
- `get_backend(name)` factory supporting `"matplotlib"`, `"plotly"`,
  `"plotly-resampler"`, `"plotly-widget"`. ✓
- `MatplotlibBackend` — all six methods implemented; `to_json()` raises
  `NotImplementedError`. ✓
- `PlotlyBackend` — all six methods implemented including `to_json()`. ✓
- `PlotlyResamplerBackend` — all six methods implemented; `save()` and `to_json()`
  raise `NotImplementedError` as designed (live-kernel only). ✓

**Validate:**
```
pytest tests/plotting/test_backend.py -v
```

### Step 3 — Create `plotting/timeseries.py` *(done)*

- `plot_overlay()` implemented with `normalize`, `downsample`, `backend`, `colors`,
  `display_units` parameters. ✓
- `plot_subplots()` implemented with `normalize`, `display_units`, `downsample`,
  `backend`, `colors` parameters. ✓
- `_resolve_backend()` and `_apply_downsample()` helpers present. ✓
- `_resolve_units()` helper implemented (pint conversion via application registry,
  graceful fallback when no stored pint unit). ✓
- `_check_dimension_consistency()` helper implemented (warns on mixed dims). ✓
- ylabel rules applied: subplot ylabel = `"field [symbol]"` per axes; overlay
  shared-unit ylabel = `"[symbol]"`; overlay mixed-units = no ylabel, unit in
  legend entry. ✓
- `add_series()` gains `ylabel: str | None = None` kwarg in all three backends. ✓
- `units=` parameter renamed to `display_units=`; existing test updated. ✓

**Validate:**
```
pytest tests/plotting/test_timeseries.py -v
```

### Step 3b — Unit handling in `timeseries.py` *(done)*

> Depends on Step 3 todos above.

**Unit metadata carrier — `df.attrs["units"]`**

`getData()` on `PandasMagnetData` and `TdmsMagnetData` should populate the DataFrame's
`attrs` dict before returning:

```python
# in getData() — both subclasses
df = ...  # build the DataFrame as today
df.attrs["units"] = {key: self.getUnitKey(key) for key in df.columns}
# getUnitKey returns (symbol: str, pint_unit: pint.Unit)
return df
```

This is a non-breaking addition — `df.attrs` is silently ignored by all pandas
operations and is not serialised to CSV/parquet unless explicitly requested.

> **Watch out — `df.attrs` propagation is not guaranteed.**
> `df.attrs` was introduced in pandas 0.25 and is available in the installed 2.2.3.
> However, many pandas operations (groupby, merge, resample, concat, most aggregations)
> silently **drop** `attrs` on the result.  After any such transformation, callers must
> re-attach the metadata manually.  In plotting functions, always read `data.attrs`
> immediately from the DataFrame passed in — never assume attrs survived a prior
> transform.  If `data.attrs.get("units", {})` returns an empty dict, fall back
> gracefully (no conversion, use the column name as the symbol).

**Unit resolution in plotting functions**

```python
# plotting/timeseries.py  (internal helper)
def _resolve_units(
    data: pd.DataFrame,
    fields: list[str],
    display_units: dict[str, str] | None,
    ureg: pint.UnitRegistry,
) -> dict[str, tuple[np.ndarray, str]]:
    """Return {field: (converted_values, display_symbol)} for each field."""
    stored: dict[str, tuple[str, pint.Unit]] = data.attrs.get("units", {})
    result = {}
    for field in fields:
        y = data[field].to_numpy()
        symbol, src_unit = stored.get(field, (field, None))
        if display_units and field in display_units and src_unit is not None:
            tgt = ureg.parse_expression(display_units[field])
            y = (y * src_unit).to(tgt).magnitude
            symbol = f"{tgt:~P}"          # compact pint symbol, e.g. "T", "A"
        elif src_unit is not None:
            symbol = f"{src_unit:~P}"
        result[field] = (y, symbol)
    return result
```

**Dimension-consistency check (overlay only)**

```python
def _check_dimension_consistency(
    fields: list[str],
    stored_units: dict[str, tuple[str, pint.Unit]],
) -> None:
    dims = {f: stored_units[f][1].dimensionality
            for f in fields if f in stored_units and stored_units[f][1] is not None}
    unique_dims = set(str(d) for d in dims.values())
    if len(unique_dims) > 1:
        warnings.warn(
            f"plot_overlay: fields have incompatible dimensions {dims}. "
            "Consider normalize=True to compare shapes.",
            stacklevel=3,
        )
```

**Label rules**

| Plot type | ylabel | legend entry |
|-----------|--------|--------------|
| `plot_subplots` | `"field [unit]"` per subplot | `field` only |
| `plot_overlay` — all same unit | `"[unit]"` shared | `field` only |
| `plot_overlay` — mixed units | *none* | `"field [unit]"` |
| `plot_overlay` + `normalize` — all same unit | `"[unit]"` shared | `"field  (max = X.XX)"` |
| `plot_overlay` + `normalize` — mixed units | *none* | `"field [unit]  (max = X.XX)"` |

**`add_series()` protocol addition**

The `PlottingBackend.add_series()` signature gains a `ylabel` keyword used only
by `plot_subplots`; it is ignored by `plot_overlay`:

```python
def add_series(self, fig, ax_idx, t, y, label, *,
               normalize=False, color=None, ylabel: str | None = None) -> None: ...
```

**Validate (Step 3b):**
```python
# Quick smoke test — df.attrs["units"] is populated after getData()
from python_magnetrun import MagnetRun
mr = MagnetRun.fromtxt("data/some_file.txt")
df = mr.getMData().getData("Champ_magn", "t")
assert "units" in df.attrs, "attrs missing"
assert "Champ_magn" in df.attrs["units"], "unit not stored"
```
```
# Regression: timeseries tests must still pass (attrs don't break anything)
pytest tests/plotting/test_timeseries.py -v
```

### Step 4 — Create `plotting/annotations.py` *(done)*

- `AnnotationManager` class with `add()` and `connect()` methods. ✓
- Matplotlib path: pick-event handler, detail sub-figures, `open_figures` tracking. ✓
- Plotly path: `backend.add_annotation()` delegation. ✓
- **Todo:** update `analysis/plotting.py:plot_data()` to use `AnnotationManager`
  instead of the current inline procedural code (behaviour change is zero, purely
  structural).

**Validate:**
```
pytest tests/plotting/test_annotations.py -v
```

### Step 5 — Implement `PlotlyBackend` fully *(done)*

- All six protocol methods implemented. ✓
- `to_json()` returns `fig.to_json()`. ✓

**Validate:**
```
pytest tests/plotting/test_backend.py::TestPlotlyBackend -v
# Confirm to_json output is valid Plotly JSON:
python -c "
import json, numpy as np, pandas as pd
from python_magnetrun.plotting import get_backend, plot_overlay
df = pd.DataFrame({'t': np.linspace(0,1,100), 'B': np.ones(100)})
b = get_backend('plotly')
fig = plot_overlay(df, ['B'], backend=b)
spec = json.loads(b.to_json(fig))
assert 'data' in spec and 'layout' in spec
print('OK')
"
```

### Step 5b — *(likely)* Implement `PlotlyResamplerBackend` *(done)*

> `save()` and `to_json()` intentionally raise `NotImplementedError` (live-kernel only).

`plotly-resampler` (by the same authors as `tsdownsample`) wraps a Plotly figure so
that every pan/zoom triggers a new aggregation pass in Python, keeping ~1 000 visible
points regardless of dataset size.  It uses `tsdownsample` (MinMaxLTTB by default)
internally, so no separate downsampling step is needed.

Three-tier downsampling strategy:

```
Tier 1 — data loading    →  DownsampleConfig (any backend, static-safe)
Tier 2 — plot creation   →  DownsampleConfig pre-computed (static / REST / matplotlib)
Tier 3 — user interaction →  FigureResampler  (live kernel: Jupyter / voilà / marimo / Dash)
```

When `"plotly-resampler"` is selected, `plot_subplots()` / `plot_overlay()` **raise
`ValueError`** if a `DownsampleConfig` is also passed — tier 3 handles resampling
dynamically and must receive the full-resolution data.  Passing pre-downsampled data
would defeat the view-dependent aggregation and is therefore an error, not a silent
skip.

**Implementation sketch:**

```python
# plotting/plotly_resampler_backend.py
from plotly_resampler import FigureResampler, FigureWidgetResampler

class PlotlyResamplerBackend:
    def __init__(self, *, widget: bool = False):
        # widget=True  → FigureWidgetResampler  (Jupyter / marimo ipywidget)
        # widget=False → FigureResampler        (Dash / show_dash())
        self._cls = FigureWidgetResampler if widget else FigureResampler

    def subplots(self, n, *, share_x=True, style=None):
        from plotly.subplots import make_subplots
        base = make_subplots(rows=n, shared_xaxes=share_x)
        return self._cls(base)

    def add_series(self, fig, ax_idx, t, y, label, *, normalize=False, color=None):
        import plotly.graph_objects as go
        if normalize:
            max_val = float(np.abs(y).max())
            y = y / max_val
            label = f"{label}  (max = {max_val:.3g})"
        fig.add_trace(go.Scatter(x=t, y=y, name=label), row=ax_idx + 1, col=1)

    def save(self, fig, path, *, dpi=300):
        raise NotImplementedError(
            "PlotlyResamplerBackend requires a live kernel — use PlotlyBackend for static export."
        )

    def show(self, fig):
        fig.show_dash()   # or display(fig) in a Jupyter context

    def to_json(self, fig):
        raise NotImplementedError(
            "PlotlyResamplerBackend requires a live kernel — use PlotlyBackend for the REST/JS path."
        )
```

`get_backend("plotly-resampler")` → `PlotlyResamplerBackend(widget=False)`
`get_backend("plotly-widget")` → `PlotlyResamplerBackend(widget=True)` *(marimo / Jupyter)*

**Capability matrix:**

| Backend | Static export | REST / JS | Jupyter | voilà | marimo | Dash |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|
| `matplotlib` | ✓ | — | ✓ | ✓ | ✓ | — |
| `plotly` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `plotly-resampler` | — | — | ✓ | ✓ | ✓ | ✓ |
| `plotly-widget` | — | — | ✓ | ✓ | ✓ | — |

**Validate (Step 5b):**
```
pytest tests/plotting/test_backend.py::TestPlotlyResamplerBackend -v
# save and to_json must raise, show must not raise when library present
```

### Step 6 — Wire up `commands/plot.py` *(done)*

- `--backend` with full choices list in `cli_args.py`. ✓
- `--json` flag in `cli_args.py`. ✓
- `--normalize` flag (pre-existing). ✓
- `plot_vs_time()` routes to `plot_overlay()` for non-matplotlib backends. ✓
- `--overlay` / `--subplots` mutually-exclusive flags added to `create_common_plot_parser()`. ✓
- `--unit FIELD=UNIT` (repeatable) added; `_parse_display_units()` builds the dict. ✓
- `plot_subplots()` wired for `--subplots` mode (matplotlib and non-matplotlib). ✓

**Validate:**
```bash
magnetrun plot --help           # shows --backend, --json, --normalize, --overlay, --subplots, --unit
magnetrun plot --backend plotly --subplots --save /tmp/smoke.html data/some_file.txt Champ_magn
test -f /tmp/smoke.html && echo OK
magnetrun plot --backend plotly --json data/some_file.txt Champ_magn   # prints JSON to stdout
```

### Step 6b — Consolidate `--save` / `--show` and output path logic *(done)*

#### Current state (problems to fix)

Three separate save sites exist in `commands/plot.py`, each with slightly different
logic:

| Site | Line | Path strategy | Format |
|------|------|--------------|--------|
| `_plot_vs_time_backend()` | 295–300 | `stem` from last input file path | `.html` or `.png` |
| `plot_vs_time()` matplotlib | 418–425 | `file` variable from loop (last file) | `.png` hardcoded |
| `plot_key_vs_key()` | 502–507 | `file` variable from loop (last file) | `.png` hardcoded |
| `plot_bkpts()` | 178–179 | `file` variable (first file) | `.png` hardcoded |

**Consistency problem:** the default save path is derived from the *input file's
path*, not CWD.  If data lives in a read-only directory (e.g. `/srv/data/`), saving
silently fails or raises a permission error.  CWD is the conventional default for CLI
tools and is always writable.

**`set_defaults` bug:** `create_managed_plots_parser()` calls
`parser.set_defaults(show=True)`.  This runs unconditionally, so `args.show` is
`True` even when `--save` is passed.  The matplotlib path works around it by checking
`if not args.save` instead of `if args.show`, but the non-matplotlib path checks
`if args.save` / `else args.show` consistently, masking the issue.

#### Changes

**`cli_args.py` — `create_managed_plots_parser()`:**

```python
def create_managed_plots_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--save",
        nargs="?",          # 0 or 1 value
        const="",           # --save (no filename) → args.save = ""
        default=None,       # flag absent         → args.save = None
        metavar="FILE",
        help=(
            "save figure to FILE (.png/.html chosen by backend); "
            "omit FILE for an auto-generated name in the current directory"
        ),
    )
    group.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="display figure interactively (default when neither flag is given)",
    )
    # No set_defaults — default-to-show handled once in command handlers.
    return parser
```

`args.save` semantics after the change:

| Command | `args.save` | `args.show` | Action |
|---------|-------------|-------------|--------|
| *(neither)* | `None` | `False` | show (handler default) |
| `--show` | `None` | `True` | show |
| `--save` | `""` | `False` | save with auto name |
| `--save myplot.png` | `"myplot.png"` | `False` | save to that path |

**`commands/plot.py` — shared helper:**

```python
def _default_save_path(
    input_files: list[str],
    fields: list[str],
    backend_name: str,
) -> Path:
    """Build a default output filename in CWD (never in the input-file directory).

    Example: CWD/M9_Overview_260331-Field_Champ_magn_vs_time.png
    """
    stem = Path(input_files[-1]).stem if input_files else "output"
    suffix = ".html" if "plotly" in backend_name else ".png"
    tag = "_".join(fields[:2]) if fields else "plot"
    return Path.cwd() / f"{stem}-{tag}_vs_time{suffix}"
```

**`commands/plot.py` — uniform save/show block** (replaces all four current sites):

```python
def _handle_output(fig, args, backend, input_files, fields, backend_name):
    """Unified save-or-show logic for all plot methods."""
    save = getattr(args, "save", None)
    show = getattr(args, "show", False)
    if save is None and not show:
        show = True                    # default: interactive display
    if save is not None:
        path = Path(save) if save else _default_save_path(input_files, fields, backend_name)
        backend.save(fig, path, dpi=backend.style_dpi)
        logger.info(f"saved to {path}")
    if show:
        backend.show(fig)
```

All four current save sites (`_plot_vs_time_backend`, `plot_vs_time`,
`plot_key_vs_key`, `plot_bkpts`) are replaced by a single call to
`_handle_output(...)`.

#### Output path rules

| Scenario | Resulting path |
|----------|---------------|
| `--save` (no filename) | `CWD/<input_stem>-<field1>_<field2>_vs_time.<ext>` |
| `--save myplot.png` | `myplot.png` (relative to CWD, as typed) |
| `--save /abs/path/fig.html` | `/abs/path/fig.html` |

The extension in the auto-generated name is determined by the backend:
`.html` for `plotly*`, `.png` otherwise.  This aligns with the existing behaviour in
`_plot_vs_time_backend()` (line 298) and makes it consistent across all methods.

**Validate (Step 6b):**
```
pytest tests/plotting/test_save_show.py -v   # once written (Step 10 todo)
# Manual smoke — --save with no filename writes to CWD, not input dir:
cd /tmp && magnetrun plot --save data/some_file.txt Champ_magn
ls -1 *.png   # must appear in /tmp, not in data/
# --save FILE and --show must be mutually exclusive:
magnetrun plot --save out.png --show data/some_file.txt Champ_magn 2>&1 | grep -i "mutually exclusive"
```

### Step 3c — Label & legend utilities *(todo — see `label-legend-uniformization.plan.md`)*

> Sub-plan: `prompts/label-legend-uniformization.plan.md`
> Execution order within this step: **3c-fix → 3c-meta → 3c**.

**Step 3c-fix** *(independent — fix now, no dependency on 3c-meta or 3c)*:

Four bugs in the **already-done** Step 6 code (`commands/plot.py`):

- **Fix 1** (`plot_vs_time` lines 644–646) — `normalize` branch is a second `if`, not `elif`:
  overwrites the unit ylabel set on line 644.  Change to `elif args.normalize:`.
- **Fix 2** (`plot_vs_time` lines 650–653) — hardcoded `plt.xlabel("t [s]")` fallback;
  always try `getUnitKey("t")` first, fall back on exception.
- **Fix 3** (`plot_vs_time`, `plot_key_vs_key`) — `plt.ylabel` / `plt.xlabel` instead of
  `my_ax.set_ylabel` / `my_ax.set_xlabel`; `plt.legend` instead of
  `my_ax.legend(loc=cfg.style.legend_loc)`.
- **Fix 4** (`plot_key_vs_key` lines 698–705) — `legends` contains filenames only
  (e.g. `"M9_260331"`); should be `"M9_260331: key1 vs key2"`.

**Validate (3c-fix):**
```bash
# Fix 1: normalize no longer overwrites unit ylabel — inspect saved PNG
magnetrun plot --normalize --save /tmp/fix1.png data/some_file.txt Champ_magn
# ylabel must be "normalized", not "[T]" AND not the empty ylabel from the double-if bug
# Fix 2/3: axes-level API — no plt.* warnings in stderr
magnetrun plot --save /tmp/fix3.png data/some_file.txt Champ_magn 2>&1 | grep -i "UserWarning\|plt\."
# Fix 4: legend contains field pair
magnetrun plot --save /tmp/fix4.png data/some_file.txt Champ_magn-I_GR1 2>&1
# Open /tmp/fix4.png and confirm legend reads "stem: Champ_magn vs I_GR1"
```

**Step 3c-meta** *(prerequisite to 3c — no new files)*:
- `field_defs.py`: `add_field_def`/`update_field_def`/`list_field_defs` gain `label=`;
  CLI `--label` on `add`/`update`.  `field_defs.py` is the single source of truth for
  the JSON schema — no separate `field_meta.py`.
- `magnetdata_base.py`: `FieldMeta` NamedTuple (4 fields: symbol, unit, label,
  description) defined here; `field_meta` dict; `getFieldMeta()`; `load_units_from_json`
  stores `FieldMeta`; fix `addData`/`computeData` type annotation and add `label=`/
  `description=` kwargs.
- `magnetdata_tdms.py`: fix Bug — `addData` never updated `self.units`; populate `field_meta`.
- `hybrid/hybrid_data.py`: fix Bug — `load_units_from_json` skips all entries due to
  `kHz/`/`rms/`/`trigger/` prefix mismatch; override with prefix-aware version; add
  `field_meta` dict, `getFieldMeta()`, and lazy `addData`.
- `hybrid/hybrid_run.py`: `getUnitKey()` already present on `HybridData` (line 935);
  add alias on `HybridRun` delegating to `self.HybridData.getUnitKey()`.
- JSON editorial: add optional `"label"` field to `pigbrother-defs.json` and
  `hybrid-defs.json` for frequently-plotted fields.

**Validate (3c-meta):**
```python
# FieldMeta round-trip via load_units_from_json
from python_magnetrun.magnetdata_base import FieldMeta
from python_magnetrun import MagnetRun
mr = MagnetRun.fromtxt("data/some_file.txt")
mdata = mr.getMData()
meta = mdata.getFieldMeta("Champ_magn")
assert isinstance(meta, FieldMeta)
assert meta.symbol is not None
assert meta.unit is not None
print(meta)   # FieldMeta(symbol='B', unit=<Unit('tesla')>, label=..., description=...)
```
```python
# Hybrid prefix fix — self.units must not be empty after loading
from python_magnetrun.hybrid import HybridRun
hr = HybridRun(...)
assert len(hr.HybridData.units) > 0, "prefix mismatch still active"
```
```bash
# field_defs CLI: --label roundtrips through JSON
magnetrun-field-defs add test_key T tesla --label "B_{test}" pigbrother-defs.json
magnetrun-field-defs list pigbrother-defs.json | grep test_key   # must show B_{test}
magnetrun-field-defs update test_key --label "" pigbrother-defs.json  # clear label
```

**Step 3c** *(depends on 3c-meta)*:
- Add `python_magnetrun/plotting/utils.py` with `format_axis_label(symbol, unit)`,
  `format_legend_label(key, basename, unit, max_val)`, and
  `resolve_legend_labels(fields, field_metas, aliases)` with symbol-clash auto-detection.

**Validate (3c):**
```python
from python_magnetrun.plotting.utils import (
    format_axis_label, format_legend_label, resolve_legend_labels,
)
import pint; ureg = pint.UnitRegistry()
assert format_axis_label("B", ureg.tesla) == "B [T]"
assert format_axis_label("x", None) == "x"
assert format_legend_label("I", basename="M9") == "M9: I"
assert format_legend_label("I", unit=ureg.ampere) == "I [A]"
assert format_legend_label("I", unit=ureg.ampere, max_val=1234.5) == "I [A]  (max = 1.23e+03 [A])"
# clash resolution: two fields sharing symbol "I" → I_GR1, I_GR2
from python_magnetrun.magnetdata_base import FieldMeta
metas = {
    "Courant_GR1": FieldMeta("I", ureg.ampere),
    "Courant_GR2": FieldMeta("I", ureg.ampere),
}
labels = resolve_legend_labels(["Courant_GR1", "Courant_GR2"], metas)
assert labels["Courant_GR1"] != labels["Courant_GR2"]
print("all assertions passed")
```

### Step 7 — Update `analysis/plotting.py` and `utils/plots.py` *(todo)*

- `analysis/plotting.py`: import `PlotStyle`, `PlotColors` from `plotting.style`;
  import `AnnotationManager` from `plotting.annotations`.  Remove duplicated definitions.
- Apply label/legend fixes from `label-legend-uniformization.plan.md` (time axis unit,
  y-axis unit, consistent `fontsize=style.label_fontsize`).
- `utils/plots.py`: replace `plt.subplots()` calls with `get_backend("matplotlib").subplots()`.
- Apply label/legend fixes: replace `plt.ylabel/xlabel` with `ax.set_*`; add optional
  label params for unit-aware callers.

**Validate:**
```bash
# No duplicate PlotStyle/PlotColors definitions — grep must find exactly one definition site
grep -rn "class PlotStyle\|class PlotColors" python_magnetrun/ | grep -v "__pycache__"
# Must print only plotting/style.py lines.

# No regressions in existing analysis tests
pytest tests/ -v --ignore=tests/plotting
```

### Step 8 — Update `hybrid/plotting.py` *(todo)*

- Replace the local downsampling call with `downsample_arrays()` from
  `utils.downsampling` (now available — `utils/downsampling.py` merged in `6d2e09b`).
- Use `get_backend(backend_name)` so hybrid plots can also be rendered via plotly.

**Validate:**
```bash
# No local downsampling logic remaining in hybrid/plotting.py
grep -n "stride\|lttb\|minmax\|resample" python_magnetrun/hybrid/plotting.py
# Must be empty (all downsampling delegated to utils.downsampling).

# Smoke test with a real hybrid file — both backends must produce output
python -c "
from python_magnetrun.hybrid import HybridRun
hr = HybridRun(...)
fig = hr.plot(['kHz/FEPC-AUX-LNCMI/ALIM1_J1'], backend='matplotlib')
fig2 = hr.plot(['kHz/FEPC-AUX-LNCMI/ALIM1_J1'], backend='plotly')
print('both backends OK')
"
```

### Step 9 — `pyproject.toml` extras *(todo)*

```toml
[project.optional-dependencies]
plotting  = ["plotly>=5.0", "kaleido>=0.2"]          # static image export
resampler = ["plotly-resampler>=0.9"]                # (likely) live-kernel dynamic resampling
                                                     # plotly-resampler pulls tsdownsample
hybrid    = ["tsdownsample>=1.0"]                    # already in downsampling plan
```

Note: `plotly-resampler` depends on `tsdownsample` internally, so installing the
`resampler` extras group implicitly satisfies the `hybrid` requirement as well.
Document in module docstrings that `plotly`, `kaleido`, and `plotly-resampler` are
soft requirements guarded by `try/except ImportError`.

**Validate:**
```bash
pip install -e ".[plotting]"
python -c "import plotly, kaleido; print('plotting extras OK')"

pip install -e ".[resampler]"
python -c "import plotly_resampler, tsdownsample; print('resampler extras OK')"

# Core install without extras must not import plotly at module level
pip install -e .
python -c "
import sys
import python_magnetrun
assert 'plotly' not in sys.modules, 'plotly imported at startup without extras'
print('no eager plotly import — OK')
"
```

### Step 10 — Tests *(partial)*

#### `tests/plotting/test_backend.py` *(done)*

- `TestGetBackend` — default, explicit matplotlib, plotly, plotly-resampler, plotly-widget. ✓
- `TestMatplotlibBackend` — subplots count, add_series, normalize, to_json raises, save to file. ✓
- `TestPlotlyBackend` — subplots, add_series (2 traces), to_json valid JSON. ✓
- `TestPlotlyResamplerBackend` — ImportError when not installed, save raises, to_json raises. ✓

#### `tests/plotting/test_timeseries.py` *(partial)*

- `TestPlotSubplots` — figure type, axes count, one line per subplot, plotly trace count,
  empty-fields raises, missing-field skipped, downsample, title. ✓
- `TestPlotOverlay` — single axes, multiple series, normalize scales to 1, normalize label
  contains max, units in label, no-normalize preserves values, plotly trace count,
  empty-fields raises. ✓
- **Todo:** tests for `display_units` pint conversion (gauss → tesla values correct). 
- **Todo:** tests for `_check_dimension_consistency` warning on mixed dimensions.
- **Todo:** tests for ylabel rules (subplot ylabel contains unit, overlay shared-unit
  ylabel, overlay mixed-units no ylabel).
- **Todo:** tests for `normalize` on `plot_subplots`.

#### `tests/plotting/test_annotations.py` *(done)*

- `TestAnnotationManagerMatplotlib` — add stores detail, connect is noop without
  annotations, connect wires pick event. ✓
- `TestAnnotationManagerPlotly` — add creates shape/annotation, connect is noop. ✓

#### `tests/plotting/test_units.py` *(todo — new file)*

- `test_resolve_units_no_display` — values unchanged, symbol from stored unit.
- `test_resolve_units_with_conversion` — gauss → tesla magnitude correct to 4 s.f.
- `test_resolve_units_missing_attrs` — graceful fallback when `df.attrs["units"]` absent.
- `test_dimension_consistency_warning` — mixed dims emit `UserWarning`.
- `test_dimension_consistency_same_dim` — no warning for matching dimensions.
- `test_overlay_shared_ylabel` — ylabel set to `"[T]"` when all fields same unit.
- `test_overlay_no_ylabel_mixed` — no ylabel when units differ.
- `test_subplots_ylabel_per_ax` — each subplot ylabel contains its unit symbol.

#### *(likely)* `tests/plotting/test_resampler_backend.py` *(todo — new file)*

- `test_get_backend_resampler` — skipped if `plotly-resampler` not installed.
- `test_resampler_subplots` — figure is `FigureResampler` instance, correct row count.
- `test_resampler_add_series` — trace count matches fields length.
- `test_resampler_save_raises` / `test_resampler_to_json_raises`. ✓ (covered in `test_backend.py`)

#### `tests/plotting/test_save_show.py` *(todo — new file, for Step 6b)*

- `test_default_save_path_in_cwd` — `_default_save_path()` returns path under `Path.cwd()`.
- `test_default_save_path_png_for_matplotlib` — suffix is `.png`.
- `test_default_save_path_html_for_plotly` — suffix is `.html`.
- `test_save_arg_none_and_no_show_defaults_to_show` — `_handle_output` calls `backend.show`.
- `test_save_arg_empty_string_uses_default_name` — `_handle_output` calls `backend.save` with auto path.
- `test_save_arg_explicit_filename` — `_handle_output` calls `backend.save` with given path.
- `test_show_arg_true_calls_show` — `_handle_output` calls `backend.show`.
- `test_save_and_show_mutually_exclusive` — argparse rejects `--save --show`.

### Step 11 — Migrate examples *(todo — after Steps 3, 3b, 6, 6b are complete)*

All plotting examples currently use `plt.*` directly.  This step replaces scattered
matplotlib calls with `plot_subplots()` / `plot_overlay()` from
`python_magnetrun.plotting` and standardises the save/show pattern from Step 6b.

#### Audit of example files

| File | Current API | Migration needed |
|------|-------------|-----------------|
| `plot_vprocess.py` | Already uses `plot_overlay()`, `plot_subplots()`, `get_backend()` — **mixed** | Remove remaining `plt.*` calls; adopt `_handle_output()` pattern for `--save`/`--no_show` |
| `plot_hybrid_with_pupitre_tdms.py` | `plt.subplots`, `plt.plot`, `plt.savefig`, `plt.show`; `args.save` / `args.show` | Replace with `plot_overlay()` or `plot_subplots()`; adopt `--save [FILE]` / `--show` |
| `plot_hybrid_minimal.py` | `plt.subplots`, `plt.plot`, `plt.savefig`, `plt.show`; boolean `save` flag | Same as above |
| `plot_rms.py` | `plt.subplots`, `plt.plot`, `plt.savefig`, `plt.show`; `args.output` (different name) | Replace; rename `--output` → `--save` for consistency |
| `plot_fepc_data.py` | `plt.subplots`, `plt.plot`, `plt.savefig`, `plt.show`; `output_file` local var | Replace; adopt `--save [FILE]` / `--show` |
| `plot_trigger_data.py` | `plt.subplots`, `plt.plot`, `plt.savefig`, `plt.show`; `args.save` / `--no_show` | Replace; normalise `--no_show` → `--show` (inverse logic) |
| `cmp_fields.py` | `plt.gca`, `plt.plot`, `plt.savefig`, `plt.show`; `mdata.plotData()`; `args.save` | Replace `plt.*` with `plot_overlay()`; `plotData()` can stay as data-layer call |
| `corr_Ih_Ib.py` | Same as `cmp_fields.py` | Same as above |
| `outliers.py` | `plt.subplot`, `plt.plot`, `plt.show`; `args.save` (conditional) | Replace with `plot_subplots()`; adopt unified save/show |
| `timeseries-anomaly-detection.py` | `plt.subplots`, `plt.plot`, `plt.scatter`, `plt.savefig`, `plt.show` in helper; no CLI flags | Replace `plot_anomalies()` helper with `plot_overlay()` + `AnnotationManager` |

#### Changes per example

**Common pattern for all files:**

```python
# Before (scattered)
fig, ax = plt.subplots()
ax.plot(t, y, label="Field")
if args.save:
    plt.savefig(f"{stem}.png")
else:
    plt.show()

# After (unified)
from python_magnetrun.plotting import plot_overlay, get_backend
fig = plot_overlay(df, ["Field"], backend=args.backend)
_handle_output(fig, args, get_backend(args.backend), input_files, ["Field"], args.backend)
```

**Flag normalisation:**
- `args.output` (`plot_rms.py`) → `--save [FILE]`
- `--no_show` (`plot_trigger_data.py`, `plot_vprocess.py`) → `--show` (inverse sense;
  new default is show-unless-save)
- bare boolean `save` (`plot_hybrid_minimal.py`) → `--save [FILE]`

**`timeseries-anomaly-detection.py`** is a slightly deeper change: its `plot_anomalies()`
helper builds annotations manually; replace with `plot_overlay()` +
`AnnotationManager.add()` + `AnnotationManager.connect()`.

**`cmp_fields.py` / `corr_Ih_Ib.py`:** `mdata.plotData()` is a thin wrapper that calls
`MagnetDataBase.plotData(x, y, ax, ...)`.  Keep it for now — it is a data-layer
method, not a plotting-module method.  The wrapping `plt.*` frame code is what gets
replaced.

**Validate (Step 11 — per example):**
```bash
# Each migrated example must run headless with --save and produce output in CWD
for script in examples/plot_vprocess.py examples/plot_rms.py examples/plot_fepc_data.py \
              examples/plot_trigger_data.py examples/outliers.py; do
    python "$script" --save /tmp/smoke_$(basename $script .py).png <required_args> \
      && echo "$script OK" || echo "$script FAILED"
done

# No plt.* calls remain in migrated examples
grep -rn "plt\." examples/ | grep -v "__pycache__\|#"
# Must be empty after full migration.

# Flag consistency: --no_show must not appear in any example
grep -rn "\-\-no.show\|no_show" examples/ | grep -v "__pycache__"
```

---

## File change summary

| File | Status | Notes |
|------|--------|-------|
| `python_magnetrun/plotting/__init__.py` | ✅ done | Full public API re-exports |
| `python_magnetrun/plotting/backend.py` | ✅ done | `PlottingBackend` protocol, `get_backend()` |
| `python_magnetrun/plotting/matplotlib_backend.py` | ✅ done | All six protocol methods |
| `python_magnetrun/plotting/plotly_backend.py` | ✅ done | All six methods, `to_json()` |
| `python_magnetrun/plotting/plotly_resampler_backend.py` | ✅ done *(likely)* | Live-kernel only; `save()`/`to_json()` raise |
| `python_magnetrun/plotting/style.py` | ✅ done | `PlotStyle`, `PlotColors`, `PlotConfig`, `load_plot_config` |
| `python_magnetrun/plotting/cli.py` | ✅ done | `magnetrun-plot-config` entry point |
| `python_magnetrun/plotting/timeseries.py` | ✅ done | `plot_subplots`/`plot_overlay` complete; `normalize`, `display_units`, `_resolve_units`, `_check_dimension_consistency`, ylabel rules all implemented |
| `python_magnetrun/plotting/annotations.py` | ✅ done | `AnnotationManager` with matplotlib pick-events and plotly path |
| `python_magnetrun/field_defs.py` | ⏳ todo (3c-meta) | `add_field_def`/`update_field_def`/`list_field_defs` gain `label=`; CLI `--label` |
| `python_magnetrun/plotting/utils.py` | ⏳ todo (3c) | new — `format_axis_label`, `format_legend_label`, `resolve_legend_labels`, `_extract_suffix` |
| `python_magnetrun/analysis/plotting.py` | ⏳ todo | Still defines its own `PlotStyle`/`PlotColors`; needs to import from `plotting.style` and use `AnnotationManager` |
| `python_magnetrun/utils/downsampling.py` | ✅ done | `DownsampleConfig`, `downsample_arrays`, `downsample_dataframe` (commit `6d2e09b`) |
| `python_magnetrun/magnetdata_base.py` | ⏳ todo (3c-meta) | `FieldMeta` NamedTuple; `field_meta` dict; `getFieldMeta()`; `addData`/`computeData` type fix + `label=`/`description=` |
| `python_magnetrun/magnetdata_pandas.py` | ⚠️ partial | `getData()` populates `df.attrs["units"]`; `addData`/`computeData` still need `field_meta` population (3c-meta) |
| `python_magnetrun/magnetdata_tdms.py` | ⚠️ partial | `getData()` populates `df.attrs["units"]`; `addData` Bug fix (never updated `self.units`) + `field_meta` pending (3c-meta) |
| `python_magnetrun/hybrid/hybrid_run.py` | ⚠️ partial | Downsampling done (commit `6d2e09b`); `getUnitKey()` alias pending (3c-meta) |
| `python_magnetrun/hybrid/hybrid_data.py` | ⏳ todo (3c-meta) | `field_meta` dict; prefix-aware `load_units_from_json` (Bug 2 fix); `getFieldMeta()`; `addData` lazy |
| `python_magnetrun/utils/plots.py` | ⏳ todo | Still uses raw `plt.subplots()` |
| `python_magnetrun/hybrid/plotting.py` | ⏳ todo | Uses local downsampling (not `downsample_arrays`); not backend-aware |
| `python_magnetrun/commands/plot.py` | ⚠️ partial | Step 6/6b done; Step 3c-fix pending (normalize/elif bug, plt.* → ax.set_*, legend content) |
| `python_magnetrun/cli_args.py` | ✅ done | `--backend`, `--json`, `--normalize`, `--overlay`/`--subplots` (mutex), `--unit` present; `--save` fixed; `set_defaults` bug fixed |
| `pyproject.toml` | ⏳ todo | `plotting` and `resampler` extras groups not yet added |
| `tests/plotting/test_backend.py` | ✅ done | All backends, factory, save, to_json |
| `tests/plotting/test_timeseries.py` | ⚠️ partial | Core cases done; unit conversion, ylabel rules, normalize-on-subplots missing |
| `tests/plotting/test_annotations.py` | ✅ done | Matplotlib pick-event and plotly path |
| `tests/plotting/test_units.py` | ⏳ todo | New file — pint conversion, dimension warning, ylabel rules |
| `tests/plotting/test_save_show.py` | ⏳ todo | New file — Step 6b `_handle_output`, `_default_save_path` (logic done; tests pending) |
| `examples/plot_vprocess.py` | ⚠️ partial | Already uses new API; remaining `plt.*` and `--no_show` to clean up |
| `examples/plot_hybrid_with_pupitre_tdms.py` | ⏳ todo | Replace `plt.*`; adopt `--save`/`--show` |
| `examples/plot_hybrid_minimal.py` | ⏳ todo | Replace `plt.*`; adopt `--save`/`--show` |
| `examples/plot_rms.py` | ⏳ todo | Replace `plt.*`; rename `--output` → `--save` |
| `examples/plot_fepc_data.py` | ⏳ todo | Replace `plt.*`; adopt `--save`/`--show` |
| `examples/plot_trigger_data.py` | ⏳ todo | Replace `plt.*`; invert `--no_show` → `--show` |
| `examples/cmp_fields.py` | ⏳ todo | Replace `plt.*` frame; keep `plotData()` |
| `examples/corr_Ih_Ib.py` | ⏳ todo | Replace `plt.*` frame; keep `plotData()` |
| `examples/outliers.py` | ⏳ todo | Replace `plt.subplot` with `plot_subplots()` |
| `examples/timeseries-anomaly-detection.py` | ⏳ todo | Replace `plot_anomalies()` with `plot_overlay()` + `AnnotationManager` |

---

## JS frontend integration

`to_json()` is the only seam between Python and a JS frontend.

```
Python REST endpoint (FastAPI / Flask)
  GET /api/plot?fields=B,I&t_start=…&t_end=…&backend=plotly
  ──► load data  (MagnetData / HybridRun)
  ──► DownsampleConfig(n_out=10_000)
  ──► plot_subplots(data, fields, downsample=cfg, backend="plotly")
  ──► return backend.to_json(fig)          # application/json

JS frontend
  fetch("/api/plot?…")
  .then(spec => Plotly.react("div-id", spec.data, spec.layout))
```

All scientific logic (downsampling, normalization, regime spans, annotations) stays
in Python.  The JS layer only calls `Plotly.react()`.  No duplication.

For **marimo** / **voilà** / **voici** dashboards, skip `to_json()` entirely:

```python
import marimo as mo
fig = plot_subplots(data, fields, backend="plotly")
mo.ui.plotly(fig)   # native marimo widget — interactive out of the box
```

*(likely)* For very large datasets in a dashboard context, use the resampler backend
instead — no pre-downsampling needed:

```python
# marimo / Jupyter — live kernel available
fig = plot_subplots(data, fields, backend="plotly-widget")   # FigureWidgetResampler
mo.ui.plotly(fig)

# Dash app
fig = plot_subplots(data, fields, backend="plotly-resampler")   # FigureResampler
fig.show_dash()
```

**Backend selection guide:**

| Goal | Backend |
|------|---------|
| Static PNG / SVG file | `matplotlib` |
| JSON for JS frontend / REST API | `plotly` + `DownsampleConfig` |
| Jupyter / marimo notebook | `plotly` or `plotly-widget` *(likely)* |
| voilà / voici dashboard | `plotly` or `plotly-resampler` *(likely)* |
| Dash app with huge datasets | `plotly-resampler` *(likely)* |

---

## Interaction with `REVIEW.md` overall plan

### Directly resolves

- **Issue #9** (`analysis/__init__.py` exports 80+ names flat) — `PlotStyle`,
  `PlotColors`, and plotting functions move to `python_magnetrun.plotting`;
  `analysis/__init__.py` can drop those re-exports, shrinking its namespace.

### Depends on

- **Downsampling refactoring plan** (`downsampling-refactoring.plan.md`) — **✅ done**
  (commit `6d2e09b`).  `utils/downsampling.py` now exports `DownsampleConfig`,
  `downsample_arrays`, and `downsample_dataframe`.  `hybrid/hybrid_run.py` has been
  updated to use these.  Steps 3 and 8 of this plan can proceed without any stride
  fallback — all methods (`minmax_lttb`, `lttb`, `minmax`, `stride`) are available.

### Enables / unblocks

- **Cross-domain comparison Phase D (`CHANNEL_ALIASES` + `KeyMapping`) and
  Phase E (`ComparisonSession`)** — `ComparisonSession` will need to plot data from
  heterogeneous sources on the same axes.  With the backend protocol in place,
  `ComparisonSession.plot()` can accept a `backend=` parameter and call
  `plot_overlay()` directly, passing pre-aligned DataFrames.  Without this plan,
  `ComparisonSession` would have to duplicate matplotlib calls.

- **`magnetrun-compare` CLI (Phase F)** — the `--backend` / `--json` flags in Step 6
  of this plan establish the CLI pattern that `magnetrun-compare` will follow.

### Does not conflict with

- Timestamp convention work (`hybriddata-timestamp-plan.md`) — orthogonal.
- `HybridData` timestamp support — orthogonal.
- Cross-domain comparison Phase A1–A3 (protocol compliance) — orthogonal.
- CLI consolidation (REVIEW.md issues #9, #10) — this plan moves plot logic out of
  `commands/plot.py` into `plotting/`, which makes issue #10 easier to address later,
  but does not depend on it.
- Housing config, `MagnetData` factory, `saveData` delegation — all done, unrelated.

---

## Recommended execution order

Steps 1–6b are **done** (commits up to `6d2e09b`).  Each step carries an inline
**Validate:** block; run it before marking the step complete.  Remaining work in
priority order:

1. **Step 3c-fix** *(~S, independent — safe to land now)*
   - Fix `normalize` double-`if` → `elif` in `plot_vs_time` (line 645).
   - Replace `plt.ylabel`/`plt.xlabel`/`plt.legend` with `my_ax.set_*` calls.
   - Fix `plot_key_vs_key` legend to include field pair (`"basename: k1 vs k2"`).
   - Try `getUnitKey("t")` for time axis; fall back to `"t [s]"` on exception.

2. **Step 3c-meta** *(~M, independent of Steps 7–8)*
   - `field_defs.py`: `label=` kwarg on `add_field_def`/`update_field_def`/`list_field_defs`; CLI `--label`.
   - `magnetdata_base.py`: `FieldMeta` NamedTuple, `field_meta` dict, `getFieldMeta()`,
     `load_units_from_json` stores `FieldMeta`, `addData`/`computeData` signature fix.
   - `magnetdata_pandas.py` + `magnetdata_tdms.py`: populate `field_meta` in `addData`.
   - `hybrid/hybrid_data.py`: prefix-aware `load_units_from_json` (Bug 2 fix),
     `field_meta` dict, `getFieldMeta()`, lazy `addData`.
   - `hybrid/hybrid_run.py`: add `getUnitKey()` alias.

3. **Step 3c** *(~S, depends on 3c-meta)*
   - Add `plotting/utils.py`: `format_axis_label`, `format_legend_label`,
     `resolve_legend_labels`, `_extract_suffix`.

4. **Steps 7–8** *(~S each, depends on 3c)*
   - `analysis/plotting.py` → import from `plotting.style`, use `AnnotationManager`,
     apply label fixes from `label-legend-uniformization.plan.md`.
   - `hybrid/plotting.py` → use `get_backend()` + `downsample_arrays()`.
   - `utils/plots.py` → `ax.set_*` API; optional label params.

5. **Step 9** *(~S, independent)*
   - `pyproject.toml` `plotting` + `resampler` extras groups.

6. **Step 10** *(~M, after Steps 3c and 7–8)*
   - Complete `test_timeseries.py` (unit conversion, ylabel rules, normalize-on-subplots).
   - Add `test_units.py` and `test_save_show.py`.

7. **Step 11** *(~M, after Steps 3c, 7–8 — last)*
   - `plot_vprocess.py` — remove remaining `plt.*`, normalise `--no_show` → `--show`.
   - `plot_hybrid_with_pupitre_tdms.py`, `plot_hybrid_minimal.py` — replace `plt.*`.
   - `plot_rms.py` — replace `plt.*`, rename `--output` → `--save`.
   - `plot_fepc_data.py`, `plot_trigger_data.py`, `outliers.py` — replace `plt.*`.
   - `cmp_fields.py`, `corr_Ih_Ib.py` — replace `plt.*` frame; keep `plotData()`.
   - `timeseries-anomaly-detection.py` — replace `plot_anomalies()` with
     `plot_overlay()` + `AnnotationManager`.
