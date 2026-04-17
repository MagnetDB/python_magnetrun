# Plotting Refactoring Plan

Date: 2026-04-17

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
9. *(likely)* **Dynamic resampling path** — a `"plotly-resampler"` backend wraps the
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
    downsample: "DownsampleConfig | None" = None,
    backend: "str | PlottingBackend" = "matplotlib",
    style: "PlotStyle | None" = None,
) -> Any:
    """One subplot per field, all sharing the same x (time) axis.

    Parameters
    ----------
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
    normalize: bool = False,
    downsample: "DownsampleConfig | None" = None,
    backend: "str | PlottingBackend" = "matplotlib",
    style: "PlotStyle | None" = None,
) -> Any:
    """All fields on one axes.

    When ``normalize=True``:
    - each series is divided by its absolute maximum before plotting;
    - the legend entry is amended: ``"Field_B  (max = 12.3 T)"``.
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

### Step 1 — Create `plotting/style.py` *(effort: S)*

- Move `PlotStyle` and `PlotColors` dataclasses verbatim from `analysis/plotting.py`.
- Add `__all__ = ["PlotStyle", "PlotColors"]`.
- In `analysis/plotting.py`, replace the class bodies with:
  ```python
  from python_magnetrun.plotting.style import PlotStyle, PlotColors  # noqa: F401
  ```
  No public API change.

### Step 2 — Create `plotting/backend.py` *(effort: S)*

- Define `PlottingBackend` Protocol and `get_backend()` factory as above.
- Create `plotting/matplotlib_backend.py` wrapping the existing matplotlib calls
  (`fig, axes = plt.subplots(n, 1, sharex=True)`…).  No new behaviour yet.
- Create `plotting/plotly_backend.py` stub — `subplots()` returns
  `plotly.graph_objects.Figure` with `make_subplots(rows=n, shared_xaxes=True)`.

### Step 3 — Create `plotting/timeseries.py` *(effort: M)*

- Implement `plot_subplots()` using the backend protocol.
  - If `downsample` is not None, call `downsample_dataframe()` from
    `python_magnetrun.utils.downsampling` (requires downsampling plan Step 1 done).
  - Fallback: if downsampling plan not yet merged, inline a simple stride-based fallback
    guarded by `try/except ImportError`.
- Implement `plot_overlay()` with normalization.
  - Normalisation: `y_norm = y / np.abs(y).max(); label_with_max = f"{field}  (max = {max_val:.3g} {unit})"`.
  - Units resolved via `MagnetDataBase.getUnitKey()` if data carries a units dict,
    otherwise omitted from the label.

### Step 4 — Create `plotting/annotations.py` *(effort: M)*

- Extract `AnnotationManager` from `analysis/plotting.py:458–606`.
- Keep the full matplotlib interactive behaviour (pick events, detail subplots,
  `open_figures` tracking).
- Add a plotly stub that stores annotations as `fig.add_annotation(...)` shapes.
- Update `analysis/plotting.py:plot_data()` to instantiate `AnnotationManager` and
  call `manager.add()` in its loop — behaviour is unchanged, code is tidier.

### Step 5 — Implement `PlotlyBackend` fully *(effort: M)*

- `add_series()`: `fig.add_trace(go.Scatter(x=t, y=y, name=label), row=ax_idx+1, col=1)`.
- `add_annotation()`: `fig.add_annotation(...)` with `hovertext=detail`.
- `save()`: `fig.write_image(path, scale=dpi/72)` (requires `kaleido`).
- `show()`: `fig.show()`.
- `to_json()`: `fig.to_json()`.
- Add `kaleido` to `pyproject.toml` as an optional dependency (new `plotting` extras group).

### Step 5b — *(likely)* Implement `PlotlyResamplerBackend` *(effort: M)*

> Depends on Step 5.  Requires `plotly-resampler` installed (new `resampler` extras group).

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

When `"plotly-resampler"` is selected, `plot_subplots()` / `plot_overlay()` skip the
`DownsampleConfig` step (tier 2 is redundant — tier 3 handles it dynamically).

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

### Step 6 — Wire up `commands/plot.py` *(effort: S)*

- Add `--backend {matplotlib,plotly,plotly-resampler,plotly-widget}` argument
  (default `matplotlib`).  Document in help text that `plotly-resampler` and
  `plotly-widget` require a live kernel.
- Replace direct `plt.*` calls in `plot_vs_time()` with calls to
  `plot_subplots(data, fields, backend=args.backend)`.
- Add `--output` / `--show` flags that call `backend.save()` / `backend.show()`.
- Add `--json` flag that prints `backend.to_json(fig)` to stdout for piping to a
  REST endpoint.

### Step 7 — Update `analysis/plotting.py` and `utils/plots.py` *(effort: S)*

- `analysis/plotting.py`: import `PlotStyle`, `PlotColors` from `plotting.style`;
  import `AnnotationManager` from `plotting.annotations`.  Remove duplicated definitions.
- `utils/plots.py`: replace `plt.subplots()` calls with `get_backend("matplotlib").subplots()`.
  These are internal utilities — behaviour unchanged, backend-aware for future.

### Step 8 — Update `hybrid/plotting.py` *(effort: S)*

- Replace the local downsampling call with `downsample_arrays()` from
  `utils.downsampling` (same dependency as downsampling plan Step 2).
- Use `get_backend(backend_name)` so hybrid plots can also be rendered via plotly.

### Step 9 — `pyproject.toml` extras *(effort: S)*

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

### Step 10 — Tests *(effort: M)*

- `tests/plotting/test_timeseries.py`
  - `test_plot_subplots_matplotlib` — creates figure, checks axes count and `sharex`.
  - `test_plot_subplots_plotly` — checks `Figure.data` length matches `fields`.
  - `test_plot_overlay_normalize` — checks max is 1.0 per series, legend contains `"max ="`.
  - `test_plot_overlay_no_normalize` — checks raw values preserved.
- `tests/plotting/test_backend.py`
  - `test_get_backend_names` — `get_backend("matplotlib")` and `get_backend("plotly")` return correct types.
  - `test_to_json_plotly` — `to_json()` returns valid JSON with `"data"` key.
  - `test_to_json_matplotlib_raises` — `MatplotlibBackend.to_json()` raises `NotImplementedError`.
- `tests/plotting/test_annotations.py`
  - `test_add_annotation_matplotlib` — annotation object present in `ax.texts`.
  - `test_annotation_manager_connect` — pick event wired without error.
- *(likely)* `tests/plotting/test_resampler_backend.py`
  - `test_get_backend_resampler` — `get_backend("plotly-resampler")` returns
    `PlotlyResamplerBackend`; skipped if `plotly-resampler` not installed.
  - `test_resampler_subplots` — figure is a `FigureResampler` instance with correct
    row count.
  - `test_resampler_add_series` — trace count matches `fields` length.
  - `test_resampler_save_raises` — `save()` raises `NotImplementedError`.
  - `test_resampler_to_json_raises` — `to_json()` raises `NotImplementedError`.

---

## File change summary

| File | Change |
|------|--------|
| `python_magnetrun/plotting/__init__.py` | **New** |
| `python_magnetrun/plotting/backend.py` | **New** — `PlottingBackend` protocol, `get_backend()` |
| `python_magnetrun/plotting/matplotlib_backend.py` | **New** |
| `python_magnetrun/plotting/plotly_backend.py` | **New** — static Plotly, `to_json()` |
| `python_magnetrun/plotting/plotly_resampler_backend.py` | *(likely)* **New** — `PlotlyResamplerBackend`, live kernel only |
| `python_magnetrun/plotting/style.py` | **New** — move `PlotStyle`, `PlotColors` from `analysis/plotting.py` |
| `python_magnetrun/plotting/timeseries.py` | **New** — `plot_subplots()`, `plot_overlay()` |
| `python_magnetrun/plotting/annotations.py` | **New** — `AnnotationManager` extracted from `analysis/plotting.py` |
| `python_magnetrun/analysis/plotting.py` | Update imports; remove moved definitions |
| `python_magnetrun/utils/plots.py` | Use `get_backend("matplotlib")` |
| `python_magnetrun/hybrid/plotting.py` | Use `get_backend()`; use `downsample_arrays()` |
| `python_magnetrun/commands/plot.py` | Add `--backend`, `--json` flags; delegate to `plot_subplots()` |
| `pyproject.toml` | Add `plotting` extras group |
| `tests/plotting/` | **New** — test files for steps above |

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

- **Downsampling refactoring plan** (`downsampling-refactoring.plan.md`) — Steps 3, 8
  call `downsample_dataframe()` and `downsample_arrays()` from `utils.downsampling`.
  Steps 3 and 8 can use a local stride fallback until the downsampling plan is merged,
  but full method selection (`minmax_lttb`, etc.) requires Steps 1–2 of that plan done.

  **Sequencing constraint:** Steps 1–2 of the downsampling plan (create
  `utils/downsampling.py`, update `hybrid_run.py`) should land before Step 3 of this
  plan.  All other steps of this plan are independent.

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

1. **Now (independent of everything):**
   - Step 1 — move `PlotStyle` / `PlotColors` to `plotting/style.py`.
   - Step 2 — create `plotting/backend.py` + `MatplotlibBackend` stub.
   Zero behaviour change; safe to land immediately.

2. **After downsampling plan Steps 1–2 are merged:**
   - Step 3 — `timeseries.py` (`plot_subplots`, `plot_overlay`).
   - Step 4 — `annotations.py` (`AnnotationManager`).

3. **After Step 3:**
   - Step 5 — full `PlotlyBackend`.
   - Step 6 — wire `commands/plot.py`.
   - *(likely)* Step 5b — `PlotlyResamplerBackend` (can be done in parallel with Step 5,
     purely additive, guarded by `try/except ImportError`).

4. **After Steps 5–6:**
   - Steps 7–8 — update `analysis/plotting.py`, `utils/plots.py`, `hybrid/plotting.py`.
   - Step 9 — `pyproject.toml` extras.
   - Step 10 — tests.
