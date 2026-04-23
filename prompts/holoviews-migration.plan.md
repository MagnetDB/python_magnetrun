# HoloViews Migration Plan

## Motivation

The current `plotting/` subpackage implements a custom `PlottingBackend` Protocol with three
concrete backends (matplotlib, plotly, plotly-resampler) totalling ~636 LOC. Backend switching
requires instantiating different classes; large-data handling uses `plotly-resampler`'s
`FigureResampler`; interactive annotations are backend-specific and hard to generalise.

HoloViews replaces all of that: backend switching is a one-liner (`hv.extension(...)`), large-data
resampling is handled by `datashader.rasterize`, and interactive annotations use Panel streams.

---

## Scope

### Files removed / replaced

| File | LOC | Fate |
|------|-----|------|
| `plotting/backend.py` | 123 | Deleted — `PlottingBackend` Protocol superseded by `hv.extension()` |
| `plotting/matplotlib_backend.py` | 124 | Deleted |
| `plotting/plotly_backend.py` | 184 | Deleted |
| `plotting/plotly_resampler_backend.py` | 205 | Deleted — replaced by `hv.operation.datashader.rasterize` |
| `plotting/timeseries.py` | 379 | Rewritten — `hv.Curve`, `hv.Layout`, `hv.Overlay` |
| `plotting/annotations.py` | 198 | Rewritten — Panel streams (`Tap`, `Points`) |

### Files kept / lightly modified

| File | LOC | Change |
|------|-----|--------|
| `plotting/style.py` | 121 | Keep `PlotStyle`, `PlotColors`, `PlotConfig`; adapt `figsize`/`dpi` to `hv.opts` |
| `plotting/utils.py` | 103 | Keep `format_label()` — unchanged |
| `plotting/cli.py` | 150 | Update `--backend` choices to `matplotlib\|bokeh\|plotly`; `hv.extension()` call |
| `plotting/__init__.py` | 60 | Update public API |

### Downstream files updated

| File | Change |
|------|--------|
| `analysis/plotting.py` | Rewrite `plot_data()`, `plot_regimes()`, `plot_incidents_markers()` to use `hv` API |
| `commands/plot.py` | Remove `_backend_name` fragility; use `hv.extension()`; `hv.save()` / Panel |
| `pyproject.toml` | Remove `plotting` / `resampler` extras; add `holoviews`, `panel`, `bokeh`, `datashader` |

---

## New Dependencies

| Package | Purpose | Replaces |
|---------|---------|---------|
| `holoviews` | Core plotting abstraction | Custom `PlottingBackend` Protocol |
| `panel` | Show/serve, interactive callbacks | `backend.show()`, matplotlib pick-events |
| `bokeh` | Primary interactive backend | plotly interactive |
| `datashader` | Large time-series rasterization | `plotly-resampler` |
| `hvplot` | (optional) pandas `.hvplot()` accessor | — |

`matplotlib` and `plotly` remain as optional HoloViews rendering backends — no need to
remove them from the user environment.

---

## Phases

### Phase HV-1 — Core HoloViews Primitives (2 days)

Replace the `PlottingBackend` / three-backend architecture with HoloViews primitives.

**Steps:**

1. Remove `backend.py`, `matplotlib_backend.py`, `plotly_backend.py`,
   `plotly_resampler_backend.py`.
2. Rewrite `timeseries.py`:
   - `plot_subplots(df, fields, …)` → `hv.Layout([hv.Curve(df, kdims=['t'], vdims=[f]) for f in fields]).cols(1)`
   - `plot_overlay(df, fields, …)` → `hv.Overlay([hv.Curve(df, kdims=['t'], vdims=[f]) for f in fields])`
   - Unit conversion (`_resolve_units()`) stays as a pre-processing step before constructing `hv.Curve`
   - Normalization applied at data level (divide by max) before constructing curves
   - Multi-file coloring: `hv.Cycle` via `hv.opts.Curve(color=hv.Cycle(colors))`
3. Backend selection in `plotting/cli.py`:
   ```python
   hv.extension(args.backend)  # 'matplotlib' | 'bokeh' | 'plotly'
   ```
4. Add `hv.opts` pass-through from `PlotStyle`: map `figsize`, `dpi`, `grid`, `legend_loc`
   to equivalent `hv.opts.Curve(...)` / `hv.opts.Overlay(...)` kwargs.

**Acceptance criteria:**
- `pytest tests/` passes
- `magnetrun plot --backend matplotlib` produces correct static output
- `magnetrun plot --backend bokeh` launches interactive browser plot

---

### Phase HV-2 — Large Data with Datashader (1 day)

Replace `PlotlyResamplerBackend` with `datashader.rasterize`.

**Steps:**

1. Add `use_datashader: bool` option to `plot_subplots()` / `plot_overlay()`.
2. When enabled, wrap each `hv.Curve` with `hd.datashade()` or `hd.rasterize()` +
   `hd.dynspread()`:
   ```python
   import holoviews.operation.datashader as hd
   curve = hv.Curve(df, kdims=['t'], vdims=[field])
   shaded = hd.rasterize(curve).opts(cmap='blue')
   ```
3. Auto-detect: if `len(df) > DATASHADER_THRESHOLD` (default 100 000 rows), enable automatically.
4. Update `commands/plot.py`: `--datashader` / `--no-datashader` flags; remove
   `--backend plotly-resampler` (now subsumed).

**Acceptance criteria:**
- A 1M-row dataset renders without browser slowdown using `--backend bokeh`
- `--datashader` flag explicitly enables rasterization
- `--no-datashader` forces raw curves (for small data)

---

### Phase HV-3 — Save / Show / CLI Integration (1 day)

Replace `backend.save()` / `backend.show()` with HoloViews / Panel equivalents.

**Steps:**

1. Show:
   ```python
   # was: backend.show()
   panel.panel(hvplot).show()        # bokeh: launches browser
   hv.render(hvplot, backend='matplotlib')  # matplotlib: plt.show()
   ```
2. Save:
   ```python
   hv.save(hvplot, filename)         # .png / .svg / .html depending on extension
   ```
3. Remove `to_json()` method (was plotly-specific). Replace with `hv.save(..., fmt='html')`
   for shareable interactive output.
4. Update `commands/plot.py`:
   - Remove the legacy matplotlib fast-path (direct `pandas.plotData` calls).
   - `--json` flag renamed / repurposed to `--html` for interactive HTML export.
   - `_plot_vs_time_backend()` grouping logic unchanged (extension-based grouping is a
     data concern, not a backend concern).
5. Update `commands/plot.py` `plot_key_vs_key()`: use `hv.Scatter` instead of
   `matplotlib.scatter` — now backend-agnostic.

**Acceptance criteria:**
- `magnetrun plot --save output.png` saves static image
- `magnetrun plot --save output.html` saves interactive bokeh HTML
- `magnetrun plot --show` opens browser / matplotlib window

---

### Phase HV-4 — Annotations Rewrite (2 days)

Replace `AnnotationManager` (matplotlib pick-events + plotly yellow dots) with
Panel + HoloViews streams.

This is the highest-risk phase. Panel streams work inside Jupyter or a running Panel
server; the CLI `--show` path needs `panel.serve()` instead of a bare browser open.

**Steps:**

1. Define incident markers as `hv.Points(incidents_df, kdims=['t', 'B0'], vdims=['label'])`
   with `hv.opts.Points(marker='x', color='red', size=10)`.
2. Add a `Tap` stream:
   ```python
   tap = hv.streams.Tap(source=markers, x=0, y=0)
   ```
3. Define a callback that opens a detail panel (sub-figure) when a point is clicked:
   ```python
   def show_detail(x, y):
       nearest = find_nearest_incident(incidents_df, x)
       return hv.Curve(detail_df(nearest), kdims=['t'], vdims=['B0'])
   dmap = hv.DynamicMap(show_detail, streams=[tap])
   ```
4. Compose: `main_plot + dmap` as a Panel `Row`.
5. CLI path: `panel.serve(layout)` replaces `backend.show()`.
6. Non-interactive (static export) path: render markers as static `hv.Points`, skip stream.

**Note on matplotlib backend:** matplotlib click-events (`mpl_connect('pick_event', ...)`) are
not supported in HoloViews. For non-interactive static matplotlib output, annotations are
rendered as static markers with text labels (no click-to-detail). Interactive annotation
requires bokeh backend + Panel.

**Acceptance criteria:**
- `magnetrun analyse --show --backend bokeh` opens a panel with clickable incident markers
- `magnetrun analyse --save output.png --backend matplotlib` renders static markers with labels
- Existing `tests/` suite still passes

---

### Phase HV-5 — analysis/ Integration (1 day)

Update `analysis/plotting.py` to use the new HoloViews API.

**Steps:**

1. `plot_data()`:
   - Replace `AnnotationManager` call with Phase HV-4 Panel streams composition.
   - Replace `downsample_for_plot()` / `downsample_minmax()` calls with
     `hd.rasterize()` (Phase HV-2) — these functions can now be deleted from
     `analysis/plotting.py`, simplifying `analysis-subpackage-refactoring` Phase 2.
   - Replace `PlotStyle`/`PlotColors` opts with `hv.opts` equivalents (using the
     adapter from Phase HV-1 step 4).
2. `plot_regimes()`: replace `ax.axvspan()` with `hv.VSpan` elements.
3. `plot_incidents_markers()`: replace `ax.axvline()` with `hv.VLine` elements.
4. Delete `analysis/plotting.py` `downsample_for_plot()`, `downsample_dataframe()`,
   `downsample_minmax()`, `estimate_downsample_percent()` — superseded by datashader.
   (This is `analysis-subpackage-refactoring` Phase 2, done here instead.)

**Acceptance criteria:**
- `magnetrun analyse --show` produces correct multi-source overlay
- Downsampling functions removed without regressions

---

## Interaction with Existing Plans

### `analysis-subpackage-refactoring.plan.md`

| Phase | Interaction |
|-------|------------|
| Phase 2 (downsampling) | **Superseded by HV-5.** `datashader.rasterize` replaces manual downsampling for plotting. `DownsampleConfig` remains for non-plotting uses (processing pipeline). |
| Phase 5.3 (break down `main()`) | `_emit_plots()` now calls `hv.save()` / `panel.serve()` instead of `backend.save()`. Logic unchanged, surface API changes. |
| Phases 1, 3, 4, 6 | **Unaffected.** |

Sequencing: HoloViews migration can proceed **before or after** analysis refactoring Phases 1/3/4/6.
Phase 2 of analysis refactoring should be **skipped** and replaced by HV-5.

### `mrun-cache-implementation.plan.md`

| Phase | Interaction |
|-------|------------|
| Phase 2 Step 10 ("Migrate downstream consumers to narwhals API … plotting") | HoloViews natively accepts pandas DataFrames. When narwhals migration lands, `hv.Dataset(nw_frame.to_pandas())` is the bridge until HoloViews has native narwhals support. Low-friction. |
| All other phases | **Unaffected.** |

Sequencing: HoloViews migration is **independent** of narwhals migration. Either can go first.

### `REVIEW.md` — item 9 (plotting refactoring marked Done)

The current `plotting/` subpackage is the completed item 9. This plan supersedes it.
REVIEW.md should be updated to mark plotting as "Done (replaced by HoloViews migration)"
and add a new item for the HoloViews migration.

### Cross-domain comparison (REVIEW.md Phases D–E)

`ComparisonSession` will produce comparison plots. Using HoloViews from day one avoids
adding backend-specific paths in the comparison module. No action needed — just build
`ComparisonSession` plotting on `hv.Curve` / `hv.Overlay` from the start.

---

## Effort Estimate

| Phase | Effort | Risk | Dependency |
|-------|--------|------|-----------|
| HV-1 Core primitives | 2 d | Medium | None |
| HV-2 Datashader | 1 d | Low | HV-1 |
| HV-3 Save/Show/CLI | 1 d | Low | HV-1 |
| HV-4 Annotations | 2 d | High | HV-1, HV-3 |
| HV-5 analysis/ integration | 1 d | Low | HV-2, HV-4 |
| pyproject.toml + test fixes | 0.5 d | Low | All phases |
| **Total** | **~8 days** | | |

HV-1 through HV-3 are low-risk and can be shipped independently before tackling HV-4.
HV-4 is the only genuinely hard phase (Panel streams are a new mental model).

---

## Recommended Overall Sequencing

```
HV-1 (core primitives)
    → HV-2 (datashader) + HV-3 (save/show)   [parallel]
        → HV-4 (annotations)
            → HV-5 (analysis/ integration)    [= analysis refactoring Phase 2]
                → analysis refactoring Phases 1, 3, 4, 5, 6
                    → mrun-cache Phases 1+2b-tdms → 2 → 3
                        → Cross-domain comparison Phases D–E
```

---

## pyproject.toml Changes

```toml
# Remove
[project.optional-dependencies]
plotting = ["plotly>=5", "kaleido"]
resampler = ["plotly-resampler"]

# Add
[project.optional-dependencies]
plotting = ["holoviews", "panel", "bokeh"]
datashader = ["datashader", "holoviews[recommended]"]
hvplot = ["hvplot"]  # optional convenience accessor
```

`matplotlib` stays in core dependencies (used by HoloViews matplotlib backend and
by non-plotting code).
