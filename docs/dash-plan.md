# Dash GUI — Implementation Plan

Two web-app GUI applications built with [Plotly Dash](https://dash.plotly.com/),
runnable locally or deployable on a lab server.

The key advantage over the pywry plan: Dash callbacks are **pure Python functions** —
no JS, no message protocol, no HTML templates.  The figure-building logic from
`marimo/12_pigbrother_viewer.py` translates almost line-for-line into a Dash callback.

---

## Why Dash vs alternatives

| Feature | TUI (Textual) | pywry | **Dash** | Marimo |
|---|---|---|---|---|
| Works over SSH | Yes | No | Yes (port-forward) | Yes |
| Plot interactivity | ASCII only | Full Plotly | **Full Plotly** | Full Plotly |
| Shared x-axis | No | JS hack | **Native** (`make_subplots`) | Native |
| Python-only | Yes | No (HTML/JS templates) | **Yes** | Yes |
| Deployable on server | No | No | **Yes** | Yes |
| Reactive model | Event-based | Message passing | **Decorated callbacks** (cleanest) | Reactive cells |
| Dependency weight | Light | Medium | Medium | Medium |

Dash's reactive callback model is the closest Python equivalent to Marimo's reactive
cells — both automatically re-run a function when its inputs change.

---

## Shared prerequisites

Add to `pyproject.toml` under `[project.optional-dependencies]`:

```toml
dash = [
    "dash>=2.16",
    "dash-bootstrap-components>=1.6",
    "flask-caching>=2.3",
    "plotly>=5.0",
]
```

Package skeleton (note: folder cannot be named `dash` — conflicts with the package):

```
python_magnetrun/dash_app/
├── __init__.py
├── cache.py          # shared Flask-Caching setup + MagnetRun cache helpers
├── figures.py        # shared figure builders (ported from 12_pigbrother_viewer.py)
├── app_basic.py      # App 1 — MagnetRun Demonstrator
└── app_pigbrother.py # App 2 — Pigbrother Viewer
```

---

## Shared figure builder (`dash_app/figures.py`)

Port the Marimo `_add_traces` + `_make_subplots` logic here so both apps and any
future script can call it.

```python
def build_pigbrother_figure(
    mdata,
    group_a: str,
    channels_a: list[str],
    group_b: str,
    channels_b: list[str],
    *,
    fft: bool = False,
) -> go.Figure:
    """Build a two-panel Plotly figure with shared x-axis.

    Parameters
    ----------
    mdata :
        MagnetData object (TDMS-backed).
    group_a, group_b : str
        TDMS group names for Panel A and Panel B.
    channels_a, channels_b : list[str]
        Selected channel names for each panel.
    fft : bool
        If True, compute rfft and plot vs frequency instead of time.

    Returns
    -------
    go.Figure
        Two-row Plotly figure with ``shared_xaxes=True``.
    """
```

This function is a direct translation of the Marimo cell (lines 185–227 of
`12_pigbrother_viewer.py`): same `make_subplots`, same `_add_traces`, same
`uirevision="shared"` layout.

---

## Shared MagnetRun cache (`dash_app/cache.py`)

TDMS files can be large; re-loading on every callback would be slow.
Use `flask_caching.Cache` with `SimpleCache` (in-process, no Redis needed locally):

```python
from flask_caching import Cache

cache = Cache(config={"CACHE_TYPE": "SimpleCache", "CACHE_DEFAULT_TIMEOUT": 600})

def get_mrun(path: str, housing: str):
    key = f"{path}::{housing}"
    hit = cache.get(key)
    if hit is not None:
        return hit
    mrun = load_mrun(path, housing=housing, auto_resolve=False)
    cache.set(key, mrun)
    return mrun
```

---

## App 1 — MagnetRun Demonstrator

### Layout

```
┌─ Sidebar ────────────────────┐  ┌─ Main ──────────────────────────────────────────┐
│ File path                    │  │                                                  │
│ [___________________________]│  │  Plotly time-series chart                        │
│ [Load]                       │  │  (full zoom / pan / hover / legend toggle)       │
│                              │  │                                                  │
│ Keys                         │  ├──────────────────────────────────────────────────┤
│ [✓] Field                    │  │                                                  │
│ [✓] Courant_A1               │  │  Stats table  (df[keys].describe())              │
│ [ ] Courant_A2               │  │  shown when "Stats" tab active                  │
│ [✓] Debit                    │  │                                                  │
│ [ ] Twater_in                │  └──────────────────────────────────────────────────┘
│                              │
│ [Enable ALL]                 │
│ ────────────────────────     │
│ [Stats]   [Export CSV]       │
└──────────────────────────────┘
```

Uses `dash-bootstrap-components` `dbc.Row` / `dbc.Col` for the two-column layout.

### Callbacks

```
[Load] click + file path  ──→  keys-store (dcc.Store)
                           ──→  key-checklist options

key-checklist values      ──→  main-graph figure
                           ──→  stats-table (via dcc.Tabs)

[Export CSV] click        ──→  dcc.Download (CSV bytes)
```

### Implementation steps

| # | Task | File | Verify |
|---|------|------|--------|
| 1 | `cache.py` — `Cache` setup + `get_mrun(path, housing)` helper | `dash_app/cache.py` | cache hit on second call |
| 2 | `figures.py` — `build_timeseries_figure(mdata, keys)`: one-panel Plotly figure; wraps `PlotlyBackend` | `dash_app/figures.py` | valid figure renders in browser |
| 3 | `app_basic.py` — layout: `dbc.Row([sidebar_col, main_col])`; sidebar has `dcc.Input`, `dcc.Checklist`, `dbc.Button`s; main has `dcc.Tabs([graph_tab, stats_tab])` | `dash_app/app_basic.py` | layout renders at `localhost:8050` |
| 4 | Callback 1 — file load: `Input("load-btn","n_clicks")` + `State("file-path","value")` → `Output("keys-store","data")` + `Output("key-checklist","options")`; calls `get_mrun()` | same | checklist populates after click |
| 5 | Callback 2 — figure update: `Input("key-checklist","value")` + `State("keys-store","data")` → `Output("main-graph","figure")`; calls `build_timeseries_figure()` | same | chart updates on toggle |
| 6 | Callback 3 — stats: `Input("key-checklist","value")` + `State("keys-store","data")` → `Output("stats-table","data")` + `Output("stats-table","columns")`; uses `df[keys].describe()` | same | table populates |
| 7 | Callback 4 — export: `Input("export-btn","n_clicks")` + `State("key-checklist","value")` → `Output("download","data")`; uses `dcc.send_data_frame(df.to_csv)` | same | CSV downloads |
| 8 | Add `magnetrun-dash` entry point → `python_magnetrun.dash_app.app_basic:main` | `pyproject.toml` | `magnetrun-dash` opens browser |

---

## App 2 — Pigbrother Viewer

Closest possible translation of `marimo/12_pigbrother_viewer.py` into a standalone
Dash app.  The figure builder (`build_pigbrother_figure`) is shared from `figures.py`.

### Layout

```
┌─ Sidebar (300 px) ───────────────────────┐  ┌─ Main ─────────────────────────────────┐
│ File path                                │  │                                        │
│ [____________________________________]   │  │  ┌─ Panel A ──────────────────────┐    │
│ [Load]  Housing: [M9 ▼]                  │  │  │                                │    │
│                                          │  │  │  Plotly traces — group A       │    │
│ Preset: [GR1 ▼]   FFT [●]               │  │  │                                │    │
│ ──────────────────────────────────────   │  │  └────────────────────────────────┘    │
│ Panel A                                  │  │  ┌─ Panel B ──────────────────────┐    │
│ Group: [Alim_GR1 ▼]  [Enable ALL ●]      │  │  │                                │    │
│  [✓] Courant_A1                          │  │  │  Plotly traces — group B       │    │
│  [✓] Courant_A2                          │  │  │  (shared x-axis via            │    │
│  [✓] Référence_A1                        │  │  │   make_subplots rows=2)        │    │
│  [ ] Référence_A2                        │  │  └────────────────────────────────┘    │
│ ──────────────────────────────────────   │  │                                        │
│ Panel B                                  │  └────────────────────────────────────────┘
│ Group: [Alim_GR2 ▼]  [Enable ALL ●]      │
│  [✓] Courant_B1                          │
│  [ ] Debit                               │
│  [✓] Champ_mag                           │
│                                          │
│ [Stats]                                  │
└──────────────────────────────────────────┘
```

A single `dcc.Graph` renders the two-row `make_subplots` figure —
`shared_xaxes=True` is native, no JS hack needed.

### Callbacks

```
[Load] + path + housing  ──→  mrun-store (dcc.Store, stores path+housing key)
                          ──→  preset-dropdown options + value
                          ──→  group-a options, group-b options

preset value             ──→  group-a value  (preset drives panel A)
                          ──→  group-b value  (next group drives panel B)

group-a value            ──→  channels-a checklist options + all selected

group-b value            ──→  channels-b checklist options + all selected

enable-all-a toggle      ──→  channels-a value  (all or none)
enable-all-b toggle      ──→  channels-b value

ANY of the above +       ──→  main-graph figure
  fft-toggle
  channels-a
  channels-b
```

### Implementation steps

| # | Task | File | Verify |
|---|------|------|--------|
| 1 | `figures.py` — `build_pigbrother_figure()`: port of Marimo lines 185–227; `make_subplots(rows=2, shared_xaxes=True)`, same `_add_traces` FFT logic, same `uirevision="shared"` | `dash_app/figures.py` | figure matches Marimo output |
| 2 | `app_pigbrother.py` — layout: sidebar + single `dcc.Graph` + `dcc.Store` for cached mrun key | `dash_app/app_pigbrother.py` | layout renders |
| 3 | Callback — file load: `Input("load-btn","n_clicks")` + `State`s for path + housing → `Output("mrun-store","data")` + group dropdown `options` for preset, A, B; calls `get_mrun()` then extracts groups with `{k.split("/")[0] for k in mrun.getKeys() if "/" in k}` — identical to Marimo | same | groups populate |
| 4 | Callback — preset → group A + group B values: mirrors Marimo default (`groups[idx]` and `groups[(idx+1) % len]`) | same | both dropdowns update |
| 5 | Callback — group A → channels A options (all selected by default); same for group B | same | checklists rebuild |
| 6 | Callback — enable-all switches → channels value (`all channels` or `[]`) | same | all/none toggles |
| 7 | Callback — figure: `Input`s = group A, channels A, group B, channels B, FFT toggle, mrun-store → `Output("main-graph","figure")`; calls `build_pigbrother_figure()` | same | chart updates on any control change |
| 8 | Callback — stats modal: `Input("stats-btn","n_clicks")` → `Output("stats-modal","is_open")` + `Output("stats-table","children")`; `dbc.Modal` with two `dbc.Table`s from `describe()` | same | modal opens |
| 9 | Add `magnetrun-pigbrother-dash` entry point | `pyproject.toml` | command opens browser |

---

## Comparison with Marimo (`12_pigbrother_viewer.py`)

The translation is nearly 1:1:

| Marimo construct | Dash equivalent |
|---|---|
| `mo.ui.text(value=..., label=...)` | `dcc.Input(value=..., placeholder=...)` |
| `mo.ui.dropdown(options=..., value=...)` | `dcc.Dropdown(options=..., value=...)` |
| `mo.ui.switch(label=...)` | `dbc.Switch(label=...)` |
| `mo.ui.checkbox(label=c, value=True)` | `dcc.Checklist(options=[c], value=[c])` |
| `mo.ui.array([...])` | Single `dcc.Checklist(options=[...])` |
| `@app.cell` reactive dependency | `@app.callback(Input(...), ...)` |
| `mo.stop(condition, msg)` | `raise PreventUpdate` in callback |
| `mo.ui.plotly(_fig)` | `dcc.Graph(figure=_fig)` |
| `mdata.getTdmsData(group, None)` | same — no change |
| `_make_subplots(rows=2, shared_xaxes=True)` | **same** — direct reuse |

The figure-building code (`build_pigbrother_figure`) is essentially the Marimo cell
pasted into a plain Python function with `_` prefixes removed.

---

## Running the apps

```bash
# App 1
magnetrun-dash
# or
python -m python_magnetrun.dash_app.app_basic

# App 2
magnetrun-pigbrother-dash
# or
python -m python_magnetrun.dash_app.app_pigbrother

# Both default to http://localhost:8050 — pass --port to override
```

For remote access (lab server or SSH tunnel):

```bash
# On the server
magnetrun-pigbrother-dash --host 0.0.0.0 --port 8050

# On your laptop
ssh -L 8050:localhost:8050 user@server
# then open http://localhost:8050
```

---

## File tree (final)

```
python_magnetrun/dash_app/
├── __init__.py
├── cache.py              # Flask-Caching setup + get_mrun() helper
├── figures.py            # build_timeseries_figure(), build_pigbrother_figure()
├── app_basic.py          # App 1 — MagnetRun Demonstrator
└── app_pigbrother.py     # App 2 — Pigbrother Viewer
```

Entry points in `pyproject.toml`:

```toml
magnetrun-dash             = "python_magnetrun.dash_app.app_basic:main"
magnetrun-pigbrother-dash  = "python_magnetrun.dash_app.app_pigbrother:main"
```

---

## Three-way comparison: TUI / pywry / Dash

| Criterion | TUI (Textual) | pywry GUI | **Dash** |
|---|---|---|---|
| Works over SSH | Yes (native) | No | Yes (port-forward) |
| Plot quality | ASCII (plotext) | Full Plotly | **Full Plotly** |
| Shared x-axis | No | JS `relayout` sync | **`make_subplots` native** |
| Python-only | Yes | No (HTML/JS) | **Yes** |
| Deployable on server | No | No | **Yes** |
| Code reuse from Marimo | Partial (logic only) | Partial (logic + JSON) | **High (figure code ~verbatim)** |
| Dependency size | Smallest | Medium | Medium |
| Debug / dev UX | `textual console` | browser devtools | **`debug=True` hot-reload** |
