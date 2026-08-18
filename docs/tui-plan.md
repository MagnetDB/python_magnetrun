# Textual TUI — Implementation Plan

Two terminal-UI applications built with [Textual](https://textual.textualize.io/) and
[plotext](https://github.com/piccolomo/plotext) for ASCII inline charts.

Both apps live under `python_magnetrun/tui/` and share a common widget library.

---

## Shared prerequisites

Add to `pyproject.toml` under `[project.optional-dependencies]`:

```toml
tui = [
    "textual>=0.61",
    "plotext>=5.2",
]
```

Create the package skeleton:

```
python_magnetrun/tui/
├── __init__.py
├── widgets.py        # shared reusable widgets
├── app_basic.py      # App 1 — MagnetRun Demonstrator
└── app_pigbrother.py # App 2 — Pigbrother Viewer
```

---

## App 1 — MagnetRun Demonstrator

A lightweight file-browser + channel-picker + stats viewer.
Good entry point for new users; reuses existing CLI loaders with no extra dependencies.

### Layout

```
┌─ Files (25 cols) ──────────┐┌─ Keys (22 cols) ────────────┐┌─ Stats / Plot ────────────────┐
│ ▸ data/                    ││ [✓] Field                    ││  key         mean   std  min  │
│   M9_240510.txt  [→]       ││ [✓] Courant_A1               ││  Field       12.3   0.4  11.8 │
│   M9_240511.txt            ││ [ ] Courant_A2               ││  Courant_A1  31200  1200 ...  │
│   M10_2024.tdms            ││ [✓] Debit                    ││                               │
│   ...                      ││ [ ] Twater_in                ││  (ASCII plot — vs time)       │
│                            ││ ...                          ││                               │
│                            ││ [Enable ALL ○]               ││                               │
└────────────────────────────┘└──────────────────────────────┘└───────────────────────────────┘
 [l] Load   [p] Plot extern   [s] Stats   [e] Export CSV   [q] Quit
```

### Implementation steps

| # | Task | File | Verify |
|---|------|------|--------|
| 1 | `FileList` widget — `ListView` populated by scanning data dirs via `expand_input_files`; emits `FileSelected` message | `tui/widgets.py` | selecting a row fires message |
| 2 | `KeyList` widget — `ScrollableContainer` of `Checkbox` rows + "Enable ALL" `Switch`; emits `KeysChanged` message | `tui/widgets.py` | toggling checkbox fires message |
| 3 | `StatsPane` widget — `DataTable` showing `DataFrame.describe()` for selected keys | `tui/widgets.py` | numbers appear |
| 4 | `PlotextPane` widget — `Static` that renders a `plotext` time-series chart as a string; exposes `refresh(df, keys)` | `tui/widgets.py` | ASCII traces visible |
| 5 | `App` skeleton — `Horizontal(FileList, KeyList, Vertical(StatsPane, PlotextPane))` layout | `tui/app_basic.py` | layout visible, no data |
| 6 | Wire `FileSelected` → load `MagnetRun` in `@work` thread; populate `KeyList` from `mdata.get_keys()` | `app_basic.py` | keys update after file load |
| 7 | Wire `KeysChanged` → call `StatsPane.update()` and `PlotextPane.refresh()` with filtered DataFrame | `app_basic.py` | stats and plot update |
| 8 | Key `p` → launch existing `plot_vs_time()` / Plotly in a background thread | `app_basic.py` | external plot opens |
| 9 | Key `e` → call existing `convert_to_csv()` on selected keys; show success notification | `app_basic.py` | CSV written |
| 10 | Key `s` → toggle between stats table and plot views in the right pane | `app_basic.py` | view switches |
| 11 | Add `magnetrun tui` entry point in `pyproject.toml` → `python_magnetrun.tui.app_basic:main` | `pyproject.toml` | `magnetrun tui` launches app |

### Key decisions

| Topic | Decision |
|---|---|
| File discovery | Reuse `expand_input_files` + `data_dirs.py` — no new scanning logic |
| Data loading | `@work` background thread to keep TUI responsive during TDMS/CSV parse |
| Inline chart | `plotext` ASCII time-series; one curve per selected key, colour-coded |
| External plot | Launch existing matplotlib/Plotly in a thread; works when `$DISPLAY` is set |
| Stats | `DataFrame.describe()` rendered in a `DataTable` |

---

## App 2 — Pigbrother Viewer

Mirrors the Marimo notebook `12_pigbrother_viewer.py`:
two independent TDMS-group panels, per-channel checkboxes, FFT toggle,
shared time axis, and an optional external Plotly launch.

### Layout

```
┌─ Controls (32 cols) ──────────────────────┐┌─ Plot Area ──────────────────────────────────┐
│ File: [______________________________]    ││                                              │
│ Housing: [M9 ▼]   Preset: [GR1 ▼]        ││  Panel A: Alim_GR1                           │
│ View: [ASCII ↔ extern]  [FFT: OFF ○]      ││                                              │
│ ────────────────────────────────────      ││  (plotext — Panel A selected channels)       │
│ ── Panel A ──────────────────────────     ││                                              │
│ Group: [Alim_GR1 ▼]   [Enable ALL ○]      ││──────────────────────────────────────────────│
│  [✓] Courant_A1                           ││                                              │
│  [✓] Courant_A2                           ││  Panel B: Alim_GR2                           │
│  [✓] Référence_A1                         ││                                              │
│  [ ] Référence_A2                         ││  (plotext — Panel B selected channels)       │
│  ...                                      ││                                              │
│ ── Panel B ──────────────────────────     │└──────────────────────────────────────────────┘
│ Group: [Alim_GR2 ▼]   [Enable ALL ○]      │
│  [✓] Courant_B1                           │
│  [ ] Twater_in                            │
│  [✓] Champ_mag                            │
│  ...                                      │
│                                           │
│ [p] Plotly extern   [s] Stats   [q] Quit  │
└───────────────────────────────────────────┘
```

### Implementation steps

| # | Task | File | Verify |
|---|------|------|--------|
| 1 | `PanelControls` widget — group `Select` + `ChannelList` (from App 1 shared widget); emits `PanelChanged(panel_id, group, channels)` message | `tui/widgets.py` | message fires on change |
| 2 | `DualPlotPane` widget — `Vertical(PlotextPane, PlotextPane)`; exposes `refresh_panel(panel_id, df, channels, fft)` | `tui/widgets.py` | two ASCII chart areas visible |
| 3 | `App` skeleton — `Horizontal(ControlPanel, DualPlotPane)`; `ControlPanel` = `Vertical(file Input, housing/preset Select, FFT Switch, PanelControls×2)` | `tui/app_pigbrother.py` | layout visible |
| 4 | Wire file `Input.Submitted` → load `MagnetRun` in `@work` thread; extract TDMS groups; populate both `PanelControls` group selects and preset select | `app_pigbrother.py` | groups populate after load |
| 5 | Wire preset `Select.Changed` → update Panel A group to preset value, Panel B to next group (mirrors Marimo logic) | `app_pigbrother.py` | both panels update |
| 6 | Wire `PanelChanged` → call `mdata.getTdmsData(group)` in `@work`; pass result to `DualPlotPane.refresh_panel()` | `app_pigbrother.py` | ASCII chart updates |
| 7 | Wire FFT `Switch.Changed` → recompute `numpy.fft.rfft` on each panel's current data and re-render; update x-axis label to "Frequency [Hz]" | `app_pigbrother.py` | FFT traces appear |
| 8 | Key `p` → launch Plotly figure (replicating `_make_subplots` from `12_pigbrother_viewer.py`) in a background thread | `app_pigbrother.py` | browser/window opens |
| 9 | Key `s` → open `ModalScreen` with `DataTable` showing `describe()` for both panels side-by-side | `app_pigbrother.py` | modal appears |
| 10 | Key `r` → force re-render of both panels (useful after terminal resize) | `app_pigbrother.py` | charts redraw |
| 11 | Add `magnetrun tui pigbrother` subcommand or `magnetrun-pigbrother-tui` entry point | `pyproject.toml` | command launches app |

### Key decisions

| Topic | Decision |
|---|---|
| Data loading | `@work` thread; groups extracted with `sorted({k.split("/")[0] for k in mrun.getKeys() if "/" in k})` — identical to Marimo notebook |
| Inline chart | `plotext` — two separate plots stacked; no shared-axis zoom (terminal limitation); note this in the status bar |
| FFT | `numpy.fft.rfft` + `rfftfreq` applied before passing to `plotext`; mirrors Marimo `_add_traces` logic exactly |
| External plot | Reuse the `_make_subplots` Plotly code from `12_pigbrother_viewer.py`, extracted to a standalone function in `tui/plotly_export.py` |
| Channel enable-all | `Switch` sets all checkboxes without triggering N individual messages — one batch `PanelChanged` message fired after |
| Resize | Textual fires `on_resize`; hook it to call `DualPlotPane.refresh_all()` so `plotext` redraws at new terminal size |

---

## Shared widget summary (`tui/widgets.py`)

| Widget | Used by | Description |
|---|---|---|
| `ChannelList` | both apps | Scrollable checkbox list + "Enable ALL" switch; posts `KeysChanged` |
| `PlotextPane` | both apps | `Static` wrapper around a `plotext` canvas; `refresh(df, keys, fft)` public API |
| `StatsPane` | App 1 | `DataTable` showing `df[keys].describe()` |
| `PanelControls` | App 2 | Group `Select` + `ChannelList`; posts `PanelChanged` |
| `DualPlotPane` | App 2 | Two stacked `PlotextPane`s; `refresh_panel(id, df, keys, fft)` |

---

## File tree (final)

```
python_magnetrun/tui/
├── __init__.py
├── widgets.py           # ChannelList, PlotextPane, StatsPane, PanelControls, DualPlotPane
├── app_basic.py         # App 1 — MagnetRun Demonstrator
├── app_pigbrother.py    # App 2 — Pigbrother Viewer
└── plotly_export.py     # shared Plotly figure builder (extracted from 12_pigbrother_viewer.py)
```

Entry points in `pyproject.toml`:

```toml
[project.scripts]
magnetrun-tui            = "python_magnetrun.tui.app_basic:main"
magnetrun-pigbrother-tui = "python_magnetrun.tui.app_pigbrother:main"
```
