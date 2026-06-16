# pywry GUI — Implementation Plan

Two desktop GUI applications built with [pywry](https://github.com/OpenBB-finance/pywry)
(native webview window) + [Plotly.js](https://plotly.com/javascript/) for fully interactive charts.

pywry opens a lightweight native OS webview (no browser required) and supports
bidirectional Python ↔ JavaScript message passing.  The existing
`PlotlyBackend.to_json()` method is the natural bridge: Python builds figures,
serialises them to JSON, and sends them to the webview for Plotly.js to render.

---

## Why pywry over alternatives

| Option | Pros | Cons |
|---|---|---|
| **pywry** | Native window, no browser, full Plotly interactivity (zoom/hover/pan) | Requires OS webview (WebKit2GTK on Linux) |
| Dash/Panel | Rich ecosystem | Runs a server, needs a browser tab |
| PyWebView | Similar to pywry | Less maintained, heavier |
| Marimo | Great notebooks | Not a standalone app; requires notebook runtime |

---

## Shared prerequisites

Add to `pyproject.toml` under `[project.optional-dependencies]`:

```toml
gui = [
    "pywry>=0.5",
    "plotly>=5.0",
]
```

Linux also needs the system package `python3-gi` + `gir1.2-webkit2-4.1` (WebKit2GTK).

Package skeleton:

```
python_magnetrun/gui/
├── __init__.py
├── backend.py          # shared: message router, figure builders, data helpers
├── app_basic.py        # App 1 — MagnetRun Demonstrator
├── app_pigbrother.py   # App 2 — Pigbrother Viewer
└── templates/
    ├── base.html       # shared JS helpers + Plotly.js include
    ├── basic.html      # App 1 HTML template
    └── pigbrother.html # App 2 HTML template
```

---

## Architecture: Python ↔ JS message protocol

All messages are JSON objects with a `type` discriminator.

### Python → JS (outgoing)

| `type` | Payload | Trigger |
|---|---|---|
| `file_loaded` | `{keys, housing, filename, duration}` | File parsed successfully |
| `plot_update` | `{figure: <Plotly JSON>}` | Channel selection or FFT toggle changes |
| `stats_update` | `{html: "<table>..."}` | Stats view requested |
| `groups_loaded` | `{groups: [...]}` | TDMS groups extracted after file load |
| `channels_loaded` | `{panel, channels}` | Group selected in Panel A or B |
| `error` | `{message}` | Any Python-side exception |

### JS → Python (incoming)

| `type` | Payload | Action |
|---|---|---|
| `file_select` | `{path, housing}` | Load file via `load_mrun()` |
| `keys_change` | `{keys: [...]}` | Regenerate plot for App 1 |
| `panel_change` | `{panel, group, channels}` | Reload TDMS group + regenerate figure |
| `fft_toggle` | `{enabled}` | Recompute FFT traces + regenerate figure |
| `export_csv` | `{keys, path}` | Call `convert_to_csv()` |
| `stats_request` | `{keys}` | Compute `df.describe()`, send HTML table |

Messages are dispatched in `backend.py`'s `MessageRouter.handle(msg)` and processed in
a background thread so the webview stays responsive during file I/O.

---

## App 1 — MagnetRun Demonstrator

### Layout (`basic.html`)

```
┌──────────── Window ───────────────────────────────────────┐
│ ┌─ Sidebar (280 px) ──┐  ┌─ Main area ───────────────────┐│
│ │ File path:          │  │                               ││
│ │ [_______________]   │  │  Plotly time-series chart     ││
│ │ [Load]              │  │  (full Plotly interactivity:  ││
│ │                     │  │   zoom, pan, hover, legend)   ││
│ │ Channels:           │  │                               ││
│ │ [✓] Field           │  ├───────────────────────────────┤│
│ │ [✓] Courant_A1      │  │                               ││
│ │ [ ] Courant_A2      │  │  Stats table (df.describe())  ││
│ │ [✓] Debit           │  │  (shown when [Stats] clicked) ││
│ │ [Enable ALL]        │  │                               ││
│ │                     │  └───────────────────────────────┘│
│ │ [Stats]  [Export]   │                                   │
│ └─────────────────────┘                                   │
└───────────────────────────────────────────────────────────┘
```

### Implementation steps

| # | Task | File | Verify |
|---|------|------|--------|
| 1 | `backend.py` — `MessageRouter` class: `handle(msg_dict)` dispatches to typed handler methods; runs in `threading.Thread` | `gui/backend.py` | unit-testable without pywry |
| 2 | `backend.py` — `load_file(path, housing)`: calls `load_mrun()`, extracts keys, returns `FileLoadedPayload` | `gui/backend.py` | loads `.txt`, `.tdms`, `.csv` |
| 3 | `backend.py` — `build_timeseries_figure(mdata, keys)`: calls `PlotlyBackend` + `to_json()`; returns Plotly JSON string | `gui/backend.py` | valid JSON, renders in browser |
| 4 | `backend.py` — `build_stats_html(mdata, keys)`: `df[keys].describe().to_html()` with Bootstrap classes | `gui/backend.py` | HTML table renders |
| 5 | `templates/base.html` — include Plotly.js (CDN or bundled), define `window.postToBackend(msg)` helper and `window.onPythonMessage(msg)` dispatcher | `gui/templates/base.html` | JS helpers callable |
| 6 | `templates/basic.html` — sidebar + main layout (CSS flexbox); channel checkboxes built dynamically from `file_loaded` message; "Enable ALL" toggle | `gui/templates/basic.html` | renders in any browser |
| 7 | `app_basic.py` — `App` class: creates `PyWry` handler, loads `basic.html`, wires `on_message` → `MessageRouter.handle()`, calls `handler.send_outgoing()` to push responses | `gui/app_basic.py` | window opens |
| 8 | Wire file load: `[Load]` click → JS posts `file_select` → Python loads file → sends `file_loaded` → JS populates checkboxes + sends initial `keys_change` → Python sends `plot_update` | end-to-end | chart appears after load |
| 9 | Wire channel toggle: checkbox change → JS posts `keys_change` → Python regenerates figure → sends `plot_update` → JS calls `Plotly.react()` | end-to-end | chart updates live |
| 10 | Wire stats: `[Stats]` click → JS posts `stats_request` → Python sends `stats_update` → JS shows HTML table in main area | end-to-end | table appears |
| 11 | Wire export: `[Export]` click → JS posts `export_csv` → Python calls `convert_to_csv()` → sends notification | end-to-end | CSV written |
| 12 | Add `magnetrun-gui` entry point | `pyproject.toml` | command opens window |

---

## App 2 — Pigbrother Viewer

Mirrors `marimo/12_pigbrother_viewer.py` exactly, as a standalone native window.

### Layout (`pigbrother.html`)

```
┌──────────── Window ───────────────────────────────────────┐
│ ┌─ Sidebar (300 px) ────────────────┐  ┌─ Main area ─────┐│
│ │ File: [______________________]    │  │                 ││
│ │ [Browse]  Housing: [M9 ▼]         │  │  Panel A plot   ││
│ │ Preset: [GR1 ▼]   [FFT: OFF ●]   │  │  (Plotly,       ││
│ │ ──────────────────────────────    │  │   full zoom/    ││
│ │ Panel A                           │  │   hover/pan)    ││
│ │ Group: [Alim_GR1 ▼]               │  │                 ││
│ │ [Enable ALL ●]                    │  ├─────────────────┤│
│ │  [✓] Courant_A1                   │  │                 ││
│ │  [✓] Courant_A2                   │  │  Panel B plot   ││
│ │  [✓] Référence_A1                 │  │  (shared x-axis ││
│ │  [ ] Référence_A2                 │  │   via Plotly    ││
│ │ ──────────────────────────────    │  │   relayout      ││
│ │ Panel B                           │  │   event sync)   ││
│ │ Group: [Alim_GR2 ▼]               │  │                 ││
│ │ [Enable ALL ●]                    │  └─────────────────┘│
│ │  [✓] Courant_B1                   │                     │
│ │  [✓] Debit                        │                     │
│ │  [ ] Champ_mag                    │                     │
│ │                                   │                     │
│ │ [Stats]                           │                     │
│ └───────────────────────────────────┘                     │
└───────────────────────────────────────────────────────────┘
```

**Shared x-axis zoom**: two separate Plotly charts in the HTML; JS listens for
`plotly_relayout` events on Panel A and mirrors the x-range to Panel B via
`Plotly.relayout()` — gives the same effect as `shared_xaxes=True` in Marimo.

### Implementation steps

| # | Task | File | Verify |
|---|------|------|--------|
| 1 | `backend.py` — `load_pigbrother(path, housing)`: calls `load_mrun()`, extracts TDMS groups with `{k.split("/")[0] for k in mrun.getKeys() if "/" in k}` — identical to Marimo logic | `gui/backend.py` | groups list correct |
| 2 | `backend.py` — `build_panel_figure(mdata, group, channels, fft)`: builds single Plotly figure for one panel; FFT path mirrors Marimo `_add_traces`; returns JSON | `gui/backend.py` | FFT and time-domain both render |
| 3 | `templates/pigbrother.html` — sidebar with all controls; two `<div>` containers for Panel A and Panel B charts; JS `syncXAxis()` listener on `plotly_relayout` | `gui/templates/pigbrother.html` | layout renders, sync works |
| 4 | `app_pigbrother.py` — `App` class; same `PyWry` + `MessageRouter` pattern as App 1 | `gui/app_pigbrother.py` | window opens |
| 5 | Wire file load: sends `groups_loaded` → JS populates both group dropdowns + preset; then auto-triggers `panel_change` for A (preset group) and B (next group) — mirrors Marimo default | end-to-end | both panels populated |
| 6 | Wire preset change: JS updates Panel A group + fires `panel_change` for A; Panel B gets next group automatically | end-to-end | preset drives both panels |
| 7 | Wire `panel_change`: Python calls `getTdmsData(group)`, rebuilds figure, sends `plot_update` with `{panel, figure}` → JS calls `Plotly.react(divId, ...)` on the correct div | end-to-end | panel updates independently |
| 8 | Wire FFT toggle: JS posts `fft_toggle` → Python recomputes both panels → sends two `plot_update` messages | end-to-end | both panels switch to FFT |
| 9 | Wire "Enable ALL": JS checks/unchecks all channel boxes, fires single `panel_change` with full/empty channel list | end-to-end | all channels appear/disappear |
| 10 | Wire stats modal: `[Stats]` click → JS posts `stats_request` with both panels' keys → Python sends combined `stats_update` → JS shows overlay modal with two `describe()` tables | end-to-end | modal shows both panels |
| 11 | Add `magnetrun-pigbrother-gui` entry point | `pyproject.toml` | command opens window |

---

## Shared backend (`gui/backend.py`)

```python
class MessageRouter:
    """Dispatch JSON messages from the webview to typed handler methods."""

    def __init__(self, app: PyWry): ...

    def handle(self, raw: str) -> None:
        """Entry point called by pywry's on_message callback."""
        msg = json.loads(raw)
        handler = getattr(self, f"_on_{msg['type']}", self._on_unknown)
        threading.Thread(target=handler, args=(msg,), daemon=True).start()

    def _send(self, payload: dict) -> None:
        self._app.send_outgoing(json.dumps(payload))

    def _on_file_select(self, msg): ...
    def _on_keys_change(self, msg): ...
    def _on_panel_change(self, msg): ...
    def _on_fft_toggle(self, msg): ...
    def _on_stats_request(self, msg): ...
    def _on_export_csv(self, msg): ...
    def _on_unknown(self, msg): ...
```

---

## Shared JS helpers (`templates/base.html`)

```javascript
// Send a message to Python
function postToBackend(msg) {
    window.webkit.messageHandlers.pywry.postMessage(JSON.stringify(msg));
}

// Receive a message from Python — pywry calls this
window.onmessage = function(event) {
    const msg = JSON.parse(event.data);
    dispatch[msg.type]?.(msg);
};

const dispatch = {
    plot_update:    (m) => Plotly.react(m.div_id, m.figure.data, m.figure.layout),
    file_loaded:    (m) => rebuildChannelList(m.keys),
    groups_loaded:  (m) => rebuildGroupSelects(m.groups),
    channels_loaded:(m) => rebuildPanelChannels(m.panel, m.channels),
    stats_update:   (m) => showStatsPanel(m.html),
    error:          (m) => showError(m.message),
};
```

---

## File tree (final)

```
python_magnetrun/gui/
├── __init__.py
├── backend.py              # MessageRouter, figure builders, data helpers
├── app_basic.py            # App 1 — MagnetRun Demonstrator
├── app_pigbrother.py       # App 2 — Pigbrother Viewer
└── templates/
    ├── base.html           # Plotly.js include + shared JS helpers
    ├── basic.html          # App 1 full template (extends base.html via Jinja2)
    └── pigbrother.html     # App 2 full template
```

Entry points in `pyproject.toml`:

```toml
magnetrun-gui              = "python_magnetrun.gui.app_basic:main"
magnetrun-pigbrother-gui   = "python_magnetrun.gui.app_pigbrother:main"
```

---

## Relation to TUI plan

The GUI and TUI apps share no code directly, but the logical split is the same:

| Layer | TUI | GUI |
|---|---|---|
| Data loading | `load_mrun()`, `getTdmsData()` | same |
| Figure building | `plotext` ASCII | `PlotlyBackend.to_json()` → Plotly.js |
| FFT | `numpy.fft` | `numpy.fft` (same) |
| Stats | `DataTable` widget | `df.describe().to_html()` |
| Controls | Textual widgets | HTML form elements |
| Interactivity | Key bindings | Mouse + Plotly native (zoom/hover/pan) |

Both complement each other: TUI for SSH / headless servers, GUI for local desktop use.
