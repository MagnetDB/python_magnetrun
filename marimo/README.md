# Interactive Demonstrators with Marimo

> This directory contains [Marimo](https://marimo.io) reactive notebooks that walk you
> through how `python_magnetrun` organises, loads, plots, and downsamples
> magnet-run data.

---

## Requirements

`marimo` must be installed in the project virtualenv alongside the optional
plotting/downsampling extras:

```bash
source magnetrun-env/bin/activate
pip install marimo                          # if not already present
pip install plotly                          # for 02_pupitre_plot.py
pip install tsdownsample                   # enables m4 / lttb / minmax_lttb
pip install simplification                 # enables rdp / vw geometry methods
```

`pandas` and `numpy` are already declared in `requirements.txt`.

---

## Notebooks

| File | Title | Purpose |
|------|-------|---------|
| `00_nas_setup.py` *(private)* | NAS Setup | Mount the NAS via rclone, write `~/.config/python_magnetrun/data_dirs.json`, verify connectivity |
| `00_data_organization.py` | Data Organization | Data sources, NAS layout, file naming conventions, housing IDs, runtime path resolution, smoke-test |
| `01_pupitre_loading.py` | Pupitre: Loading | Interactive file picker, housing selector, keys/units table, raw DataFrame, descriptive stats, `PandasMagnetData` API tour |
| `02_pupitre_plot.py` | Pupitre: Plotting | Time-series multi-channel plot, key-vs-key (X–Y) scatter plot, both using interactive Plotly |
| `02b_downsampling.py` | Downsampling Comparison | Side-by-side comparison of `stride`, `minmax`, `lttb`, `m4`, `rdp`, and `vw` methods with an interactive point-count slider |

All notebooks default to the bundled sample file in `tests/data/` so they work
immediately without a NAS connection.

---

## Running the notebooks

### Read-only presentation mode

Launches the app in a browser tab; no code is visible.

```bash
source magnetrun-env/bin/activate
marimo run 00_data_organization.py
marimo run 01_pupitre_loading.py
marimo run 02_pupitre_plot.py
marimo run 02b_downsampling.py
```

### Edit / development mode

Shows source cells alongside their output — useful for extending or debugging.

```bash
marimo edit 01_pupitre_loading.py
```

By default Marimo opens `http://localhost:2718` (or the next free port).

---

## What each notebook covers

### `00_nas_setup.py` *(private)*

> [!NOTE] This notebook is for internal use only; it won't run successfully without the
> correct rclone config and NAS credentials.

Details on mounting the NAS with rclone, writing the `data_dirs.json` config file, and verifying connectivity by listing the contents of the root data directory.

### `00_data_organization.py`

Conceptual overview with no interactive widgets:

- The three acquisition systems (Pupitre, PigBrother, FEPC) and their file
  formats (`.txt`, `.tdms`).
- NAS directory tree and file-naming conventions.
- Housing identifiers (M9, M10, ...) and the corresponding defs files.
- Runtime resolution order: explicit path → env vars → config file → housing
  subdirectory.
- A live smoke-test cell that loads the bundled sample and prints the package
  version.

### `01_pupitre_loading.py`

Fully interactive:

- Text field for the `.txt` file path (pre-filled with the sample).
- Housing dropdown (M9 / M10 / ... / unknown).
- Table of all channel keys with their symbol and unit.
- First 20 rows of the raw DataFrame.
- `df.describe()` statistics table.
- Tour of the `PandasMagnetData` API: `getData()`, `stats()`, `addData()`,
  `extractTimeData()`, `saveData()`.

### `02_pupitre_plot.py`

Interactive Plotly charts:

- **Time-series** — multiselect channels plotted against elapsed time `t [s]`
  with unified hover.
- **Key-vs-key (X–Y)** — any channel on X, any channel on Y, rendered as a
  scatter + line trace.

### `02b_downsampling.py`

Algorithm comparison:

| Method | Package | Parameter |
|--------|---------|-----------|
| `stride` | built-in | `n_out` |
| `minmax` | built-in | `n_out` |
| `lttb` | `tsdownsample` | `n_out` |
| `minmax_lttb` | `tsdownsample` | `n_out` |
| `m4` / `nan_m4` | `tsdownsample` | `n_out` |
| `rdp` / `vw` | `simplification` | `epsilon` (auto-searched via `from_n_out_rdp`) |

A slider controls the target point count; only methods whose dependencies are
installed are shown.

---

## Extending the notebooks

Open any notebook in edit mode and append new cells.  Each cell is a plain
Python function; variables it returns are available to all later cells.
`mo.ui.*` widgets (sliders, dropdowns, tables, …) are reactive — changing a
widget automatically re-runs every dependent cell.
