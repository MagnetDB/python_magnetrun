# Marimo Tutorial Plan for python_magnetrun

## Goal

Build a set of marimo notebooks that explain how python_magnetrun is organized
and how to use it, covering both CLI and API modes. Notebooks are stored in
`marimo/` at the repo root.

---

## Prerequisites before starting

1. **Finish downsampling plans** — the tutorial demonstrates finished, stable
   APIs. All downsampling refactoring (`rdp-downsampling.plan.md`,
   `m4-downsampling.plan.md`, `downsampling-refactoring.plan.md`) must be
   closed before notebook work begins, so the notebooks do not need rewriting
   later.

2. **Plotting backend: plotly throughout** — all notebooks use plotly, not
   matplotlib, for two reasons:
   - Interactive zoom/pan/hover lets users see where RDP vs M4 differ in a way
     static figures cannot show.
   - marimo reactive UI + plotly means a downsample-factor slider immediately
     re-renders the plot, making trade-offs tangible.
   The codebase already supports `--backend plotly`; no new dependency needed.

---

## Sensitive data strategy

- `marimo/00_nas_setup.py` is added to `.gitignore` — never committed.
- A `.env.example` template (committed) holds placeholder values for NAS
  credentials; users copy it to `.env` (gitignored) and fill in their details.
- The rclone config (`~/.config/rclone/rclone.conf`) stores actual secrets;
  notebooks only document how to create it, never embed credentials.
- `marimo/00_data_organization.py` is fully committed and safe to share.

---

## Notebook inventory

### Preamble — Setup & Data Organization

| File | Status | Topic |
|------|--------|-------|
| `00_data_organization.py` | committed | NAS layout, file naming conventions, data sources overview |
| `00_nas_setup.py` | **gitignored** | rclone install & config, mounting NAS, sensitive values via `os.environ` |

### Part 1 — Pupitre data (`.txt`)

| File | Topic |
|------|-------|
| `01_pupitre_loading.py` | Load a `.txt` file, inspect metadata, list fields & units |
| `02_pupitre_plot.py` | Time series & key-vs-key plots for one file (plotly) |
| `02b_downsampling.py` | Why downsampling matters; raw vs RDP vs M4 side-by-side; slider UI; when to use each |
| `03_pupitre_multifile.py` | Overlay / compare fields across several `.txt` files |
| `04_pupitre_stats.py` | Statistics, derived fields (`addData`), plateau signatures |
| `05_pupitre_extract_export.py` | Extract by time range / threshold, export to CSV/TSV |

### Part 2 — PigBrother data (`.tdms`)

| File | Topic |
|------|-------|
| `06_pigbrother_loading.py` | Load a `.tdms` file, explore groups & channels |
| `07_pigbrother_plot.py` | Plot from TDMS groups |
| `08_pigbrother_stats.py` | Statistics & export on TDMS data |

### Part 3 — Hybrid data (FEPC)

| File | Topic |
|------|-------|
| `09_hybrid_loading.py` | Load kHz / RMS / trigger data |
| `10_hybrid_plot.py` | Visualize hybrid signals |

### Part 4 — Cross-analysis & Dashboard

| File | Topic |
|------|-------|
| `11_cross_analysis.py` | Sync and compare fields across Pupitre + PigBrother + Hybrid |
| `12_dashboard.py` | Interactive marimo app: file picker, field selector, time range, multi-source overlay |

---

## Build order

0. **Close all downsampling plans** (prerequisite — do not start notebooks before this).
1. Validate approach with Part 1 (Pupitre) notebooks first; `02b_downsampling.py`
   establishes the downsample pattern used by all subsequent plotting notebooks.
2. Extend to PigBrother once Part 1 patterns are stable.
3. Add Hybrid notebooks once TDMS patterns are confirmed.
4. Cross-analysis and dashboard last — they reuse all prior patterns.

---

## Key API reference

```python
from python_magnetrun.MagnetRun import load_mrun

mrun = load_mrun("file.txt", housing="M9")
keys  = mrun.getKeys()
df    = mrun.getDataFrame()           # single DataFrame for .txt
sym, unit = mrun.getUnit("Field")
mdata = mrun.getMData()
mdata.addData("PowerH", formula="IH * UH / 1.e+6", symbol="P", unit="megawatt", ...)
stats = mdata.stats("Field")
sub   = mdata.extractTimeData("2019-02-14 23:00;2019-02-15 00:00", time_zone="Europe/Paris")
mdata.saveData(["Field", "IH"], "output.tsv")
```

For TDMS:
```python
mrun  = load_mrun("file.tdms", housing="M9")
mdata = mrun.getMData()
for group in mdata.Groups:
    df = mdata.getData(group)
```
