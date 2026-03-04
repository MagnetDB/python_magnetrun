# Phase 4 – Dashboards & Deprecation Removal (Weeks 15–20)

## Goal

Deliver interactive dashboards for data exploration, complete the Jupyter
notebook integration, and remove all backwards-compatibility shims introduced
in Phase 2. At the end of this phase the codebase is clean: no deprecated
module aliases, no commented dead code, no hard-coded developer paths.

**Prerequisite:** Phase 3 is complete (Pydantic models, API client, YAML site config, all tests passing).

---

## Scope

### 4.1 Remove all backwards-compatibility shims

The following shim files were introduced in Phase 2 to allow gradual migration.
They must now be deleted:

| Shim file | Original location | Real location (since Phase 2) |
|-----------|------------------|-------------------------------|
| `python_magnetrun/requests/__init__.py` | shadowed `requests` PyPI | `python_magnetrun/fetchers/` |
| `python_magnetrun/python_magnetrun.py` | same name as package | `python_magnetrun/cli_main.py` |
| `python_magnetrun/magnetdata.py` | monolithic module | `python_magnetrun/magnetdata/` package |
| `python_magnetrun/processing/hysteresis.py` | monolithic module | `python_magnetrun/processing/hysteresis/` package |

**Steps:**

1. Search for any remaining internal imports of the shim paths:

   ```bash
   grep -rn "from python_magnetrun.requests\b" python_magnetrun/ tests/ examples/ --include="*.py"
   grep -rn "python_magnetrun.python_magnetrun" python_magnetrun/ tests/ examples/ --include="*.py"
   grep -rn "from python_magnetrun import magnetdata\b" python_magnetrun/ tests/ examples/ --include="*.py"
   grep -rn "from python_magnetrun.processing import hysteresis\b" python_magnetrun/ tests/ examples/ --include="*.py"
   ```

2. Fix any remaining callers to import from the canonical locations.

3. Delete the four shim files/directories.

4. Also delete the `python_magnetrun/requests/` directory entirely (all files).

**Acceptance criteria:**
- None of the deleted paths exist in `python_magnetrun/`.
- `pytest tests/ -x` passes with no import errors.
- `python-magnetrun --help` works.
- `srvdata-to-magnetrun --help` works.

---

### 4.2 Restructure `panels/` → `python_magnetrun/dashboards/`

**Current state:** Two standalone scripts in `python_magnetrun/panels/`:
- `panel-mrecord.py`
- `panel-mrecord-vs-time.py`

Neither is importable as a module.

**Target structure:**

```
python_magnetrun/dashboards/
├── __init__.py
├── run_overview.py      # Time-series overview dashboard
├── field_analysis.py    # Field vs current dashboard
├── comparison.py        # Multi-run comparison dashboard
├── widgets.py           # Shared Panel widgets / helper functions
└── cli.py               # magnetrun-dashboard entry point
```

**Implementation notes:**
- Use `panel` and `hvplot` (already partially used in the old `panels/` scripts).
- All dashboard functions must be importable (no top-level `panel.serve()` at import time).
- Keep the old `panels/` scripts as thin wrappers that call into `dashboards/` for a release cycle, then remove them.

---

### 4.3 Implement `run_overview` dashboard

**File:** `python_magnetrun/dashboards/run_overview.py`

```python
"""
Time-series overview dashboard for a single MagnetRun.

Displays:
- Field strength vs time
- Currents IH, IB vs time
- Water flow rates vs time
- Temperatures vs time (if available)
- Regime annotations (U/P/D from Signature) overlaid

Usage:
    from python_magnetrun.dashboards.run_overview import RunOverviewDashboard
    dash = RunOverviewDashboard.from_file("M9_20230415.txt")
    dash.show()           # opens browser
    dash.servable()       # embed in panel serve
"""
from __future__ import annotations

import logging
import panel as pn
import hvplot.pandas  # noqa: F401  registers hvplot accessor
import pandas as pd

from python_magnetrun.MagnetRun import MagnetRun

logger = logging.getLogger(__name__)

pn.extension()


class RunOverviewDashboard:
    def __init__(self, run: MagnetRun) -> None:
        self._run = run
        self._mdata = run.getMData()
        self._df: pd.DataFrame = self._mdata.getData()
        self._panel = self._build()

    @classmethod
    def from_file(cls, path: str, *, housing: str = "") -> "RunOverviewDashboard":
        run = MagnetRun.fromtxt(path, housing=housing)
        run.prepareData()
        return cls(run)

    def _build(self) -> pn.viewable.Viewable:
        df = self._df

        # Time-range slider
        time_slider = pn.widgets.DatetimeRangeSlider(
            name="Time range",
            start=df.index.min(),
            end=df.index.max(),
            value=(df.index.min(), df.index.max()),
        )

        # Field selector
        available_keys = self._mdata.getKeys()
        field_select = pn.widgets.CheckBoxGroup(
            name="Channels",
            value=available_keys[:3],
            options=available_keys,
        )

        @pn.depends(time_slider, field_select)
        def _plot(time_range, fields):
            sub = df.loc[time_range[0]:time_range[1], fields]
            return sub.hvplot.line(
                responsive=True,
                height=400,
                title=f"Run: {self._run.getInsert()}",
            )

        return pn.Column(
            pn.Row(time_slider, field_select),
            pn.panel(_plot),
        )

    def show(self) -> None:
        """Open the dashboard in the default browser."""
        self._panel.show()

    def servable(self) -> pn.viewable.Viewable:
        """Return the servable Panel object for use with `panel serve`."""
        return self._panel.servable()
```

---

### 4.4 Implement `field_analysis` dashboard

**File:** `python_magnetrun/dashboards/field_analysis.py`

Focus: field strength vs current (IH, IB), with optional hysteresis overlay.

```python
class FieldAnalysisDashboard:
    """
    Dashboard for field vs current analysis.

    Optionally overlays hysteresis loops detected by
    python_magnetrun.processing.hysteresis.analysis.
    """
    def __init__(self, run: MagnetRun, *, show_hysteresis: bool = True) -> None:
        ...
```

---

### 4.5 Implement `comparison` dashboard

**File:** `python_magnetrun/dashboards/comparison.py`

Load multiple `MagnetRun` objects, overlay their field profiles normalized
to the same time axis, and show a statistics table.

```python
class ComparisonDashboard:
    """
    Multi-run comparison dashboard.

    Parameters
    ----------
    runs : list[MagnetRun]
        Runs to compare.
    normalize_time : bool
        If True, normalize all runs to [0, 1] time axis.
    """
    def __init__(self, runs: list[MagnetRun], *, normalize_time: bool = True) -> None:
        ...

    @classmethod
    def from_files(cls, paths: list[str], **kwargs) -> "ComparisonDashboard":
        from python_magnetrun.MagnetRun import MagnetRun
        runs = [MagnetRun.fromtxt(p) for p in paths]
        for r in runs:
            r.prepareData()
        return cls(runs, **kwargs)
```

---

### 4.6 Shared dashboard widgets

**File:** `python_magnetrun/dashboards/widgets.py`

Extract reusable widgets so dashboards don't duplicate code:

```python
def make_time_range_slider(df: pd.DataFrame) -> pn.widgets.DatetimeRangeSlider:
    """Return a DatetimeRangeSlider fitted to df's index."""
    ...

def make_channel_selector(
    keys: list[str],
    default: int = 3,
) -> pn.widgets.CheckBoxGroup:
    """Return a CheckBoxGroup pre-selecting the first `default` keys."""
    ...

def make_stats_table(runs: list[MagnetRun]) -> pn.widgets.Tabulator:
    """Return a Tabulator widget showing per-run statistics."""
    ...
```

---

### 4.7 Dashboard CLI entry point

**File:** `python_magnetrun/dashboards/cli.py`

```python
"""magnetrun-dashboard: serve an interactive dashboard from a data file."""
import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="magnetrun-dashboard")
    parser.add_argument("file", nargs="+", help="Data file(s) to load")
    parser.add_argument(
        "--dashboard",
        choices=["overview", "field", "compare"],
        default="overview",
        help="Dashboard type",
    )
    parser.add_argument("--housing", default="", help="Site housing (M8/M9/M10)")
    parser.add_argument("--port", type=int, default=5006)
    args = parser.parse_args(argv)

    try:
        import panel as pn
    except ImportError:
        print("panel is required: pip install python-magnetrun[dashboard]", file=sys.stderr)
        return 1

    if args.dashboard == "overview":
        from python_magnetrun.dashboards.run_overview import RunOverviewDashboard
        if len(args.file) != 1:
            parser.error("overview dashboard takes exactly one file")
        dash = RunOverviewDashboard.from_file(args.file[0], housing=args.housing)
        pn.serve(dash.servable(), port=args.port, show=True)

    elif args.dashboard == "compare":
        from python_magnetrun.dashboards.comparison import ComparisonDashboard
        dash = ComparisonDashboard.from_files(args.file)
        pn.serve(dash.servable(), port=args.port, show=True)

    return 0
```

Register the entry point:

```toml
[project.scripts]
magnetrun-dashboard = "python_magnetrun.dashboards.cli:main"
```

Add a `dashboard` optional dependency group:

```toml
[project.optional-dependencies]
dashboard = [
    "panel>=1.4",
    "hvplot>=0.9",
    "bokeh>=3.4",
]
```

**Acceptance criteria:**
- `magnetrun-dashboard --help` works without installing `panel`.
- `from python_magnetrun.dashboards.run_overview import RunOverviewDashboard` imports cleanly.
- With panel installed: `RunOverviewDashboard.from_file(sample_file)` constructs without error.

---

### 4.8 Jupyter notebook support

**New CLI command:** `magnetrun-to-notebook`

**File:** `python_magnetrun/notebooks/cli.py`

Generates a pre-filled `.ipynb` from a data file using `nbformat`.

```python
"""magnetrun-to-notebook: generate a Jupyter notebook from a data file."""
import argparse
from pathlib import Path
import nbformat as nbf


TEMPLATE_CELLS = [
    ("markdown", "# MagnetRun Analysis\n\nAuto-generated by `magnetrun-to-notebook`."),
    ("code", """\
from python_magnetrun import MagnetRun
run = MagnetRun.fromtxt("{filepath}", housing="{housing}")
run.prepareData()
mdata = run.getMData()
df = mdata.getData()
df.head()
"""),
    ("markdown", "## Field vs Time"),
    ("code", """\
import hvplot.pandas  # noqa
df[["B"]].hvplot.line(title="Field vs time", ylabel="B (T)")
"""),
    ("markdown", "## Statistics"),
    ("code", """\
run.getStats()
"""),
    ("markdown", "## Signal Processing"),
    ("code", """\
from python_magnetrun.processing.plateaux import detect_plateaus
plateaus = detect_plateaus(df)
plateaus
"""),
]


def build_notebook(filepath: str, housing: str) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells = []
    for kind, source in TEMPLATE_CELLS:
        src = source.format(filepath=filepath, housing=housing)
        if kind == "markdown":
            cells.append(nbf.v4.new_markdown_cell(src))
        else:
            cells.append(nbf.v4.new_code_cell(src))
    nb.cells = cells
    return nb


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="magnetrun-to-notebook")
    parser.add_argument("file", help="Input data file")
    parser.add_argument("-o", "--output", default=None, help="Output .ipynb path")
    parser.add_argument("--housing", default="", help="Site housing")
    args = parser.parse_args(argv)

    out_path = args.output or Path(args.file).with_suffix(".ipynb")
    nb = build_notebook(args.file, args.housing)

    with open(out_path, "w") as f:
        nbf.write(nb, f)

    print(f"Notebook written to {out_path}")
    return 0
```

Register:

```toml
[project.scripts]
magnetrun-to-notebook = "python_magnetrun.notebooks.cli:main"
```

Add to `dashboard` extras:

```toml
[project.optional-dependencies]
dashboard = [
    "panel>=1.4",
    "hvplot>=0.9",
    "bokeh>=3.4",
    "nbformat>=5.9",
]
```

**Acceptance criteria:**
- `magnetrun-to-notebook sample.txt -o out.ipynb` creates a valid notebook.
- `jupyter nbconvert --to notebook --execute out.ipynb` runs without errors
  (given the sample data file path is valid).

---

### 4.9 Add `__repr__` / `__str__` improvements

**File:** `python_magnetrun/MagnetRun.py`

Replace old `%r` formatting with f-strings:

```python
# Before
def __repr__(self):
    return "MagnetRun(%r, %r)" % (self.Housing, self.MagnetData.FileName)

# After
def __repr__(self) -> str:
    return f"MagnetRun(housing={self.Housing!r}, file={self.MagnetData.FileName!r})"
```

**File:** `python_magnetrun/magnetdata/core.py`

```python
def __repr__(self) -> str:
    return f"MagnetData(file={self.FileName!r}, type={self.Type.name}, keys={len(self.getKeys())})"
```

---

### 4.10 Add `bilan.py` and `flow_params.py` module docstrings

These scripts were repurposed as modules but lack explanations:

**`python_magnetrun/bilan.py`:**

```python
"""
Bilan (energy balance) computations for a MagnetRun.

This module computes electrical energy and heat dissipation from current
and voltage channels present in a MagnetRun data file.

Typical usage::

    from python_magnetrun.bilan import compute_bilan
    result = compute_bilan(mrun)
"""
```

**`python_magnetrun/flow_params.py`:**

```python
"""
Flow parameter extraction from MagnetRun files.

Extracts pump speed, flow rate, and pressure data from a MagnetRun,
filters it by minimum current, and prepares it for hydraulic curve fitting
via python_magnetcooling.

See also: python_magnetrun.waterflow_pipeline for the high-level API.
"""
```

---

## Files Modified / Created This Phase

| File | Action |
|------|--------|
| `python_magnetrun/requests/` | **Deleted** (entire directory) |
| `python_magnetrun/python_magnetrun.py` | **Deleted** |
| `python_magnetrun/magnetdata.py` | **Deleted** |
| `python_magnetrun/processing/hysteresis.py` | **Deleted** |
| `python_magnetrun/panels/` | Converted to thin wrappers, then deleted |
| `python_magnetrun/dashboards/__init__.py` | New |
| `python_magnetrun/dashboards/run_overview.py` | New |
| `python_magnetrun/dashboards/field_analysis.py` | New |
| `python_magnetrun/dashboards/comparison.py` | New |
| `python_magnetrun/dashboards/widgets.py` | New |
| `python_magnetrun/dashboards/cli.py` | New |
| `python_magnetrun/notebooks/__init__.py` | New |
| `python_magnetrun/notebooks/cli.py` | New |
| `python_magnetrun/MagnetRun.py` | `__repr__` f-string |
| `python_magnetrun/magnetdata/core.py` | `__repr__` f-string |
| `python_magnetrun/bilan.py` | Module docstring |
| `python_magnetrun/flow_params.py` | Module docstring |
| `pyproject.toml` | `dashboard` extras, new entry points |

---

## Verification Checklist

- [ ] `python -c "from python_magnetrun import MagnetData, MagnetRun, MRecord"` succeeds.
- [ ] `python -c "from python_magnetrun.fetchers.connect import connect"` succeeds.
- [ ] `python -c "from python_magnetrun.requests import connect"` raises `ModuleNotFoundError` (shim is gone).
- [ ] `python -c "import python_magnetrun.python_magnetrun"` raises `ModuleNotFoundError`.
- [ ] `python-magnetrun --help` works.
- [ ] `srvdata-to-magnetrun --help` works.
- [ ] `magnetrun-dashboard --help` works (without panel installed: prints error and exits 1).
- [ ] `magnetrun-to-notebook sample.txt` produces a valid `.ipynb`.
- [ ] `from python_magnetrun.dashboards.run_overview import RunOverviewDashboard` imports cleanly.
- [ ] `pytest tests/ -x` — all tests pass.
- [ ] `ruff check python_magnetrun/` exits 0.
- [ ] `mypy python_magnetrun/ --ignore-missing-imports` exits 0.
- [ ] No `print()` in library code — logging only (re-verify after shim removal).
- [ ] No absolute `/home/...` developer paths anywhere in source.
- [ ] `grep -rn "from python_magnetrun.requests\b" python_magnetrun/` → zero hits.

---

## Dependencies

- **Requires Phase 3** complete: Pydantic models, API client, YAML config, all tests passing.
- This is the final phase — no further phases depend on it.

---

## Post-Phase 4: Ongoing Maintenance Checklist

Once Phase 4 is done, the following practices should be maintained:

1. **Adding a new site** → edit `data/sites.yaml` only.
2. **Adding a new file format** → create a loader class implementing `DataLoader` protocol, register in `_LOADER_REGISTRY`. No changes to `MagnetData`.
3. **Adding a new processing algorithm** → add a function decorated with `@register_smoother` or `@register_stat`. No changes to CLI dispatch.
4. **Adding a new dashboard** → add a module in `dashboards/`, register the `--dashboard` choice in `dashboards/cli.py`.
5. **Releasing a new version** → run `ruff check`, `mypy`, `pytest`, then `git tag vX.Y.Z`.
