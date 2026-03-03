# Prompt: Migrate `python_magnetrun` to Use `python_magnetcooling.fitting` and Remove Cooling Code

## Objective

Update `python_magnetrun` to delegate all hydraulic fitting and cooling computations to `python_magnetcooling`. This involves three changes:

1. **Rewrite `flow_params.py`** to extract data from MagnetRun objects and pass clean arrays to `python_magnetcooling.fitting`.
2. **Remove `python_magnetrun/cooling/`** — all thermal-hydraulic code now lives in `python_magnetcooling`.
3. **Update `pyproject.toml`** to declare `python_magnetcooling` as an optional dependency and remove dependencies that have moved to that package.

---

## Prerequisites

- `python_magnetcooling` **version ≥ 0.2.0** (or whichever version includes the `fitting` module) must be implemented and installable. Specifically, the following must be available:

```python
from python_magnetcooling.fitting import (
    PumpSpeedFit,
    FlowPressureFit,
    fit_pump_speed_simple,
    fit_pump_speed_piecewise,
    fit_flow_rate,
    fit_pressure,
    compute_back_pressure_stats,
    fit_hydraulic_system,
    build_waterflow,
)
from python_magnetcooling.waterflow import WaterFlow
from python_magnetcooling.waterflow_factory import from_flow_params, from_fits
```

---

## Part 1: Rewrite `flow_params.py`

### Current State

`python_magnetrun/examples/flow_params.py` is a ~400-line script that:

1. Queries a database (via `python_magnetdb`) for magnet records
2. Loads MagnetRun files for each record
3. Extracts DataFrames with columns: current (`Ikey`), pump speed (`RpmKey`), flow rate (`QKey`), inlet pressure (`PinKey`), back pressure (`PoutKey`)
4. Filters data (I ≥ 300 A), detects plateaus, adjusts Imax
5. Defines inline fit functions (`vpump_func`, `flow_func`, `pressure_func`)
6. Calls a local `fit()` helper that wraps `scipy.optimize.curve_fit`
7. Builds a `flow_params` dict
8. Saves to JSON

Additionally, `examples/flow_params_pipeline.py` and `examples/flow_params_magnetrun_pipeline.py` are standalone demonstration scripts that do the same thing with synthetic data.

### Target State

The fitting logic (steps 5–7) is replaced by calls to `python_magnetcooling.fitting`. The data extraction logic (steps 1–4) stays in `python_magnetrun`.

### New File: `python_magnetrun/waterflow_pipeline.py`

Create a new module that replaces the fitting portions of `flow_params.py`. This is importable library code, not just a script.

```python
"""
Waterflow pipeline: extract hydraulic data from MagnetRun files
and fit pump curves using python_magnetcooling.

This module handles the data-extraction side. The actual curve fitting
is delegated to python_magnetcooling.fitting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class HydraulicData:
    """
    Clean hydraulic arrays extracted from a MagnetRun DataFrame.

    All arrays are filtered (e.g., I ≥ threshold) and aligned.

    Attributes
    ----------
    current : np.ndarray
        Current [A].
    pump_speed : np.ndarray
        Pump speed [rpm].
    flow_rate : np.ndarray
        Flow rate [l/s].
    pressure : np.ndarray
        Inlet pressure [bar].
    back_pressure : np.ndarray
        Back (outlet) pressure [bar].
    imax : float or None
        Known Imax if available (from plateaus or prior knowledge).
    name : str
        Identifier (e.g., "M9_M10" or magnet name).
    """
    current: np.ndarray
    pump_speed: np.ndarray
    flow_rate: np.ndarray
    pressure: np.ndarray
    back_pressure: np.ndarray
    imax: Optional[float] = None
    name: str = ""


def extract_hydraulic_data(
    df: pd.DataFrame,
    current_col: str,
    rpm_col: str,
    flow_col: str,
    pressure_in_col: str,
    pressure_out_col: str,
    current_threshold: float = 300.0,
    name: str = "",
) -> HydraulicData:
    """
    Extract and filter hydraulic arrays from a MagnetRun DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from MagnetRun (via getMData().getData() or similar).
    current_col : str
        Column name for current (e.g., "IH", "Icoil", site-dependent).
    rpm_col : str
        Column name for pump speed (e.g., "Rpm").
    flow_col : str
        Column name for flow rate (e.g., "Flow", "FlowH").
    pressure_in_col : str
        Column name for inlet pressure (e.g., "HP", "HPH").
    pressure_out_col : str
        Column name for back pressure (e.g., "BP", "BPH").
    current_threshold : float
        Minimum current for filtering (default 300 A).
    name : str
        Identifier for this dataset.

    Returns
    -------
    HydraulicData
        Filtered and aligned arrays ready for fitting.

    Raises
    ------
    KeyError
        If any column is missing from the DataFrame.
    ValueError
        If fewer than 3 data points remain after filtering.
    """
    required = [current_col, rpm_col, flow_col, pressure_in_col, pressure_out_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in DataFrame: {missing}")

    mask = df[current_col] >= current_threshold
    filtered = df.loc[mask]

    if len(filtered) < 3:
        raise ValueError(
            f"Only {len(filtered)} points after filtering "
            f"{current_col} >= {current_threshold}. Need at least 3."
        )

    logger.info(
        "Extracted %d points from %d total (threshold=%.0f A)",
        len(filtered), len(df), current_threshold,
    )

    return HydraulicData(
        current=filtered[current_col].to_numpy(),
        pump_speed=filtered[rpm_col].to_numpy(),
        flow_rate=filtered[flow_col].to_numpy(),
        pressure=filtered[pressure_in_col].to_numpy(),
        back_pressure=filtered[pressure_out_col].to_numpy(),
        name=name,
    )


def detect_imax_from_plateaus(
    data: HydraulicData,
    plateau_threshold: float = 0.01,
) -> Optional[float]:
    """
    Detect Imax by identifying plateau regions in the pump speed curve.

    A plateau is a region where d(Vp)/d(I) ≈ 0, indicating the pump
    has reached maximum speed and current is still increasing.

    Parameters
    ----------
    data : HydraulicData
        Extracted hydraulic data.
    plateau_threshold : float
        Relative derivative threshold for plateau detection.

    Returns
    -------
    float or None
        Detected Imax, or None if no plateau found.
    """
    # Implementation: compute numerical derivative of pump_speed w.r.t. current,
    # identify regions where the derivative is near zero relative to the
    # overall range. Return the current at the onset of the first plateau.
    ...


def compute_waterflow(
    data: HydraulicData,
    method: str = "simple",
) -> "WaterFlow":
    """
    Full pipeline: extract → fit → build WaterFlow.

    This is the main convenience function. It calls into
    python_magnetcooling.fitting for the actual curve fitting.

    Parameters
    ----------
    data : HydraulicData
        Cleaned hydraulic arrays (from extract_hydraulic_data).
    method : str
        "simple" for scipy quadratic fits (requires data.imax to be set),
        "piecewise" for pwlf with automatic Imax detection.

    Returns
    -------
    WaterFlow
        Fully configured WaterFlow object.

    Raises
    ------
    ImportError
        If python_magnetcooling is not installed.
    ValueError
        If method="simple" and data.imax is None.
    """
    try:
        from python_magnetcooling.fitting import (
            fit_hydraulic_system,
            build_waterflow,
        )
    except ImportError:
        raise ImportError(
            "python_magnetcooling is required for hydraulic fitting. "
            "Install it with: pip install python-magnetcooling[fitting]"
        )

    if method == "simple" and data.imax is None:
        raise ValueError(
            "imax must be set in HydraulicData for method='simple'. "
            "Either set it manually, call detect_imax_from_plateaus(), "
            "or use method='piecewise' for automatic detection."
        )

    pump_fit, flow_pressure_fit = fit_hydraulic_system(
        current=data.current,
        pump_speed=data.pump_speed,
        flow_rate=data.flow_rate,
        pressure=data.pressure,
        back_pressure=data.back_pressure,
        imax=data.imax,
        method=method,
    )

    logger.info(
        "Fitted %s: Imax=%.0f A, Vpmax=%.1f rpm, Fmax=%.1f l/s, Pmax=%.1f bar",
        data.name, pump_fit.imax, pump_fit.vpmax,
        flow_pressure_fit.fmax, flow_pressure_fit.pmax,
    )

    return build_waterflow(pump_fit, flow_pressure_fit)


def compute_waterflow_from_run(
    mrun,  # MagnetRun — not typed to avoid circular import at module level
    current_col: str,
    rpm_col: str,
    flow_col: str,
    pressure_in_col: str,
    pressure_out_col: str,
    method: str = "simple",
    imax: Optional[float] = None,
    current_threshold: float = 300.0,
) -> "WaterFlow":
    """
    End-to-end: MagnetRun → HydraulicData → WaterFlow.

    Convenience function that combines data extraction and fitting
    in a single call.

    Parameters
    ----------
    mrun : MagnetRun
        A loaded MagnetRun object.
    current_col, rpm_col, flow_col, pressure_in_col, pressure_out_col : str
        Column names in the MagnetRun data (site-dependent).
    method : str
        Fitting method ("simple" or "piecewise").
    imax : float, optional
        Known Imax. If None and method="piecewise", will be auto-detected.
    current_threshold : float
        Minimum current for filtering.

    Returns
    -------
    WaterFlow
    """
    # Get DataFrame from MagnetRun
    mdata = mrun.getMData()
    df = mdata.getData()

    name = mrun.getInsert() if hasattr(mrun, "getInsert") else ""

    data = extract_hydraulic_data(
        df=df,
        current_col=current_col,
        rpm_col=rpm_col,
        flow_col=flow_col,
        pressure_in_col=pressure_in_col,
        pressure_out_col=pressure_out_col,
        current_threshold=current_threshold,
        name=name,
    )

    if imax is not None:
        data = HydraulicData(
            current=data.current,
            pump_speed=data.pump_speed,
            flow_rate=data.flow_rate,
            pressure=data.pressure,
            back_pressure=data.back_pressure,
            imax=imax,
            name=data.name,
        )

    return compute_waterflow(data, method=method)
```

### What Happens to the Existing Files

| File | Action |
|------|--------|
| `python_magnetrun/examples/flow_params.py` | Keep as-is for now. It queries the database and orchestrates multiple magnet records. Refactor it to call `compute_waterflow_from_run()` instead of inline fitting. This is a second-pass task since it depends on `python_magnetdb`. |
| `examples/flow_params_pipeline.py` | Update to import from `python_magnetcooling.fitting` directly (it uses synthetic data, no MagnetRun dependency). |
| `examples/flow_params_magnetrun_pipeline.py` | Update to use `extract_hydraulic_data()` + `compute_waterflow()`. |
| `examples/flow_params_magnetrun.py` | Same treatment — replace inline pwlf calls with `compute_waterflow()`. |

---

## Part 2: Remove `python_magnetrun/cooling/`

### Files to Delete

```
python_magnetrun/cooling/__init__.py
python_magnetrun/cooling/heatexchanger_primary.py
python_magnetrun/cooling/heatexchanger_primary_orig.py
python_magnetrun/cooling/water.py
```

### Pre-Deletion Checks

Before deleting, verify:

1. **No internal imports depend on these files.** Search for:
   ```bash
   grep -r "from.*cooling" python_magnetrun/ --include="*.py"
   grep -r "import.*cooling" python_magnetrun/ --include="*.py"
   ```
   Exclude `python_magnetcooling` hits if the packages share a workspace.

2. **No entry points reference them.** Check `pyproject.toml` `[project.scripts]` — none currently point to `cooling/`.

3. **The examples that used these modules are redirected.** The heat exchanger examples (`examples/heatexchanger_primary.py`) already exist as standalone scripts that import from `MagnetRun` and do their own analysis. These should be updated to use `python_magnetcooling` for the thermal-hydraulic computations, or moved to the `python_magnetcooling` examples if they don't depend on MagnetRun at all.

### Heat Exchanger Example Migration

`examples/heatexchanger_primary.py` loads a MagnetRun file and performs heat exchanger analysis using cooling functions. After the migration:

- Data loading (MagnetRun, adding computed columns like "Flow", "Tin", "HP") stays in the example script.
- Thermal-hydraulic computations (`steam()`, `getDT()`, `getHeatCoeff()`, `getTout()`) are replaced by calls to `python_magnetcooling`:

```python
# Before (old cooling imports)
from python_magnetrun.cooling.water import steam, getDT

# After
from python_magnetcooling.water_properties import WaterProperties
from python_magnetcooling import compute_single_channel
```

If the example becomes primarily a `python_magnetcooling` consumer with a thin MagnetRun data-loading preamble, consider moving it to `python_magnetcooling/examples/` with a note that it requires `python_magnetrun` for data loading.

---

## Part 3: Update `pyproject.toml`

### Add `python_magnetcooling` as an Optional Dependency

```toml
[project.optional-dependencies]
cooling = [
    "python-magnetcooling>=0.2.0",
]
```

### Remove Dependencies That Have Moved

The following dependencies are only used by the cooling code and can be removed from the core `dependencies` list:

```toml
# REMOVE from core dependencies:
"iapws>=1.3.4",      # → lives in python_magnetcooling
"nlopt>=2.7.0",       # → used by cooling optimization, not by magnetrun core
```

Also remove from `[project.optional-dependencies] system`:

```toml
# REMOVE from system extras:
"ht>=0.1.55",         # → lives in python_magnetcooling
```

Verify by searching for remaining imports:

```bash
# After removing cooling/, these should return no hits in python_magnetrun/
grep -r "import iapws" python_magnetrun/ --include="*.py"
grep -r "import nlopt" python_magnetrun/ --include="*.py"
grep -r "import ht" python_magnetrun/ --include="*.py"
grep -r "from iapws" python_magnetrun/ --include="*.py"
```

If any non-cooling module still imports these, keep the dependency.

### Update the `debian/control` Build-Depends

Remove `python3-iapws`, `python3-freesteam`, `python3-ht`, `python3-nlopt` from `Build-Depends` and add `python3-magnetcooling` as a dependency of `python3-magnetrun`:

```
Package: python3-magnetrun
Architecture: all
Depends: python3-magnetcooling, python3-statsmodels, ${python3:Depends}, ${misc:Depends}
```

---

## Part 4: Update Console Scripts

### Current Entry Points to Review

```toml
[project.scripts]
python-magnetrun = "python_magnetrun.python_magnetrun:main"
srvdata-to-magnetrun = "python_magnetrun.requests.cli:main"
magnetrun-analysis = "python_magnetrun.analysis:main"
hybrid-magnetrun = "python_magnetrun.hybrid.cli:main"
magnetrun-alimconfig = "python_magnetrun.configAlims.convertxml:main"
magnetrun-pigbrother-logparser = "python_magnetrun.tdms.log_parser:main"
```

None of these point to `cooling/`, so no changes needed. However, consider adding:

```toml
magnetrun-waterflow = "python_magnetrun.waterflow_pipeline:main"
```

if a CLI entry point for the waterflow fitting pipeline is desired. This would accept a MagnetRun file and column names, run the fitting, and output a JSON file.

---

## Tests

### New Tests: `tests/test_waterflow_pipeline.py`

```python
"""Tests for the waterflow pipeline (data extraction side)."""

import pytest
import numpy as np
import pandas as pd


class TestExtractHydraulicData:
    """Test data extraction from DataFrames."""

    def test_basic_extraction(self):
        """Verify arrays are correctly extracted and filtered."""
        df = pd.DataFrame({
            "I": [100, 200, 500, 1000, 5000],
            "Rpm": [1000, 1010, 1050, 1100, 1500],
            "Flow": [10, 11, 15, 20, 50],
            "Pin": [5, 5.1, 6, 8, 15],
            "Pout": [4, 4, 4, 4, 4],
        })
        from python_magnetrun.waterflow_pipeline import extract_hydraulic_data
        data = extract_hydraulic_data(
            df, "I", "Rpm", "Flow", "Pin", "Pout",
            current_threshold=300.0,
        )
        assert len(data.current) == 3  # 500, 1000, 5000
        assert data.current[0] == 500.0

    def test_missing_column_raises(self):
        """Verify KeyError for missing columns."""
        df = pd.DataFrame({"I": [1, 2], "Rpm": [3, 4]})
        from python_magnetrun.waterflow_pipeline import extract_hydraulic_data
        with pytest.raises(KeyError, match="Missing columns"):
            extract_hydraulic_data(df, "I", "Rpm", "Flow", "Pin", "Pout")

    def test_insufficient_data_raises(self):
        """Verify ValueError when too few points after filtering."""
        df = pd.DataFrame({
            "I": [100, 200],
            "Rpm": [1000, 1010],
            "Flow": [10, 11],
            "Pin": [5, 5.1],
            "Pout": [4, 4],
        })
        from python_magnetrun.waterflow_pipeline import extract_hydraulic_data
        with pytest.raises(ValueError, match="at least 3"):
            extract_hydraulic_data(
                df, "I", "Rpm", "Flow", "Pin", "Pout",
                current_threshold=300.0,
            )


class TestComputeWaterflow:
    """Test the full pipeline (requires python_magnetcooling)."""

    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic HydraulicData with known parameters."""
        from python_magnetrun.waterflow_pipeline import HydraulicData

        n = 200
        current = np.linspace(300, 28000, n)
        imax = 28000.0
        vpmax, vp0 = 2840.0, 1000.0

        vp = vpmax * (current / imax) ** 2 + vp0
        vp_ratio = vp / (vpmax + vp0)
        flow = 0.0 + 140.0 * vp_ratio
        pressure = 4.0 + 22.0 * vp_ratio ** 2
        back_pressure = np.full(n, 4.0)

        return HydraulicData(
            current=current,
            pump_speed=vp,
            flow_rate=flow,
            pressure=pressure,
            back_pressure=back_pressure,
            imax=imax,
            name="synthetic",
        )

    def test_simple_method(self, synthetic_data):
        """End-to-end with simple fitting."""
        pytest.importorskip("python_magnetcooling")
        from python_magnetrun.waterflow_pipeline import compute_waterflow

        wf = compute_waterflow(synthetic_data, method="simple")
        assert abs(wf.current_max - 28000) < 1
        assert abs(wf.pump_speed_max - 2840) < 50
        assert wf.flow_rate(20000) > 0

    def test_missing_imax_raises(self, synthetic_data):
        """Verify ValueError when imax is None with simple method."""
        from python_magnetrun.waterflow_pipeline import HydraulicData, compute_waterflow

        data_no_imax = HydraulicData(
            current=synthetic_data.current,
            pump_speed=synthetic_data.pump_speed,
            flow_rate=synthetic_data.flow_rate,
            pressure=synthetic_data.pressure,
            back_pressure=synthetic_data.back_pressure,
            imax=None,
        )
        with pytest.raises(ValueError, match="imax must be set"):
            compute_waterflow(data_no_imax, method="simple")

    def test_missing_magnetcooling_raises(self, monkeypatch):
        """Verify ImportError when python_magnetcooling not installed."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "python_magnetcooling" in name:
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        from python_magnetrun.waterflow_pipeline import HydraulicData, compute_waterflow

        data = HydraulicData(
            current=np.array([1, 2, 3]),
            pump_speed=np.array([1, 2, 3]),
            flow_rate=np.array([1, 2, 3]),
            pressure=np.array([1, 2, 3]),
            back_pressure=np.array([1, 2, 3]),
            imax=100,
        )
        with pytest.raises(ImportError, match="python_magnetcooling"):
            compute_waterflow(data, method="simple")
```

### Deletion Tests

After removing `python_magnetrun/cooling/`, verify:

```bash
# No broken imports
python -c "import python_magnetrun; print('OK')"
python -c "from python_magnetrun.analysis import AnalysisConfig; print('OK')"
python -c "from python_magnetrun.hybrid import HybridData; print('OK')"

# Existing test suite passes
pytest tests/ -x --ignore=tests/test_cooling.py  # if cooling tests exist, skip them
```

---

## Migration Order

Execute these steps in order, committing after each:

### Commit 1: Add `waterflow_pipeline.py`

- Create `python_magnetrun/waterflow_pipeline.py` with `HydraulicData`, `extract_hydraulic_data()`, `compute_waterflow()`, `compute_waterflow_from_run()`.
- Create `tests/test_waterflow_pipeline.py`.
- All existing tests still pass (no files removed yet).

### Commit 2: Update examples

- Update `examples/flow_params_pipeline.py` to use `python_magnetcooling.fitting` directly.
- Update `examples/flow_params_magnetrun_pipeline.py` to use `extract_hydraulic_data()` + `compute_waterflow()`.
- Update `examples/heatexchanger_primary.py` to use `python_magnetcooling` for thermal computations.

### Commit 3: Remove `python_magnetrun/cooling/`

- Delete the directory.
- Update `python_magnetrun/__init__.py` if it imports from `cooling`.
- Remove `iapws`, `nlopt` from core dependencies (if confirmed unused elsewhere).
- Update `debian/control`.

### Commit 4: Update `pyproject.toml`

- Add `cooling` optional dependency group.
- Clean up dependency list.
- Add `magnetrun-waterflow` entry point if desired.

---

## Verification Checklist

- [ ] `waterflow_pipeline.py` created with all functions
- [ ] `extract_hydraulic_data()` works with real MagnetRun DataFrames
- [ ] `compute_waterflow()` correctly delegates to `python_magnetcooling.fitting`
- [ ] `compute_waterflow_from_run()` works end-to-end with a MagnetRun object
- [ ] `python_magnetrun/cooling/` directory deleted
- [ ] No remaining imports from `python_magnetrun.cooling` anywhere in `python_magnetrun/`
- [ ] `iapws`, `nlopt`, `ht` removed from core dependencies (if unused)
- [ ] `python_magnetcooling` added as optional dependency
- [ ] `debian/control` updated
- [ ] All existing tests pass (analysis, hybrid, etc.)
- [ ] `tests/test_waterflow_pipeline.py` passes
- [ ] Examples updated and working
- [ ] No `print()` in library code — logging only
- [ ] All new functions have type hints and NumPy-style docstrings
