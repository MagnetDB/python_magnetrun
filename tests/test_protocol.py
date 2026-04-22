"""Protocol compliance tests for DataLoader.

Verifies that MagnetRun, HybridRun, SimulationRun, and BFieldRun structurally
satisfy the DataLoader protocol via Python's runtime_checkable isinstance() check.
"""

import pandas as pd
import pytest

from python_magnetrun.bfield.bfield_run import BFieldRun
from python_magnetrun.hybrid.data_protocol import DataLoader
from python_magnetrun.hybrid.hybrid_run import HybridRun
from python_magnetrun.MagnetRun import MagnetRun
from python_magnetrun.simulation.simulation_run import SimulationRun


def test_magnetrun_satisfies_protocol():
    mrun = MagnetRun()
    assert isinstance(mrun, DataLoader), (
        "MagnetRun does not satisfy the DataLoader protocol — "
        f"missing: {[m for m in ('getData','getKeys','getType','getSite','getHousing','getDomain','get_time_range') if not hasattr(mrun, m)]}"
    )


def test_hybridrun_satisfies_protocol():
    hrun = HybridRun()
    assert isinstance(hrun, DataLoader), (
        "HybridRun does not satisfy the DataLoader protocol — "
        f"missing: {[m for m in ('getData','getKeys','getType','getSite','getHousing','getDomain','get_time_range') if not hasattr(hrun, m)]}"
    )


def test_magnetrun_getdomain():
    assert MagnetRun().getDomain() == "operational"


def test_hybridrun_getdomain():
    assert HybridRun().getDomain() == "operational"


def test_dataloader_required_methods():
    required = {"getData", "getKeys", "getType", "getSite", "getHousing", "getDomain", "get_time_range"}
    protocol_attrs = set(m for m in dir(DataLoader) if not m.startswith("_"))
    missing = required - protocol_attrs
    assert not missing, f"DataLoader protocol is missing: {missing}"


# ---------------------------------------------------------------------------
# Helpers — build minimal synthetic data objects without loading real files
# ---------------------------------------------------------------------------

def _make_pandas_magnetdata(columns: list[str] | None = None):
    """Return a minimal PandasMagnetData with a synthetic DataFrame."""
    from python_magnetrun.magnetdata_pandas import PandasMagnetData

    cols = columns or ["IH", "IB", "Field"]
    df = pd.DataFrame({c: [1.0, 2.0] for c in cols})
    return PandasMagnetData("synthetic", {}, cols, df)


def _make_bprofile_magnetdata():
    """Return a minimal BProfileMagnetData with a synthetic DataFrame."""
    from python_magnetrun.magnetdata_pandas import BProfileMagnetData

    df = pd.DataFrame({"Index": [0, 1], "Position": [0.0, 1.0], "Profile": [1.0, 1.1]})
    return BProfileMagnetData("synthetic", {}, list(df.columns), df)


# ---------------------------------------------------------------------------
# SimulationRun
# ---------------------------------------------------------------------------

def test_simulation_run_satisfies_protocol():
    data = _make_pandas_magnetdata()
    srun = SimulationRun(data, housing="M8", site="grenoble")
    assert isinstance(srun, DataLoader), (
        "SimulationRun does not satisfy the DataLoader protocol — "
        f"missing: {[m for m in ('getData','getKeys','getType','getSite','getHousing','getDomain','get_time_range') if not hasattr(srun, m)]}"
    )


def test_simulation_run_getdomain():
    data = _make_pandas_magnetdata()
    assert SimulationRun(data).getDomain() == "simulation"


def test_simulation_run_getkeys():
    data = _make_pandas_magnetdata(["IH", "IB"])
    srun = SimulationRun(data)
    assert set(srun.getKeys()) == {"IH", "IB"}


def test_simulation_run_gethousing_getsite():
    data = _make_pandas_magnetdata()
    srun = SimulationRun(data, housing="M9", site="saclay")
    assert srun.getHousing() == "M9"
    assert srun.getSite() == "saclay"


def test_simulation_run_get_time_range_no_column():
    data = _make_pandas_magnetdata()
    srun = SimulationRun(data)
    with pytest.raises(NotImplementedError):
        srun.get_time_range()


def test_simulation_run_is_transient_false():
    data = _make_pandas_magnetdata()
    assert SimulationRun(data).is_transient() is False


def test_simulation_run_is_transient_true():
    data = _make_pandas_magnetdata(["t", "IH"])
    assert SimulationRun(data, time_column="t").is_transient() is True


# ---------------------------------------------------------------------------
# BFieldRun
# ---------------------------------------------------------------------------

def test_bfield_run_satisfies_protocol():
    data = _make_bprofile_magnetdata()
    brun = BFieldRun(data, housing="M8", site="grenoble")
    assert isinstance(brun, DataLoader), (
        "BFieldRun does not satisfy the DataLoader protocol — "
        f"missing: {[m for m in ('getData','getKeys','getType','getSite','getHousing','getDomain','get_time_range') if not hasattr(brun, m)]}"
    )


def test_bfield_run_getdomain():
    data = _make_bprofile_magnetdata()
    assert BFieldRun(data).getDomain() == "bfield"


def test_bfield_run_getkeys():
    data = _make_bprofile_magnetdata()
    brun = BFieldRun(data)
    assert "Position" in brun.getKeys()
    assert "Profile" in brun.getKeys()


def test_bfield_run_get_time_range_no_timestamp():
    data = _make_bprofile_magnetdata()
    brun = BFieldRun(data)
    with pytest.raises(NotImplementedError):
        brun.get_time_range()


def test_bfield_run_get_time_range_with_timestamp():
    from datetime import datetime

    data = _make_bprofile_magnetdata()
    t = datetime(2025, 11, 5, 9, 53, 0)
    brun = BFieldRun(data, measurement_time=t)
    start, end = brun.get_time_range()
    assert start == t
    assert end == t
