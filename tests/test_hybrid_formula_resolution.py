"""Tests for HybridRun.getData formula key resolution.

Covers the fix where keys defined in hybrid_formula_map (e.g.
"FEPC-AUX-LNCMI/ALIM1") are evaluated by loading their constituent kHz
channels and summing, rather than failing with ValueError.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from python_magnetrun.hybrid.hybrid_run import HybridRun, LoadOptions

pytest.importorskip("natsort")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FORMULA_MAP = {
    "FEPC-AUX-LNCMI/ALIM1": {
        "formula": "FEPC-AUX-LNCMI/ALIM1 = FEPC-AUX-LNCMI/ALIM1_J1 + FEPC-AUX-LNCMI/ALIM1_J2",
        "symbol": "I_ALIM1",
        "unit": "ampere",
        "label": "ALIM1 Current",
        "description": "ALIM1 current sum",
    }
}


def _make_hrun() -> HybridRun:
    """Return a HybridRun with a mock HybridData and Housing set."""
    hrun = object.__new__(HybridRun)
    hrun.Housing = "M8"
    hrun._cache = {}
    hrun._cache_size_bytes = 0
    hrun._cache_max_size_bytes = 256 * 1024 * 1024  # 256 MB
    hrun.default_options = LoadOptions(lazy=False, cache=False)
    hrun.HybridData = MagicMock()
    return hrun


def _fake_read_khz(system, variable, hours=None, apply_calib=True, cnv_dir=None):
    """Return deterministic arrays based on variable name."""
    n = 100
    t = np.linspace(0, 1, n)
    if variable == "ALIM1_J1":
        return np.ones(n) * 10.0, t
    if variable == "ALIM1_J2":
        return np.ones(n) * 5.0, t
    raise KeyError(f"Unknown variable {variable!r}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_resolve_formula_two_operands():
    """getData with a formula key returns the element-wise sum of J1+J2."""
    hrun = _make_hrun()
    hrun.HybridData.read_khz_variable.side_effect = _fake_read_khz

    mock_cfg = MagicMock()
    mock_cfg.hybrid_formula_map = FORMULA_MAP
    mock_cfg.hybrid_voltage_mask_map = {}

    with patch(
        "python_magnetrun.housing_config.get_housing_config", return_value=mock_cfg
    ):
        data, time = hrun.getData("FEPC-AUX-LNCMI/ALIM1")

    assert data.shape == (100,)
    np.testing.assert_allclose(data, 15.0)  # 10 + 5
    assert time.shape == (100,)


def test_resolve_formula_cache_hit():
    """Second call with the same formula key hits the cache (no re-read)."""
    hrun = _make_hrun()
    hrun.default_options = LoadOptions(lazy=False, cache=True)
    hrun.HybridData.read_khz_variable.side_effect = _fake_read_khz

    mock_cfg = MagicMock()
    mock_cfg.hybrid_formula_map = FORMULA_MAP
    mock_cfg.hybrid_voltage_mask_map = {}

    with patch(
        "python_magnetrun.housing_config.get_housing_config", return_value=mock_cfg
    ):
        hrun.getData("FEPC-AUX-LNCMI/ALIM1")
        hrun.getData("FEPC-AUX-LNCMI/ALIM1")

    # J1 and J2 each called once only (second outer call hits cache)
    assert hrun.HybridData.read_khz_variable.call_count == 2


def test_resolve_formula_shape_mismatch():
    """Operands with different lengths raise ValueError."""
    hrun = _make_hrun()

    def _mismatched(system, variable, **kw):
        if variable == "ALIM1_J1":
            return np.ones(100), np.linspace(0, 1, 100)
        return np.ones(50), np.linspace(0, 0.5, 50)

    hrun.HybridData.read_khz_variable.side_effect = _mismatched

    mock_cfg = MagicMock()
    mock_cfg.hybrid_formula_map = FORMULA_MAP
    mock_cfg.hybrid_voltage_mask_map = {}

    with (
        patch(
            "python_magnetrun.housing_config.get_housing_config", return_value=mock_cfg
        ),
        pytest.raises(ValueError, match="shape mismatch"),
    ):
        hrun.getData("FEPC-AUX-LNCMI/ALIM1")


def test_getData_normal_key_unaffected():
    """A normal kHz key is not intercepted by the formula resolver."""
    hrun = _make_hrun()
    hrun.HybridData.read_khz_variable.return_value = (np.ones(10), np.arange(10))

    mock_cfg = MagicMock()
    mock_cfg.hybrid_formula_map = {}  # no formula keys
    mock_cfg.hybrid_voltage_mask_map = {}

    with patch(
        "python_magnetrun.housing_config.get_housing_config", return_value=mock_cfg
    ):
        data, time = hrun.getData("kHz/FEPC-AUX-LNCMI/ALIM1_J1")

    assert data.shape == (10,)
    hrun.HybridData.read_khz_variable.assert_called_once_with(
        "FEPC-AUX-LNCMI", "ALIM1_J1", hours=None, apply_calib=True, cnv_dir=None
    )


def test_getData_unknown_key_raises():
    """A key that is neither a formula key nor a valid type/... raises ValueError."""
    hrun = _make_hrun()

    mock_cfg = MagicMock()
    mock_cfg.hybrid_formula_map = {}
    mock_cfg.hybrid_voltage_mask_map = {}

    with (
        patch(
            "python_magnetrun.housing_config.get_housing_config", return_value=mock_cfg
        ),
        pytest.raises(ValueError, match="Unknown data type"),
    ):
        hrun.getData("FEPC-AUX-LNCMI/ALIM1")
