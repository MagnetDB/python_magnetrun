"""flow_params.compute: pump law from a 2-segment piecewise fit."""

import json

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

pytest.importorskip("pwlf")

from python_magnetrun.flow_params import compute  # noqa: E402

VP0, VPMAX, IMAX = 1000.0, 1800.0, 27000.0  # rpm, rpm (rise above Vp0), A


def _hydraulics(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Synthetic pupitre hydraulics: ramp up to 1.2 Imax and back down, in time order."""
    rng = np.random.default_rng(seed)
    up = np.linspace(300.0, 1.2 * IMAX, n // 2)
    current = np.concatenate([up, up[::-1]])
    vp = VPMAX * (np.minimum(current, IMAX) / IMAX) ** 2 + VP0
    ratio = vp / (VPMAX + VP0)
    return pd.DataFrame(
        {
            "I": current,
            "Rpm": vp + rng.normal(0.0, 0.5, current.size),
            "Q": 130.0 * ratio + rng.normal(0.0, 0.2, current.size),
            "HP": 4.0 + 19.0 * ratio**2 + rng.normal(0.0, 0.02, current.size),
            "BP": 4.0 + rng.normal(0.0, 0.05, current.size),
        }
    )


def test_compute_vpmax_is_rise_above_vp0(tmp_path, monkeypatch):
    """Vpmax has the WaterFlow meaning (Vp = Vpmax (I/Imax)^2 + Vp0), not Vp(Imax).

    The samples are in time order: the current ramps up then down, so the
    last sample is at low current.
    """
    monkeypatch.chdir(tmp_path)
    compute(_hydraulics(), "I", "Rpm", "Q", "HP", "BP", "synthetic")
    params = json.loads((tmp_path / "synthetic-I-flow_params.json").read_text())

    assert params["Imax"]["value"] == pytest.approx(IMAX, rel=0.02)
    assert params["Vp0"]["value"] == pytest.approx(VP0, rel=0.02)
    assert params["Vpmax"]["value"] == pytest.approx(VPMAX, rel=0.02)
