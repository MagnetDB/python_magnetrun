"""Tests for the EXPERIENCES_LOG session mask of HybridRun.getData.

The Hybrid system records the supply currents whatever housing they feed;
with ``LoadOptions.sessions_log`` a current is kept only inside a session of
the run's housing on its supply group, and Otsu on the companion voltage is
the fallback where the log gives no answer.
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from python_magnetrun.hybrid.hybrid_run import HybridRun, LoadOptions

pytest.importorskip("natsort")

_HEADER = (
    "timestamp,UserCode,UserName,OperatorName,Magnet,ElecGroup,HStart,HStop,"
    "MinuteCount,Paimant,Pmax,Phydraulique,Ptotale,dDepassement,nDepassement,"
    "Cout,Qtotale,QCout\n"
)

CURRENT = "ALIM1_J1"
ORIGIN = datetime(2026, 10, 14, 6, 0)  # naive UTC origin of the kHz time array
TIME = np.arange(0.0, 4 * 3600.0, 60.0)  # 06:00 -> 09:59 UTC, one sample per minute


def _row(magnet: str, group: int, hstart: datetime, hstop: datetime) -> str:
    """One EXPERIENCES_LOG row (Paris wall-clock time, spurious AM/PM marker)."""

    def fmt(dt: datetime) -> str:
        return f"{dt.month}/{dt.day}/{dt:%y}  {dt:%H:%M:%S}.000 {'PM' if dt.hour >= 12 else 'AM'}"

    return f"{hstart:%Y-%m-%d %H:%M:%S},U,,,{magnet},{group},{fmt(hstart)},{fmt(hstop)},60,1.0,0.0,,1.0,,,,,\n"


def _write_log(tmp_path: Path, rows: list[str]) -> Path:
    path = tmp_path / "EXPERIENCES_LOG.csv"
    path.write_text(_HEADER + "".join(rows))
    return path


def _make_hrun() -> HybridRun:
    """HybridRun on M8 with a mock HybridData: constant 447 A current, 5 V voltage."""
    hrun = object.__new__(HybridRun)
    hrun.Housing = "M8"
    hrun._cache = {}
    hrun._cache_size_bytes = 0
    hrun._cache_max_size_bytes = 256 * 1024 * 1024
    hrun.default_options = LoadOptions(lazy=False, cache=False)
    hrun.HybridData = MagicMock()
    hrun.HybridData.khz_time_origin.return_value = ORIGIN

    def read_khz(system, variable, hours=None, apply_calib=True, cnv_dir=None):
        value = 447.0 if variable == CURRENT else 5.0
        return np.full(TIME.size, value), TIME.copy()

    hrun.HybridData.read_khz_variable.side_effect = read_khz
    return hrun


def _config(session_group: bool = True) -> MagicMock:
    cfg = MagicMock()
    cfg.hybrid_formula_map = {}
    cfg.hybrid_voltage_mask_map = {f"FEPC-AUX-LNCMI/{CURRENT}": "FEPC-AUX-LNCMI/BITTER_V2"}
    cfg.hybrid_session_group_map = {f"FEPC-AUX-LNCMI/{CURRENT}": 1} if session_group else {}
    return cfg


def _get(hrun: HybridRun, cfg: MagicMock, sessions_log: Path | None, otsu_mask=None):
    """Call getData on the current channel; return (data, binarize mock)."""
    binarize = MagicMock(side_effect=lambda v, **kw: np.ones(v.size) if otsu_mask is None else otsu_mask)
    with (
        patch("python_magnetrun.housing_config.get_housing_config", return_value=cfg),
        patch("python_magnetrun.processing.signal.binarize_signal", binarize),
    ):
        data, _ = hrun.getData(
            f"kHz/FEPC-AUX-LNCMI/{CURRENT}",
            hours=[6, 7, 8, 9],
            options=LoadOptions(lazy=False, cache=False, sessions_log=sessions_log),
        )
    return data, binarize


def _utc_minutes(hhmm: str) -> int:
    """Index in TIME of a UTC ``HH:MM`` time."""
    h, m = (int(x) for x in hhmm.split(":"))
    return (h - ORIGIN.hour) * 60 + m


def test_session_mask_keeps_current_inside_session_only(tmp_path: Path) -> None:
    """M8e session 09:00-10:00 Paris (07:00-08:00 UTC), padded by 5 min."""
    log = _write_log(tmp_path, [
        _row("M8e", 1, datetime(2026, 10, 14, 9, 0), datetime(2026, 10, 14, 10, 0)),
        _row("M10e", 1, datetime(2026, 10, 14, 11, 0), datetime(2026, 10, 14, 12, 0)),
    ])
    hrun = _make_hrun()
    data, binarize = _get(hrun, _config(), log)

    assert data[_utc_minutes("06:54")] == 0.0
    assert data[_utc_minutes("06:55")] == 447.0
    assert data[_utc_minutes("08:05")] == 447.0
    assert data[_utc_minutes("08:06")] == 0.0
    assert data[_utc_minutes("09:30")] == 0.0  # M10 session on the same group
    binarize.assert_not_called()
    variables = [c.args[1] for c in hrun.HybridData.read_khz_variable.call_args_list]
    assert variables == [CURRENT]  # the voltage channel is not read


def test_otsu_fallback_after_log_end(tmp_path: Path) -> None:
    """Data after the last log entry are masked with Otsu on the voltage."""
    log = _write_log(tmp_path, [
        _row("M8e", 1, datetime(2026, 10, 13, 9, 0), datetime(2026, 10, 13, 10, 0)),
    ])
    otsu = np.zeros(TIME.size)
    otsu[:30] = 1.0
    data, binarize = _get(_make_hrun(), _config(), log, otsu_mask=otsu)

    binarize.assert_called_once()
    np.testing.assert_array_equal(data, 447.0 * otsu)


def test_otsu_only_after_log_end_when_log_partly_covers(tmp_path: Path) -> None:
    """Covered samples follow the log, later ones follow Otsu."""
    log = _write_log(tmp_path, [
        _row("M8e", 1, datetime(2026, 10, 14, 9, 0), datetime(2026, 10, 14, 10, 0)),
    ])  # log ends 08:00 UTC
    data, binarize = _get(_make_hrun(), _config(), log, otsu_mask=np.ones(TIME.size))

    binarize.assert_called_once()
    assert data[_utc_minutes("06:30")] == 0.0  # covered, no session
    assert data[_utc_minutes("07:30")] == 447.0  # covered, in session
    assert data[_utc_minutes("09:00")] == 447.0  # not covered, Otsu says on


def test_no_sessions_log_uses_otsu() -> None:
    data, binarize = _get(_make_hrun(), _config(), None)

    binarize.assert_called_once()
    np.testing.assert_array_equal(data, 447.0)


def test_channel_without_session_group_uses_otsu(tmp_path: Path) -> None:
    log = _write_log(tmp_path, [
        _row("M8e", 1, datetime(2026, 10, 14, 9, 0), datetime(2026, 10, 14, 10, 0)),
    ])
    data, binarize = _get(_make_hrun(), _config(session_group=False), log)

    binarize.assert_called_once()
    np.testing.assert_array_equal(data, 447.0)
