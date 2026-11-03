"""Tests for the txt2csv entry point: logging setup and opt-in ETL spec options."""

import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from python_magnetrun.housing_config import get_housing_config
from python_magnetrun.runetl import PupitreEtlSpec, save_pupitre_etl_spec
from python_magnetrun.utils import txt2csv

_COLS = (
    ["Date", "Time", "Field", "Flow1", "Flow2", "Tin1", "Tin2", "Rpm1", "Rpm2", "HP1", "HP2"]
    + [f"Idcct{i}" for i in range(1, 5)]
    + [f"{f}{n}" for n in range(1, 17) for f in ("Icoil", "Ucoil", "DRcoil", "Tcal")]
)


@pytest.fixture()
def m9_file(tmp_path):
    """Synthetic M9 pupitre file starting 2026-05-15 10:00:00 (Paris time)."""
    values = {"Field": "1.0", "Flow1": "50", "Flow2": "40", "Idcct1": "1000", "Idcct2": "1000"}
    rows = [
        "\t".join({"Date": "2026.05.15", "Time": t, **values}.get(c, "0") for c in _COLS)
        for t in ("10:00:00", "10:00:01")
    ]
    path = tmp_path / "2026.05.15 - 10:00:00.txt"
    path.write_text("16\tM25032101\t14800\t11899\n" + "\t".join(_COLS) + "\n" + "\n".join(rows) + "\n")
    return path


def _run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["txt2csv", *argv])
    with pytest.raises(SystemExit):  # --list exits
        txt2csv.main()


def test_logging_when_run_as_module(m9_file):
    """Under ``python -m`` the module is ``__main__``: its logger must still be configured."""
    out = subprocess.run(
        [sys.executable, "-m", "python_magnetrun.utils.txt2csv", str(m9_file), "--housing", "M9", "--list"],
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout
    assert "keys=[" in out


def test_list_prints_keys_without_debug(m9_file, monkeypatch, capsys):
    _run(monkeypatch, str(m9_file), "--housing", "M9", "--list")
    assert "keys=[" in capsys.readouterr().out


def test_etl_dir_reports_spec_file(m9_file, tmp_path, monkeypatch, capsys):
    cfg = get_housing_config("M9", at=datetime(2026, 5, 15, 10, tzinfo=ZoneInfo("Europe/Paris")))
    spec = PupitreEtlSpec.from_housing(cfg, assembly="M9_test")
    save_pupitre_etl_spec(tmp_path / "M9_test.M9-181109.pupitre-etl.json", spec)
    _run(
        monkeypatch, str(m9_file), "--housing", "M9", "--list",
        "--etl-dir", str(tmp_path), "--assembly", "M9_test",
    )
    assert "M9_test.M9-181109.pupitre-etl.json" in capsys.readouterr().out
