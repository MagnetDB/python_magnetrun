"""Tests for the sub-commands of ``examples/get-record.py`` on pupitre records.

``MagnetRun.fromtxt`` yields a Polars-backed ``MagnetData`` whose
``getData()`` returns a narwhals frame; these tests check that each
sub-command still runs end-to-end on such a record.
"""

import importlib.util
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — must be set before any plt import

import pytest  # noqa: E402

_SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "get-record.py"
_spec = importlib.util.spec_from_file_location("get_record", _SCRIPT)
assert _spec is not None and _spec.loader is not None
gr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gr)

SAMPLE_TXT = Path(__file__).parent / "data" / "sample_pupitre.txt"
_START = datetime(2022, 3, 30, 21, 55, 17)
_N_ROWS = 1200  # > 1000 s: the min duration used by ``stats``


def _write_pupitre(path: Path, n_rows: int = _N_ROWS) -> Path:
    """Write a 1 Hz pupitre file built on the header/first row of ``sample_pupitre.txt``."""
    first, header, row = SAMPLE_TXT.read_text().splitlines()[:3]
    keys = header.split("\t")
    values = row.split("\t")
    lines = [first, header]
    for i in range(n_rows):
        field = 0.5 + 0.4 * math.sin(i / 20)
        values[keys.index("Field")] = f"{field:.6f}"
        values[keys.index("teb")] = f"{10.0 + 2.0 * field:.6f}"
        stamp = _START + timedelta(seconds=i)
        values[0] = stamp.strftime("%Y.%m.%d")
        values[1] = stamp.strftime("%H:%M:%S")
        lines.append("\t".join(values))
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Synthetic M9 record in *tmp_path*, which is also the working directory."""
    monkeypatch.chdir(tmp_path)
    return _write_pupitre(tmp_path / f"{_START:%Y.%m.%d - %H:%M:%S}.txt")


def _run(monkeypatch: pytest.MonkeyPatch, record: Path, *command: str) -> int:
    """Run ``get-record.py --housing M9 <record> <command...>``."""
    monkeypatch.setattr(
        sys, "argv", ["get-record.py", "--housing", "M9", str(record), *command]
    )
    return gr.main()


def test_aggregate(monkeypatch: pytest.MonkeyPatch, record: Path) -> None:
    assert _run(monkeypatch, record, "aggregate", "--fields", "teb") == 0
    assert (record.parent / "aggregate-teb.csv").exists()


def test_aggregate_multiple_records(
    monkeypatch: pytest.MonkeyPatch, record: Path
) -> None:
    # Second record truncated mid-line before ``teb`` (as in real pupitre
    # files) -> a NaN teb, which seaborn drops before realigning on the index.
    second = _START + timedelta(hours=1)
    other = _write_pupitre(record.parent / f"{second:%Y.%m.%d - %H:%M:%S}.txt")
    lines = other.read_text().splitlines()
    teb_col = lines[1].split("\t").index("teb")
    lines[-1] = "\t".join(lines[-1].split("\t")[:teb_col])
    other.write_text("\n".join(lines))
    monkeypatch.setattr(
        sys,
        "argv",
        ["get-record.py", "--housing", "M9", str(record), str(other), "aggregate", "--fields", "teb"],
    )
    assert gr.main() == 0


def test_aggregate_all_housings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    datadir = tmp_path / "pupitre"
    for housing in ("M8", "M9"):
        (datadir / housing).mkdir(parents=True)
        _write_pupitre(datadir / housing / f"{_START:%Y.%m.%d - %H:%M:%S}.txt")

    loaded: list[str] = []
    fromtxt = gr.MagnetRun.fromtxt

    def _spy(housing, assembly, file, *args, **kwargs):
        loaded.append(housing)
        return fromtxt(housing, assembly, file, *args, **kwargs)

    monkeypatch.setattr(gr.MagnetRun, "fromtxt", _spy)
    monkeypatch.setattr(
        sys,
        "argv",
        ["get-record.py", "--pupitre_datadir", str(datadir), "--housing", "ALL",
         "2022*.txt", "aggregate", "--fields", "teb"],
    )
    assert gr.main() == 0
    assert sorted(loaded) == ["M8", "M9"]
    rows = (tmp_path / "aggregate-teb.csv").read_text().splitlines()
    assert len(rows) == 1 + 2 * _N_ROWS


def test_select(
    monkeypatch: pytest.MonkeyPatch, record: Path, capsys: pytest.CaptureFixture
) -> None:
    assert _run(monkeypatch, record, "select", "--field", "0") == 0
    assert record.name in capsys.readouterr().out


def test_stats_pairplot(monkeypatch: pytest.MonkeyPatch, record: Path) -> None:
    assert (
        _run(monkeypatch, record, "stats", "--pairplot", "--fields", "Field", "teb")
        == 0
    )


def test_stats_pearson(monkeypatch: pytest.MonkeyPatch, record: Path) -> None:
    assert (
        _run(
            monkeypatch, record, "stats", "--pearson", "--fields", "Field", "teb", "--save"
        )
        == 0
    )
    assert (record.parent / "Field-teb-pearson.png").exists()
