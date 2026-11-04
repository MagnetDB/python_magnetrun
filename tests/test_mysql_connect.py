"""Tests for the ``--sort-time`` option of ``examples/mysql_connect.py``.

The MySQL server is replaced by an in-memory DuckDB database attached as
``mysqldb``, so only the SQL/ordering logic is exercised.
"""

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "mysql_connect.py"
_spec = importlib.util.spec_from_file_location("mysql_connect", _SCRIPT)
assert _spec is not None and _spec.loader is not None
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)

# Insertion order is deliberately shuffled; Icoil = minute offset + 0.25.
_SHUFFLED = [3, 7, 0, 9, 5, 1, 8, 2, 6, 4]


def _fake_attach(con, dsn, verbose=False):
    """Attach an in-memory DuckDB as ``mysqldb`` and populate test tables."""
    con.execute("ATTACH ':memory:' AS mysqldb")
    con.execute("CREATE TABLE mysqldb.meas (t TIMESTAMP, Icoil DOUBLE)")
    for i in _SHUFFLED:
        con.execute(
            "INSERT INTO mysqldb.meas VALUES "
            "(TIMESTAMP '2026-01-01 00:00:00' + to_minutes(?), ?)",
            [i, i + 0.25],
        )
    con.execute("CREATE TABLE mysqldb.notime (name VARCHAR, val DOUBLE)")
    con.execute("INSERT INTO mysqldb.notime VALUES ('b', 2.0), ('a', 1.0)")
    # Europe/Paris local times: summer (UTC+2) and winter (UTC+1).
    con.execute(
        'CREATE TABLE mysqldb.stamps (t TIMESTAMP, "Timestamp Field" TIMESTAMP, '
        "d DATE, label VARCHAR)"
    )
    con.execute(
        "INSERT INTO mysqldb.stamps VALUES "
        "('2026-09-17 08:42:37', '2026-09-17 08:42:37', '2026-09-17', '09/17/26  08:42:37'), "
        "('2026-12-17 08:42:37', '2026-12-17 08:42:37', '2026-12-17', '12/17/26  08:42:37')"
    )


@pytest.fixture(autouse=True)
def fake_mysql(monkeypatch):
    """Replace the MySQL extension/attach steps with an in-memory DuckDB."""
    monkeypatch.setattr(mc, "_load_mysql_extension", lambda con, verbose=False: None)
    monkeypatch.setattr(mc, "_attach_mysql", _fake_attach)


def _conn_args(**kw):
    base = {
        "host": "h", "port": 3306, "user": "u", "password": "p",
        "database": "main", "verbose": False, "sort_time": None,
    }
    base.update(kw)
    return argparse.Namespace(**base)


def _export_args(tmp_path, **kw):
    base = {
        "fmt": "csv", "output": str(tmp_path / "out.duckdb"),
        "output_dir": str(tmp_path), "tables": ["meas"], "table": None,
        "export_fields": None, "time_field": None, "start": None, "end": None,
        "utc": False,
    }
    base.update(kw)
    return _conn_args(**base)


def _view_args(**kw):
    base = {
        "table": "meas", "query": None, "fields": None, "where": None,
        "limit": None, "x_field": None,
    }
    base.update(kw)
    return _conn_args(**base)


def _read_icoil(path):
    with open(path, newline="") as fh:
        return [float(r["Icoil"]) for r in csv.DictReader(fh)]


# ---------------------------------------------------------------------------
# _time_sort_column
# ---------------------------------------------------------------------------


@pytest.fixture
def con():
    import duckdb

    c = duckdb.connect()
    _fake_attach(c, "")
    yield c
    c.close()


def test_time_sort_column_autodetect(con):
    assert mc._time_sort_column(con, "meas", None) == "t"


def test_time_sort_column_override(con):
    assert mc._time_sort_column(con, "meas", "Icoil") == "Icoil"


def test_time_sort_column_override_missing(con):
    assert mc._time_sort_column(con, "meas", "nope") is None


def test_time_sort_column_none(con):
    assert mc._time_sort_column(con, "notime", None) is None


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def test_export_sort_asc(tmp_path):
    mc.mode_export(_export_args(tmp_path, sort_time="asc"))
    assert _read_icoil(tmp_path / "meas.csv") == [i + 0.25 for i in range(10)]


def test_export_sort_desc(tmp_path):
    mc.mode_export(_export_args(tmp_path, sort_time="desc"))
    assert _read_icoil(tmp_path / "meas.csv") == [i + 0.25 for i in reversed(range(10))]


def test_export_sort_desc_filtered(tmp_path):
    mc.mode_export(
        _export_args(
            tmp_path, sort_time="desc",
            start="2026-01-01 00:02:00", end="2026-01-01 00:05:00",
        )
    )
    assert _read_icoil(tmp_path / "meas.csv") == [5.25, 4.25, 3.25, 2.25]


def test_export_no_sort_has_no_order_by(tmp_path, capsys):
    mc.mode_export(_export_args(tmp_path, verbose=True))
    assert "ORDER BY" not in capsys.readouterr().out


def test_export_sort_without_time_column_warns(tmp_path, capsys):
    mc.mode_export(_export_args(tmp_path, tables=["notime"], sort_time="asc"))
    assert "no timestamp column" in capsys.readouterr().err
    assert (tmp_path / "notime.csv").exists()


# ---------------------------------------------------------------------------
# export --utc
# ---------------------------------------------------------------------------


def _read_rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def test_export_utc_converts_timestamp_columns(tmp_path):
    mc.mode_export(_export_args(tmp_path, tables=["stamps"], utc=True))
    rows = _read_rows(tmp_path / "stamps.csv")
    assert [r["t"] for r in rows] == ["2026-09-17 06:42:37", "2026-12-17 07:42:37"]
    assert [r["Timestamp Field"] for r in rows] == [r["t"] for r in rows]
    # DATE and VARCHAR columns are left untouched.
    assert [r["d"] for r in rows] == ["2026-09-17", "2026-12-17"]
    assert [r["label"] for r in rows] == ["09/17/26  08:42:37", "12/17/26  08:42:37"]


def test_export_no_utc_keeps_local_time(tmp_path):
    mc.mode_export(_export_args(tmp_path, tables=["stamps"]))
    rows = _read_rows(tmp_path / "stamps.csv")
    assert [r["t"] for r in rows] == ["2026-09-17 08:42:37", "2026-12-17 08:42:37"]


def test_export_utc_start_end_are_utc_and_sorted(tmp_path):
    # meas holds 2026-01-01 00:00..00:09 local (CET), i.e. 2025-12-31 23:00..23:09 UTC.
    mc.mode_export(
        _export_args(
            tmp_path, utc=True, sort_time="desc",
            start="2025-12-31 23:02:00", end="2025-12-31 23:05:00",
        )
    )
    rows = _read_rows(tmp_path / "meas.csv")
    assert [float(r["Icoil"]) for r in rows] == [5.25, 4.25, 3.25, 2.25]
    assert rows[0]["t"] == "2025-12-31 23:05:00"


def test_export_utc_duckdb_format(tmp_path):
    import duckdb

    out = tmp_path / "out.duckdb"
    mc.mode_export(_export_args(tmp_path, fmt="duckdb", output=str(out), tables=["stamps"], utc=True))
    with duckdb.connect(str(out)) as c:
        assert c.execute("SELECT min(t)::VARCHAR FROM stamps").fetchone()[0] == "2026-09-17 06:42:37"


# ---------------------------------------------------------------------------
# view
# ---------------------------------------------------------------------------


def test_view_latest_rows_desc(capsys):
    pytest.importorskip("rich")
    mc.mode_view(_view_args(limit=3, sort_time="desc"))
    out = capsys.readouterr().out
    positions = [out.find(v) for v in ("9.25", "8.25", "7.25")]
    assert all(p >= 0 for p in positions)
    assert positions == sorted(positions)
    assert "6.25" not in out


def test_view_query_warns_sort_ignored(capsys):
    pytest.importorskip("rich")
    mc.mode_view(_view_args(table=None, query="SELECT * FROM mysqldb.meas", sort_time="asc"))
    assert "--sort-time is ignored" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_CLI_BASE = ["mysql_connect.py", "--user", "u", "--password", "p", "--database", "d"]


def test_parse_args_sort_time(monkeypatch):
    monkeypatch.setattr(sys, "argv", _CLI_BASE + ["--sort-time", "desc"])
    assert mc.parse_args().sort_time == "desc"


def test_parse_args_sort_time_default(monkeypatch):
    monkeypatch.setattr(sys, "argv", _CLI_BASE)
    assert mc.parse_args().sort_time is None


def test_parse_args_utc_default(monkeypatch):
    monkeypatch.setattr(sys, "argv", _CLI_BASE)
    assert mc.parse_args().utc is True


def test_parse_args_no_utc(monkeypatch):
    monkeypatch.setattr(sys, "argv", _CLI_BASE + ["--no-utc"])
    assert mc.parse_args().utc is False


def test_parse_args_sort_time_invalid(monkeypatch):
    monkeypatch.setattr(sys, "argv", _CLI_BASE + ["--sort-time", "foo"])
    with pytest.raises(SystemExit):
        mc.parse_args()
