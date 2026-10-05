"""Offline tests for python_magnetrun.requests.cirrus_logs."""

import csv
from datetime import date

import pytest

from python_magnetrun.requests.cirrus_logs import (
    COLUMNS,
    daily_files,
    find_duplicates,
    load_cirrus_log,
    normalize_file,
    save_cirrus_log,
)

BASE_URL = "https://srv-data-install.lncmi.cnrs.fr/"
PAGES_URL = BASE_URL + "site/sba/pages/"


class FakeResponse:
    def __init__(self, url, status_code=200, payload=None, text=""):
        self.url = url
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.requested = []

    def get(self, url, verify=True):
        self.requested.append(url)
        return self.response


ROWS = [
    {"date": "2026-07-27", "time": "08:00:01", "message": "Marche, redresseur", "type": "Info"},
    {"date": "2026-07-27", "time": "08:05:12", "message": 'Defaut "T"', "type": "Erreur"},
]


@pytest.mark.parametrize(
    "value",
    [
        "A1/2026-07-27_cirrus_out.log",
        PAGES_URL + "cirrus.php?file=A1/2026-07-27_cirrus_out.log",
        PAGES_URL + "cirrus.php?filedata=A1/2026-07-27_cirrus_out.log",
    ],
)
def test_normalize_file(value):
    assert normalize_file(value) == "A1/2026-07-27_cirrus_out.log"


def test_daily_files():
    assert daily_files("A2", date(2026, 7, 30), date(2026, 8, 1)) == [
        "A2/2026-07-30_cirrus_out.log",
        "A2/2026-07-31_cirrus_out.log",
        "A2/2026-08-01_cirrus_out.log",
    ]


def test_load_cirrus_log():
    url = PAGES_URL + "cirrus.php?filedata=A1/2026-07-27_cirrus_out.log"
    session = FakeSession(FakeResponse(url, payload={"data": ROWS}))
    df = load_cirrus_log(session, BASE_URL, "A1/2026-07-27_cirrus_out.log")
    assert session.requested == [url]
    assert list(df.columns) == COLUMNS
    assert len(df) == 2
    assert df.loc[1, "Message"] == 'Defaut "T"'


def test_load_cirrus_log_empty():
    session = FakeSession(FakeResponse(PAGES_URL + "cirrus.php", payload={"data": []}))
    df = load_cirrus_log(session, BASE_URL, "A1/2026-07-27_cirrus_out.log")
    assert df.empty
    assert list(df.columns) == COLUMNS


def test_load_cirrus_log_login_redirect():
    session = FakeSession(FakeResponse(PAGES_URL + "login.php", payload={"data": []}))
    with pytest.raises(RuntimeError, match="login"):
        load_cirrus_log(session, BASE_URL, "A1/2026-07-27_cirrus_out.log")


def test_load_cirrus_log_http_error():
    session = FakeSession(FakeResponse(PAGES_URL + "cirrus.php", status_code=404))
    with pytest.raises(RuntimeError, match="HTTP 404"):
        load_cirrus_log(session, BASE_URL, "A1/2026-07-27_cirrus_out.log")


def test_load_cirrus_log_not_json():
    session = FakeSession(FakeResponse(PAGES_URL + "cirrus.php", text="<html></html>"))
    with pytest.raises(RuntimeError, match="JSON"):
        load_cirrus_log(session, BASE_URL, "A1/2026-07-27_cirrus_out.log")


def test_save_cirrus_log(tmp_path):
    session = FakeSession(FakeResponse(PAGES_URL + "cirrus.php", payload={"data": ROWS}))
    df = load_cirrus_log(session, BASE_URL, "A1/2026-07-27_cirrus_out.log")

    path = save_cirrus_log(df, "A1/2026-07-27_cirrus_out.log", tmp_path)
    assert path == tmp_path / "cirrus" / "A1" / "2026-07-27_cirrus_out.csv"

    lines = path.read_text().splitlines()
    assert lines[0] == '"Date","Time","Message","Type"'
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[1]["Message"] == 'Defaut "T"'
    assert rows[0]["Message"] == "Marche, redresseur"


def test_save_cirrus_log_no_overwrite(tmp_path):
    target = tmp_path / "cirrus" / "A1" / "2026-07-27_cirrus_out.csv"
    target.parent.mkdir(parents=True)
    target.write_text("old")

    session = FakeSession(FakeResponse(PAGES_URL + "cirrus.php", payload={"data": ROWS}))
    df = load_cirrus_log(session, BASE_URL, "A1/2026-07-27_cirrus_out.log")

    assert save_cirrus_log(df, "A1/2026-07-27_cirrus_out.log", tmp_path) is None
    assert target.read_text() == "old"

    assert save_cirrus_log(df, "A1/2026-07-27_cirrus_out.log", tmp_path, overwrite=True) == target
    assert target.read_text() != "old"


def _df(rows):
    import pandas as pd

    return pd.DataFrame(rows, columns=["date", "time", "message", "type"]).set_axis(
        COLUMNS, axis=1
    )


def test_find_duplicates_within_file():
    seen = set()
    df = _df([ROWS[0], ROWS[1], ROWS[0]])
    assert find_duplicates(df, seen).tolist() == [False, False, True]
    assert len(seen) == 2


def test_find_duplicates_across_files():
    seen = set()
    find_duplicates(_df(ROWS), seen)
    other = {"date": "2026-07-28", "time": "00:00:01", "message": "Marche", "type": "Info"}
    mask = find_duplicates(_df([ROWS[1], other]), seen)
    assert mask.tolist() == [True, False]
    assert len(seen) == 3


def test_find_duplicates_empty():
    mask = find_duplicates(_df([]), set())
    assert mask.empty
