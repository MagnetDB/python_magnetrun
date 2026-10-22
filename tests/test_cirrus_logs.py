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
    save_day,
    split_by_day,
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


def test_load_cirrus_log_as_str():
    rows = [{"date": "2026-07-27 08:00:01", "time": 9.72, "message": "Marche", "type": 7}]
    session = FakeSession(FakeResponse(PAGES_URL + "cirrus.php", payload={"data": rows}))
    df = load_cirrus_log(session, BASE_URL, "A1/2026-07-27_cirrus_out.log")
    assert df.loc[0, "Type"] == "7"
    assert df.loc[0, "Time"] == "9.72"


def test_save_day_new_file(tmp_path):
    added = save_day(_df(ROWS), "A1", date(2026, 7, 27), tmp_path)
    path = tmp_path / "cirrus" / "A1" / "2026-07-27_cirrus_out.csv"
    assert added == 2

    lines = path.read_text().splitlines()
    assert lines[0] == '"Date","Time","Message","Type"'
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["Message"] == "Marche, redresseur"
    assert rows[1]["Message"] == 'Defaut "T"'


def test_save_day_merge(tmp_path):
    first = {"date": "2026-07-27 08:00:01.000000", "time": "9.72", "message": "a", "type": "7"}
    second = {"date": "2026-07-27 08:05:12.000000", "time": "320.8", "message": "b", "type": "7"}
    save_day(_df([second]), "A1", date(2026, 7, 27), tmp_path)
    added = save_day(_df([first, second]), "A1", date(2026, 7, 27), tmp_path)
    assert added == 1

    path = tmp_path / "cirrus" / "A1" / "2026-07-27_cirrus_out.csv"
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    # existing row kept, missing row added, sorted by Date
    assert [r["Message"] for r in rows] == ["a", "b"]


def test_save_day_idempotent(tmp_path):
    save_day(_df(ROWS), "A1", date(2026, 7, 27), tmp_path)
    path = tmp_path / "cirrus" / "A1" / "2026-07-27_cirrus_out.csv"
    before = path.read_text()
    assert save_day(_df(ROWS), "A1", date(2026, 7, 27), tmp_path) == 0
    assert path.read_text() == before


def _df(rows):
    import pandas as pd

    return pd.DataFrame(rows, columns=["date", "time", "message", "type"]).set_axis(
        COLUMNS, axis=1
    )


def _row(date_, time_, message, type_="5"):
    return {"date": date_, "time": time_, "message": message, "type": type_}


# parameter dump: the KI/KP line is a continuation row (empty Time and Type)
# sharing the Date of the message it follows
DUMP = [
    _row("2026-07-22 09:19:23.893318", "573267.653168895", "les nouveaux parametres pour reg 0 sont :"),
    _row("2026-07-22 09:19:23.893318", "", " KI : 2.000000\tKP : 4.000000 \n", ""),
    _row("2026-07-22 09:19:23.893396", "573267.653246392", "frequence_calcul:300.000000"),
]


def _shift(rows, us):
    """Rows with ``date`` shifted by ``us`` microseconds, as in a re-download."""
    from datetime import datetime, timedelta

    return [
        {**r, "date": f"{datetime.fromisoformat(r['date']) + timedelta(microseconds=us):%Y-%m-%d %H:%M:%S.%f}"}
        for r in rows
    ]


def test_save_day_redownload_shifted(tmp_path):
    assert save_day(_df(DUMP), "A1", date(2026, 7, 22), tmp_path) == 3
    assert save_day(_df(_shift(DUMP, 8)), "A1", date(2026, 7, 22), tmp_path) == 0


def test_save_day_repeated_message_kept(tmp_path):
    rows = [
        _row("2026-07-22 09:13:08.999211", "572892.759061980", "sans présence de défauts"),
        _row("2026-07-22 09:13:08.999243", "572892.759093145", "sans présence de défauts"),
    ]
    assert save_day(_df(rows), "A1", date(2026, 7, 22), tmp_path) == 2
    assert save_day(_df(_shift(rows, 8)), "A1", date(2026, 7, 22), tmp_path) == 0


@pytest.mark.parametrize(
    "shift_us, tolerance_us, added",
    [(50, 1000, 0), (100_000, 1000, 1), (50, 10, 1)],
)
def test_save_day_continuation_tolerance(tmp_path, shift_us, tolerance_us, added):
    cont = [DUMP[1]]
    save_day(_df(cont), "A1", date(2026, 7, 22), tmp_path)
    new = _df(_shift(cont, shift_us))
    assert save_day(new, "A1", date(2026, 7, 22), tmp_path, tolerance_us=tolerance_us) == added


def test_find_duplicates_within_file():
    df = _df([ROWS[0], ROWS[1], ROWS[0]])
    assert find_duplicates(df).tolist() == [False, False, True]


def test_find_duplicates_across_files():
    import pandas as pd

    other = {"date": "2026-07-28", "time": "00:00:01", "message": "Marche", "type": "Info"}
    df = pd.concat([_df(ROWS), _df([ROWS[1], other])], ignore_index=True)
    assert find_duplicates(df).tolist() == [False, False, True, False]


def test_find_duplicates_shifted_copies():
    df = _df(DUMP + _shift(DUMP, 8))
    assert find_duplicates(df).tolist() == [False] * 3 + [True] * 3


def test_find_duplicates_continuation_far_apart():
    df = _df(DUMP + _shift(DUMP[1:2], 100_000))
    assert find_duplicates(df).tolist() == [False] * 4
    assert find_duplicates(df, tolerance_us=200_000).tolist() == [False] * 3 + [True]


def test_find_duplicates_empty():
    mask = find_duplicates(_df([]))
    assert mask.empty


@pytest.mark.parametrize(
    "day", ["2026-07-27 15:12:29.575500", "2026-07-27", "2026-7-27", "27/07/2026"]
)
def test_split_by_day(day):
    rows = [
        {"date": "2026-07-26 23:59:59", "time": "1", "message": "veille", "type": "Info"},
        {"date": day, "time": "2", "message": "jour-1", "type": "Info"},
        {"date": "n/a", "time": "3", "message": "illisible", "type": "Info"},
        {"date": "2026-07-28 00:00:01", "time": "4", "message": "lendemain", "type": "Info"},
        {"date": day, "time": "5", "message": "jour-2", "type": "Info"},
    ]
    days = split_by_day(_df(rows))
    assert list(days) == [date(2026, 7, 26), date(2026, 7, 27), date(2026, 7, 28)]
    assert days[date(2026, 7, 27)]["Message"].tolist() == ["jour-1", "jour-2"]
    assert days[date(2026, 7, 28)]["Message"].tolist() == ["lendemain"]


def test_split_by_day_empty():
    assert split_by_day(_df([])) == {}
