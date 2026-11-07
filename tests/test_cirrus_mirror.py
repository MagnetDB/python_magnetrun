"""Offline tests for python_magnetrun.requests.cirrus_mirror."""

import os
from datetime import date

import pytest

from python_magnetrun.requests.cirrus_mirror import (
    ListedFile,
    log_day,
    mirror_feed,
    parse_listing,
    parse_size,
    run,
)

SERVER = "https://srv-data-install.lncmi.cnrs.fr/"
TODAY = date(2026, 11, 5)
LAST_MODIFIED = "Sat, 19 Sep 2026 20:16:27 GMT"


def _row(name, modified, size, icon="unknown.gif"):
    return (
        f'<tr><td valign="top"><img src="/icons/{icon}" alt="[   ]"></td>'
        f'<td><a href="{name}">{name}</a></td><td align="right">{modified}  </td>'
        f'<td align="right">{size}</td><td>&nbsp;</td></tr>\n'
    )


def _listing(*rows):
    return (
        "<html><body><table>\n"
        '<tr><th valign="top"><img src="/icons/blank.gif" alt="[ICO]"></th>'
        '<th><a href="?C=N;O=D">Name</a></th><th><a href="?C=M;O=A">Last modified</a></th>'
        '<th><a href="?C=S;O=A">Size</a></th><th><a href="?C=D;O=A">Description</a></th></tr>\n'
        '<tr><td valign="top"><img src="/icons/back.gif" alt="[PARENTDIR]"></td>'
        '<td><a href="/cirrus/">Parent Directory</a></td><td>&nbsp;</td>'
        '<td align="right">  - </td><td>&nbsp;</td></tr>\n'
        + "".join(rows)
        + "</table></body></html>"
    )


class FakeResponse:
    def __init__(self, status_code=200, text="", content=b"", headers=None):
        self.status_code = status_code
        self.text = text
        self._content = content
        self.headers = headers or {}
        self.closed = False

    def iter_content(self, chunk_size=1):
        for i in range(0, len(self._content), chunk_size):
            yield self._content[i : i + chunk_size]

    def close(self):
        self.closed = True


class FakeSession:
    """Serve fixed responses by URL; unknown URLs answer 404."""

    def __init__(self, responses):
        self.responses = responses
        self.requested = []

    def get(self, url, timeout=None, stream=False):
        self.requested.append(url)
        return self.responses.get(url, FakeResponse(status_code=404))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _feed_session(log_rows, xml_rows, files):
    responses = {
        SERVER + "cirrus/A1/": FakeResponse(text=_listing(*log_rows)),
        SERVER + "cirrus/A1/xml/": FakeResponse(text=_listing(*xml_rows)),
    }
    for path, content in files.items():
        responses[SERVER + "cirrus/A1/" + path] = FakeResponse(
            content=content, headers={"Last-Modified": LAST_MODIFIED}
        )
    return FakeSession(responses)


def _mirror(session, root, **kwargs):
    kwargs.setdefault("today", TODAY)
    kwargs.setdefault("delay", 0)
    return mirror_feed(session, SERVER, "A1", root, **kwargs)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("271K", 271 * 1024),
        ("9.5K", int(9.5 * 1024)),
        ("1.2M", int(1.2 * 1024**2)),
        ("0", 0),
        ("812", 812),
        ("  - ", None),
        ("", None),
    ],
)
def test_parse_size(text, expected):
    assert parse_size(text) == expected


def test_parse_listing():
    html = _listing(
        _row("2026-09-19_cirrus_out.log", "2026-09-19 22:16", "271K"),
        _row("cirrus_out.log", "2026-11-05 12:14", "767K"),
        _row("xml/", "2026-09-01 09:02", "  - ", icon="folder.gif"),
    )
    assert parse_listing(html) == [
        ListedFile("2026-09-19_cirrus_out.log", "2026-09-19 22:16", 271 * 1024),
        ListedFile("cirrus_out.log", "2026-11-05 12:14", 767 * 1024),
        ListedFile("xml/", "2026-09-01 09:02", None),
    ]


def test_log_day():
    assert log_day("2026-09-19_cirrus_out.log") == date(2026, 9, 19)
    assert log_day("cirrus_out.log") is None
    assert log_day("A1_version_config_25.xml") is None


def test_mirror_downloads_missing_files(tmp_path):
    session = _feed_session(
        [
            _row("2026-09-19_cirrus_out.log", "2026-09-19 22:16", "1K"),
            _row("cirrus_out.log", "2026-11-05 12:14", "1K"),
            _row("xml/", "2026-09-01 09:02", "  - ", icon="folder.gif"),
        ],
        [_row("A1_version_config_25.xml", "2025-09-18 13:50", "1K")],
        {"2026-09-19_cirrus_out.log": b"7 9.38 Debut\n", "xml/A1_version_config_25.xml": b"<x/>"},
    )
    summary = _mirror(session, tmp_path)

    log = tmp_path / "cirrus" / "A1" / "2026-09-19_cirrus_out.log"
    xml = tmp_path / "cirrus" / "A1" / "xml" / "A1_version_config_25.xml"
    assert log.read_bytes() == b"7 9.38 Debut\n"
    assert xml.read_bytes() == b"<x/>"
    # mtime from Last-Modified
    assert int(os.stat(log).st_mtime) == 1789848987
    # the undated live file and the xml/ folder are not log files
    assert not (tmp_path / "cirrus" / "A1" / "cirrus_out.log").exists()
    assert (summary.logs_downloaded, summary.xml_downloaded, summary.failed) == (1, 1, 0)
    assert summary.bytes == len(b"7 9.38 Debut\n") + len(b"<x/>")
    assert not list(tmp_path.rglob("*.part"))


def test_mirror_skips_old_days_and_refetches_recent(tmp_path):
    feed = tmp_path / "cirrus" / "A1"
    feed.mkdir(parents=True)
    for name in (
        "2026-11-03_cirrus_out.log",
        "2026-11-04_cirrus_out.log",
        "2026-11-05_cirrus_out.log",
    ):
        (feed / name).write_bytes(b"old")
    session = _feed_session(
        [
            _row("2026-11-03_cirrus_out.log", "2026-11-03 18:17", "1K"),
            _row("2026-11-04_cirrus_out.log", "2026-11-04 22:30", "1K"),
            _row("2026-11-05_cirrus_out.log", "2026-11-05 12:14", "1K"),
        ],
        [],
        {
            "2026-11-03_cirrus_out.log": b"new",
            "2026-11-04_cirrus_out.log": b"new",
            "2026-11-05_cirrus_out.log": b"new",
        },
    )
    summary = _mirror(session, tmp_path)

    assert (feed / "2026-11-03_cirrus_out.log").read_bytes() == b"old"  # before yesterday
    assert (feed / "2026-11-04_cirrus_out.log").read_bytes() == b"new"  # yesterday
    assert (feed / "2026-11-05_cirrus_out.log").read_bytes() == b"new"  # today
    assert SERVER + "cirrus/A1/2026-11-03_cirrus_out.log" not in session.requested
    assert (summary.logs_downloaded, summary.logs_skipped) == (2, 1)


def test_mirror_date_range(tmp_path):
    session = _feed_session(
        [_row(f"2026-09-{d}_cirrus_out.log", f"2026-09-{d} 22:00", "1K") for d in (16, 17, 18)],
        [],
        {f"2026-09-{d}_cirrus_out.log": b"x" for d in (16, 17, 18)},
    )
    summary = _mirror(session, tmp_path, start=date(2026, 9, 17), end=date(2026, 9, 17), xml=False)

    assert [p.name for p in (tmp_path / "cirrus" / "A1").iterdir()] == ["2026-09-17_cirrus_out.log"]
    assert SERVER + "cirrus/A1/xml/" not in session.requested
    assert summary.logs_downloaded == 1


def test_mirror_never_overwrites_xml(tmp_path, caplog):
    xml_dir = tmp_path / "cirrus" / "A1" / "xml"
    xml_dir.mkdir(parents=True)
    (xml_dir / "A1_version_config_25.xml").write_bytes(b"a" * 1024)
    (xml_dir / "A1_version_config_26.xml").write_bytes(b"b" * 1024)
    session = _feed_session(
        [],
        [
            _row("A1_version_config_25.xml", "2025-09-18 13:50", "1.0K"),
            _row("A1_version_config_26.xml", "2025-09-18 13:50", "200K"),
        ],
        {"xml/A1_version_config_25.xml": b"new", "xml/A1_version_config_26.xml": b"new"},
    )
    summary = _mirror(session, tmp_path)

    assert (xml_dir / "A1_version_config_25.xml").read_bytes() == b"a" * 1024
    assert (xml_dir / "A1_version_config_26.xml").read_bytes() == b"b" * 1024
    assert not any(u.endswith(".xml") for u in session.requested)
    assert (summary.xml_downloaded, summary.xml_skipped) == (0, 2)
    # size mismatch reported for 26 only
    assert "A1_version_config_26.xml" in caplog.text
    assert "A1_version_config_25.xml" not in caplog.text


def test_mirror_failure_keeps_going(tmp_path):
    session = _feed_session(
        [
            _row("2026-09-18_cirrus_out.log", "2026-09-18 21:56", "1K"),
            _row("2026-09-19_cirrus_out.log", "2026-09-19 22:16", "1K"),
        ],
        [],
        {"2026-09-19_cirrus_out.log": b"ok"},  # 2026-09-18 answers 404
    )
    summary = _mirror(session, tmp_path)

    feed = tmp_path / "cirrus" / "A1"
    assert not (feed / "2026-09-18_cirrus_out.log").exists()
    assert (feed / "2026-09-19_cirrus_out.log").read_bytes() == b"ok"
    assert (summary.logs_downloaded, summary.failed) == (1, 1)
    assert not list(tmp_path.rglob("*.part"))


def test_mirror_listing_failure(tmp_path):
    summary = _mirror(FakeSession({}), tmp_path)
    assert summary.failed == 2  # log listing and xml listing


def test_mirror_dry_run(tmp_path):
    session = _feed_session(
        [_row("2026-09-19_cirrus_out.log", "2026-09-19 22:16", "1K")],
        [_row("A1_version_config_25.xml", "2025-09-18 13:50", "1K")],
        {"2026-09-19_cirrus_out.log": b"x", "xml/A1_version_config_25.xml": b"x"},
    )
    summary = _mirror(session, tmp_path, dry_run=True)

    assert not (tmp_path / "cirrus").exists()
    assert session.requested == [SERVER + "cirrus/A1/", SERVER + "cirrus/A1/xml/"]
    assert (summary.logs_downloaded, summary.xml_downloaded) == (1, 1)


class _Args:
    root = None
    feeds = ["A1"]
    start = None
    end = None
    no_xml = False
    server = SERVER
    delay = 0.0
    dry_run = True
    log_level = "INFO"


def test_run_requires_root(monkeypatch):
    monkeypatch.setattr("python_magnetrun.requests.cirrus_mirror.PUPITRE_RUNLOG_DIR", "")
    with pytest.raises(SystemExit, match="--root"):
        run(_Args())


def test_run_exit_code(monkeypatch, tmp_path):
    args = _Args()
    args.root = str(tmp_path)
    monkeypatch.setattr(
        "python_magnetrun.requests.cirrus_mirror.requests.Session", lambda: FakeSession({})
    )
    assert run(args) == 1  # both listings fail
