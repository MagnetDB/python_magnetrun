"""Tests that ``summary.json`` written by the analysis CLI includes Stats files."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd

from python_magnetrun.analysis.cli import _load_records
from python_magnetrun.analysis.loaders import FileSet


def _fake_record() -> MagicMock:
    """Build a record whose sources include one overview and two Stats files."""
    record = MagicMock()
    record.filename = "M5_Overview_260101-1000"
    record.housing = "M5"
    record.duration = 60.0
    record.has_data.return_value = False
    record.get_overview.return_value = pd.DataFrame()
    record.get_archive.return_value = pd.DataFrame()
    record.get_pupitre.return_value = pd.DataFrame()
    record.get_incidents.return_value = {}
    record.sources = FileSet(
        overview=["/data/M5/Overview/M5_Overview_260101-1000.tdms"],
        stats=[
            "/data/M5/Fichiers_stats/M5_Stats_260101-0900.tdms",
            "/data/M5/Fichiers_stats/M5_Stats_260101-1000.tdms",
        ],
    )
    return record


def test_summary_json_includes_stats(tmp_path):
    """Discovered Stats files are written to the ``stats`` key of summary.json."""
    record = _fake_record()
    parsed_args = SimpleNamespace(dry_run=False, debug=False, output_dir=str(tmp_path))

    with (
        patch("python_magnetrun.analysis.cli.process_overview_file", return_value=record),
        patch("python_magnetrun.analysis.cli.print_record_summary"),
    ):
        _load_records(
            [record.sources.overview[0]], MagicMock(), parsed_args, "M5", MagicMock()
        )

    data = json.loads((tmp_path / "summary.json").read_text())
    assert len(data) == 1
    assert data[0]["stats"] == ", ".join(record.sources.stats)
