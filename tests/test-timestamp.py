"""Tests for python_magnetrun.utils.timestamps.

Covers:
  - parse_txt_filename   — three date/time formats + housing prefix
  - parse_tdms_filename  — YYMMDD-HHMM and YYMMDD-HHMMSS variants
  - parse_filename_timestamp — extension-based dispatch
  - parse_wf_start_time  — extraction from TDMS Groups dict
  - seconds_since_midnight
  - convert_to_timestamp_aware
  - convert_to_timestamp
"""

from datetime import datetime

import numpy as np

from python_magnetrun.utils.timestamps import (
    convert_to_timestamp,
    convert_to_timestamp_aware,
    parse_filename_timestamp,
    parse_tdms_filename,
    parse_txt_filename,
    parse_wf_start_time,
    seconds_since_midnight,
)

# ---------------------------------------------------------------------------
# parse_txt_filename
# ---------------------------------------------------------------------------


class TestParseTxtFilename:
    def test_new_standard_format(self) -> None:
        dt = parse_txt_filename("M9_2019.02.14 - 23:00:38.txt")
        assert dt == datetime(2019, 2, 14, 23, 0, 38)

    def test_legacy_triple_dash_format(self) -> None:
        dt = parse_txt_filename("M9_2019.02.14---23:00:38.txt")
        assert dt == datetime(2019, 2, 14, 23, 0, 38)

    def test_no_housing_prefix(self) -> None:
        dt = parse_txt_filename("2022.03.30 - 21:55:17.txt")
        assert dt == datetime(2022, 3, 30, 21, 55, 17)

    def test_wrong_extension_returns_none(self) -> None:
        assert parse_txt_filename("M9_2019.02.14 - 23:00:38.csv") is None

    def test_unrecognised_format_returns_none(self) -> None:
        assert parse_txt_filename("garbage.txt") is None


# ---------------------------------------------------------------------------
# parse_tdms_filename
# ---------------------------------------------------------------------------


class TestParseTdmsFilename:
    def test_four_digit_time(self) -> None:
        dt = parse_tdms_filename("M9_Overview_230718-1506.tdms")
        assert dt == datetime(2023, 7, 18, 15, 6)

    def test_six_digit_time(self) -> None:
        dt = parse_tdms_filename("M8_Default_251105-095300.tdms")
        assert dt == datetime(2025, 11, 5, 9, 53, 0)

    def test_with_dmode_suffix(self) -> None:
        dt = parse_tdms_filename("M8_Default_251105-095300_raw.tdms")
        assert dt == datetime(2025, 11, 5, 9, 53, 0)

    def test_wrong_extension_returns_none(self) -> None:
        assert parse_tdms_filename("M9_Overview_230718-1506.txt") is None

    def test_too_few_parts_returns_none(self) -> None:
        assert parse_tdms_filename("bad.tdms") is None

    def test_missing_hyphen_in_timestamp_returns_none(self) -> None:
        # No '-' between date and time → cannot split
        assert parse_tdms_filename("M9_Overview_231506.tdms") is None


# ---------------------------------------------------------------------------
# parse_filename_timestamp  (dispatch)
# ---------------------------------------------------------------------------


class TestParseFilenameTimestamp:
    def test_txt_dispatches_correctly(self) -> None:
        dt = parse_filename_timestamp("M9_2019.02.14 - 23:00:38.txt")
        assert dt == datetime(2019, 2, 14, 23, 0, 38)

    def test_tdms_dispatches_correctly(self) -> None:
        dt = parse_filename_timestamp("M9_Overview_230718-1506.tdms")
        assert dt == datetime(2023, 7, 18, 15, 6)

    def test_unsupported_extension_returns_none(self) -> None:
        assert parse_filename_timestamp("data.xlsx") is None


# ---------------------------------------------------------------------------
# parse_wf_start_time
# ---------------------------------------------------------------------------


class TestParseWfStartTime:
    def test_extracts_first_matching_channel(self) -> None:
        wf_start = np.datetime64("2024-01-01T10:00:00")
        groups = {"GrpX": {"ChA": {"wf_start_time": wf_start, "wf_increment": 1.0}}}
        dt = parse_wf_start_time(groups)
        assert dt == datetime(2024, 1, 1, 10, 0, 0)

    def test_skips_infos_group(self) -> None:
        groups = {
            "Infos": {"meta": {"wf_start_time": np.datetime64("2024-06-01T08:00:00")}},
            "GrpX": {"ChA": {"wf_start_time": np.datetime64("2024-01-01T10:00:00")}},
        }
        dt = parse_wf_start_time(groups)
        assert dt == datetime(2024, 1, 1, 10, 0, 0)

    def test_returns_none_when_no_wf_start_time(self) -> None:
        groups = {"GrpX": {"ChA": {"wf_increment": 1.0}}}
        assert parse_wf_start_time(groups) is None

    def test_returns_none_for_empty_groups(self) -> None:
        assert parse_wf_start_time({}) is None


# ---------------------------------------------------------------------------
# seconds_since_midnight
# ---------------------------------------------------------------------------


class TestSecondsSinceMidnight:
    def test_midnight(self) -> None:
        assert seconds_since_midnight(datetime(2024, 1, 1, 0, 0, 0)) == 0.0

    def test_noon(self) -> None:
        assert seconds_since_midnight(datetime(2024, 1, 1, 12, 0, 0)) == 43200.0

    def test_arbitrary_time(self) -> None:
        # 1h 2m 3s = 3723 s
        assert seconds_since_midnight(datetime(2024, 1, 1, 1, 2, 3)) == 3723.0


# ---------------------------------------------------------------------------
# convert_to_timestamp_aware  (UTC-aware)
# ---------------------------------------------------------------------------


class TestConvertToTimestampAware:
    def test_utc_string(self) -> None:
        # Paris CEST in July = UTC+2; 15:06 local → 13:06 UTC
        _, s = convert_to_timestamp_aware("230718", "1506")
        assert s == "2023-07-18T13:06:00"

    def test_returns_two_tuple(self) -> None:
        result = convert_to_timestamp_aware("230718", "1506")
        assert isinstance(result, tuple) and len(result) == 2

    def test_timestamp_is_float(self) -> None:
        ts, _ = convert_to_timestamp_aware("230718", "1506")
        assert isinstance(ts, float)


# ---------------------------------------------------------------------------
# convert_to_timestamp  (naive local)
# ---------------------------------------------------------------------------


class TestConvertToTimestamp:
    def test_formatted_string(self) -> None:
        _, s = convert_to_timestamp("230718", "1506")
        assert s == "2023-07-18 15:06:00"

    def test_returns_two_tuple(self) -> None:
        result = convert_to_timestamp("230718", "1506")
        assert isinstance(result, tuple) and len(result) == 2
