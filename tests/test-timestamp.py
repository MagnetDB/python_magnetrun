"""Tests for python_magnetrun.utils.timestamps and utils.timezone.

Covers:
  - parse_txt_filename   — three date/time formats + housing prefix
  - parse_tdms_filename  — YYMMDD-HHMM and YYMMDD-HHMMSS variants
  - parse_filename_timestamp — extension-based dispatch
  - parse_wf_start_time  — extraction from TDMS Groups dict
  - seconds_since_midnight
  - local_to_utc_naive
  - ensure_utc_naive
  - series_local_to_utc_naive
  - series_utc_to_local_naive
  - timerange_to_utc
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytz

from python_magnetrun.utils.timestamps import (
    parse_filename_timestamp,
    parse_tdms_filename,
    parse_txt_filename,
    parse_wf_start_time,
    seconds_since_midnight,
)
from python_magnetrun.utils.timezone import (
    ensure_utc_naive,
    local_to_utc_naive,
    series_local_to_utc_naive,
    series_utc_to_local_naive,
    timerange_to_utc,
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
# local_to_utc_naive
# ---------------------------------------------------------------------------

# Paris CEST (UTC+2) in July 2023: 15:06 local → 13:06 UTC
_PARIS_LOCAL = datetime(2023, 7, 18, 15, 6, 0)
_PARIS_UTC = datetime(2023, 7, 18, 13, 6, 0)

# Paris CET (UTC+1) in January: 10:00 local → 09:00 UTC
_PARIS_LOCAL_WIN = datetime(2023, 1, 18, 10, 0, 0)
_PARIS_UTC_WIN = datetime(2023, 1, 18, 9, 0, 0)


class TestLocalToUtcNaive:
    def test_naive_datetime_summer(self) -> None:
        result = local_to_utc_naive(_PARIS_LOCAL)
        assert result == _PARIS_UTC
        assert result.tzinfo is None

    def test_naive_datetime_winter(self) -> None:
        result = local_to_utc_naive(_PARIS_LOCAL_WIN)
        assert result == _PARIS_UTC_WIN
        assert result.tzinfo is None

    def test_naive_pd_timestamp(self) -> None:
        result = local_to_utc_naive(pd.Timestamp(_PARIS_LOCAL))
        assert result == _PARIS_UTC

    def test_aware_input_not_relocalized(self) -> None:
        # An already UTC-aware timestamp should be passed through unchanged.
        aware = pd.Timestamp("2023-07-18T13:06:00", tz="UTC")
        result = local_to_utc_naive(aware)
        assert result == _PARIS_UTC
        assert result.tzinfo is None

    def test_result_is_datetime(self) -> None:
        assert isinstance(local_to_utc_naive(_PARIS_LOCAL), datetime)


# ---------------------------------------------------------------------------
# ensure_utc_naive
# ---------------------------------------------------------------------------


class TestEnsureUtcNaive:
    def test_naive_assumed_utc(self) -> None:
        naive = datetime(2024, 3, 10, 8, 0, 0)
        result = ensure_utc_naive(naive)
        assert result == naive
        assert result.tzinfo is None

    def test_utc_aware_stripped(self) -> None:
        aware = pd.Timestamp("2024-03-10T08:00:00", tz="UTC")
        result = ensure_utc_naive(aware)
        assert result == datetime(2024, 3, 10, 8, 0, 0)
        assert result.tzinfo is None

    def test_other_tz_converted_to_utc(self) -> None:
        # CET (UTC+1) 09:00 → UTC 08:00
        tz = pytz.timezone("Europe/Paris")
        aware = tz.localize(datetime(2024, 1, 10, 9, 0, 0))
        result = ensure_utc_naive(aware)
        assert result == datetime(2024, 1, 10, 8, 0, 0)
        assert result.tzinfo is None

    def test_result_is_datetime(self) -> None:
        assert isinstance(ensure_utc_naive(datetime(2024, 1, 1)), datetime)


# ---------------------------------------------------------------------------
# series_local_to_utc_naive
# ---------------------------------------------------------------------------


class TestSeriesLocalToUtcNaive:
    def _make_series(self, *datetimes) -> pd.Series:
        return pd.Series(pd.to_datetime(list(datetimes)))

    def test_summer_offset(self) -> None:
        s = self._make_series(_PARIS_LOCAL)
        result = series_local_to_utc_naive(s)
        assert result.iloc[0] == pd.Timestamp(_PARIS_UTC)
        assert result.dt.tz is None

    def test_winter_offset(self) -> None:
        s = self._make_series(_PARIS_LOCAL_WIN)
        result = series_local_to_utc_naive(s)
        assert result.iloc[0] == pd.Timestamp(_PARIS_UTC_WIN)

    def test_multiple_values(self) -> None:
        s = self._make_series(_PARIS_LOCAL, _PARIS_LOCAL_WIN)
        result = series_local_to_utc_naive(s)
        assert result.iloc[0] == pd.Timestamp(_PARIS_UTC)
        assert result.iloc[1] == pd.Timestamp(_PARIS_UTC_WIN)

    def test_result_is_naive(self) -> None:
        s = self._make_series(_PARIS_LOCAL)
        assert series_local_to_utc_naive(s).dt.tz is None


# ---------------------------------------------------------------------------
# series_utc_to_local_naive
# ---------------------------------------------------------------------------


class TestSeriesUtcToLocalNaive:
    def _make_series(self, *datetimes) -> pd.Series:
        return pd.Series(pd.to_datetime(list(datetimes)))

    def test_summer_offset_reversed(self) -> None:
        s = self._make_series(_PARIS_UTC)
        result = series_utc_to_local_naive(s)
        assert result.iloc[0] == pd.Timestamp(_PARIS_LOCAL)
        assert result.dt.tz is None

    def test_winter_offset_reversed(self) -> None:
        s = self._make_series(_PARIS_UTC_WIN)
        result = series_utc_to_local_naive(s)
        assert result.iloc[0] == pd.Timestamp(_PARIS_LOCAL_WIN)

    def test_roundtrip(self) -> None:
        s = self._make_series(_PARIS_LOCAL, _PARIS_LOCAL_WIN)
        utc = series_local_to_utc_naive(s)
        local = series_utc_to_local_naive(utc)
        assert local.iloc[0] == pd.Timestamp(_PARIS_LOCAL)
        assert local.iloc[1] == pd.Timestamp(_PARIS_LOCAL_WIN)

    def test_result_is_naive(self) -> None:
        s = self._make_series(_PARIS_UTC)
        assert series_utc_to_local_naive(s).dt.tz is None


# ---------------------------------------------------------------------------
# timerange_to_utc
# ---------------------------------------------------------------------------


class TestTimerangeToUtc:
    def test_summer_boundaries(self) -> None:
        # Paris CEST (UTC+2): 15:06 → 13:06 UTC, 16:00 → 14:00 UTC
        t_start, t_end = timerange_to_utc(
            "2023-07-18 15:06:00;2023-07-18 16:00:00"
        )
        assert t_start == pd.Timestamp("2023-07-18 13:06:00")
        assert t_end == pd.Timestamp("2023-07-18 14:00:00")

    def test_winter_boundaries(self) -> None:
        # Paris CET (UTC+1): 10:00 → 09:00 UTC, 11:30 → 10:30 UTC
        t_start, t_end = timerange_to_utc(
            "2023-01-18 10:00:00;2023-01-18 11:30:00"
        )
        assert t_start == pd.Timestamp("2023-01-18 09:00:00")
        assert t_end == pd.Timestamp("2023-01-18 10:30:00")

    def test_returns_naive_timestamps(self) -> None:
        t_start, t_end = timerange_to_utc("2023-07-18 15:06:00;2023-07-18 16:00:00")
        assert t_start.tzinfo is None
        assert t_end.tzinfo is None

    def test_start_before_end(self) -> None:
        t_start, t_end = timerange_to_utc("2023-07-18 15:06:00;2023-07-18 16:00:00")
        assert t_start < t_end


# ---------------------------------------------------------------------------
# align_to_common_time
# ---------------------------------------------------------------------------


class _FakeSource:
    """Minimal stub satisfying _HasTimeRange."""

    def __init__(self, start: datetime, end: datetime) -> None:
        self._start = start
        self._end = end

    def get_time_range(self) -> tuple[datetime, datetime]:
        return self._start, self._end


class TestAlignToCommonTime:
    def _make(self, hour: int) -> _FakeSource:
        return _FakeSource(datetime(2025, 1, 27, hour, 0, 0),
                           datetime(2025, 1, 27, hour + 1, 0, 0))

    def test_earliest_source_offset_is_zero(self) -> None:
        from python_magnetrun.utils.timestamps import align_to_common_time

        s10, s11, s12 = self._make(10), self._make(11), self._make(12)
        offsets = align_to_common_time([s10, s11, s12])
        assert offsets[id(s10)] == 0.0

    def test_later_sources_have_positive_offsets(self) -> None:
        from python_magnetrun.utils.timestamps import align_to_common_time

        s10, s11, s12 = self._make(10), self._make(11), self._make(12)
        offsets = align_to_common_time([s10, s11, s12])
        assert offsets[id(s11)] == 3600.0
        assert offsets[id(s12)] == 7200.0

    def test_explicit_reference(self) -> None:
        from python_magnetrun.utils.timestamps import align_to_common_time

        ref = datetime(2025, 1, 27, 10, 0, 0)
        s = self._make(10)
        s_later = _FakeSource(datetime(2025, 1, 27, 10, 5, 0),
                              datetime(2025, 1, 27, 11, 5, 0))
        offsets = align_to_common_time([s, s_later], reference=ref)
        assert offsets[id(s)] == 0.0
        assert offsets[id(s_later)] == 300.0

    def test_single_source_offset_is_zero(self) -> None:
        from python_magnetrun.utils.timestamps import align_to_common_time

        s = self._make(8)
        offsets = align_to_common_time([s])
        assert offsets[id(s)] == 0.0
