"""Unit tests for magnetdata.MagnetData.

Covers: fromtxt, fromtdms (mocked), fromcsv, fromensight, getData, column
renaming, getKeys, getType, addTime, getStartDate, getDuration, shiftTime,
addTdmsTime, extractData, extractDataThreshold, extractTimeData, saveData,
computeData, and realistic data from the data/ directory.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from python_magnetrun.magnetdata import MagnetData

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_TXT = Path(__file__).parent / "data" / "sample_pupitre.txt"


@pytest.fixture()
def txt_magnetdata() -> MagnetData:
    """MagnetData loaded from the sample pupitre txt file."""
    return MagnetData.fromtxt(str(SAMPLE_TXT))


@pytest.fixture()
def simple_df() -> pd.DataFrame:
    """Minimal DataFrame for unit-level testing."""
    return pd.DataFrame(
        {
            "Date": ["2022.03.30"] * 3,
            "Time": ["21:55:17", "21:55:18", "21:55:19"],
            "Field": [0.5, 0.6, 0.7],
            "Icoil1": [1000.0, 1001.0, 1002.0],
            "Ucoil1": [2.5, 2.6, 2.7],
        }
    )


@pytest.fixture()
def simple_magnetdata(simple_df: pd.DataFrame) -> MagnetData:
    """MagnetData built directly from a DataFrame (Type=0)."""
    keys = simple_df.columns.tolist()
    return MagnetData("test.txt", {}, keys, 0, simple_df.copy())


# ---------------------------------------------------------------------------
# fromtxt
# ---------------------------------------------------------------------------


class TestFromtxt:
    def test_loads_sample_file(self) -> None:
        """fromtxt should return a MagnetData instance."""
        md = MagnetData.fromtxt(str(SAMPLE_TXT))
        assert isinstance(md, MagnetData)

    def test_type_is_zero(self, txt_magnetdata: MagnetData) -> None:
        """Pupitre files must have Type=0 (pandas)."""
        assert txt_magnetdata.getType() == 0

    def test_keys_match_header(self, txt_magnetdata: MagnetData) -> None:
        """Keys should match the column names in the txt header row."""
        keys = txt_magnetdata.getKeys()
        assert "Date" in keys
        assert "Time" in keys
        assert "Field" in keys
        assert "Icoil1" in keys

    def test_data_is_dataframe(self, txt_magnetdata: MagnetData) -> None:
        """Data attribute should be a pandas DataFrame."""
        assert isinstance(txt_magnetdata.Data, pd.DataFrame)

    def test_data_has_rows(self, txt_magnetdata: MagnetData) -> None:
        """DataFrame should contain at least one data row."""
        assert len(txt_magnetdata.Data) > 0

    def test_filename_stored(self) -> None:
        """FileName attribute should equal the path passed in."""
        md = MagnetData.fromtxt(str(SAMPLE_TXT))
        assert md.FileName == str(SAMPLE_TXT)

    def test_wrong_extension_raises(self, tmp_path: Path) -> None:
        """fromtxt with a non-.txt extension should raise RuntimeError."""
        bad_file = tmp_path / "data.csv"
        bad_file.write_text("a,b\n1,2\n")
        with pytest.raises(RuntimeError, match="expect a txt filename"):
            MagnetData.fromtxt(str(bad_file))

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """fromtxt with a nonexistent path should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            MagnetData.fromtxt(str(tmp_path / "missing.txt"))


# ---------------------------------------------------------------------------
# fromtdms (mocked — no real .tdms files required in CI)
# ---------------------------------------------------------------------------


def _make_mock_tdms(tmp_path: Path) -> tuple[str, MagicMock]:
    """Return a fake .tdms file path and a matching TdmsFile mock."""
    tdms_path = tmp_path / "M9_Overview_240101-1200.tdms"
    tdms_path.touch()

    channel_mock = MagicMock()
    channel_mock.name = "Courant GR1"
    channel_mock.properties = {"wf_start_offset": 0.5, "wf_increment": 1.0}

    group_df = pd.DataFrame(
        {
            "Courant_GR1": [100.0, 200.0, 300.0],
            "Tension_GR1": [1.0, 2.0, 3.0],
        }
    )

    group_mock = MagicMock()
    group_mock.name = "Courants Alimentations"
    group_mock.channels.return_value = [channel_mock]
    group_mock.as_dataframe.return_value = group_df

    tdms_mock = MagicMock()
    tdms_mock.groups.return_value = [group_mock]

    return str(tdms_path), tdms_mock


class TestFromtdms:
    def test_loads_valid_tdms(self, tmp_path: Path) -> None:
        """fromtdms should return a MagnetData with Type=1."""
        tdms_path, tdms_mock = _make_mock_tdms(tmp_path)
        with patch("nptdms.TdmsFile") as MockTdms:
            MockTdms.open.return_value = tdms_mock
            md = MagnetData.fromtdms(tdms_path)
        assert isinstance(md, MagnetData)
        assert md.getType() == 1

    def test_type_is_one(self, tmp_path: Path) -> None:
        """TDMS data must have Type=1."""
        tdms_path, tdms_mock = _make_mock_tdms(tmp_path)
        with patch("nptdms.TdmsFile") as MockTdms:
            MockTdms.open.return_value = tdms_mock
            md = MagnetData.fromtdms(tdms_path)
        assert md.Type == 1

    def test_data_is_dict(self, tmp_path: Path) -> None:
        """TDMS Data should be stored as a dict of DataFrames."""
        tdms_path, tdms_mock = _make_mock_tdms(tmp_path)
        with patch("nptdms.TdmsFile") as MockTdms:
            MockTdms.open.return_value = tdms_mock
            md = MagnetData.fromtdms(tdms_path)
        assert isinstance(md.Data, dict)

    def test_groups_populated(self, tmp_path: Path) -> None:
        """Groups dict should contain the group from the tdms file."""
        tdms_path, tdms_mock = _make_mock_tdms(tmp_path)
        with patch("nptdms.TdmsFile") as MockTdms:
            MockTdms.open.return_value = tdms_mock
            md = MagnetData.fromtdms(tdms_path)
        assert "Courants_Alimentations" in md.Groups

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        """fromtdms should raise FileNotFoundError for a missing file."""
        with pytest.raises(FileNotFoundError):
            MagnetData.fromtdms(str(tmp_path / "nonexistent.tdms"))

    def test_wrong_extension_raises(self, tmp_path: Path) -> None:
        """fromtdms with a non-.tdms filename should raise RuntimeError."""
        bad = tmp_path / "data.txt"
        bad.touch()
        with pytest.raises(RuntimeError, match="expect a tdms filename"):
            MagnetData.fromtdms(str(bad))

    def test_missing_courants_group_raises(self, tmp_path: Path) -> None:
        """fromtdms must raise RuntimeError when Courants_Alimentations is absent."""
        tdms_path = tmp_path / "M9_Overview_240101-1200.tdms"
        tdms_path.touch()

        group_mock = MagicMock()
        group_mock.name = "OtherGroup"
        group_mock.channels.return_value = []
        group_mock.as_dataframe.return_value = pd.DataFrame()

        tdms_mock = MagicMock()
        tdms_mock.groups.return_value = [group_mock]

        with patch("nptdms.TdmsFile") as MockTdms:
            MockTdms.open.return_value = tdms_mock
            with pytest.raises(RuntimeError, match="Courants_Alimentations"):
                MagnetData.fromtdms(str(tdms_path))

    def test_overview_t_offset(self, tmp_path: Path) -> None:
        """Overview files should apply a non-zero wf_start_offset."""
        # The t_offset for Overview is 0.5 — verify it is written back
        tdms_path = tmp_path / "M9_Overview_240101-1200.tdms"
        tdms_path.touch()

        channel_mock = MagicMock()
        channel_mock.name = "Courant GR1"
        channel_mock.properties = {"wf_start_offset": 99.9, "wf_increment": 1.0}

        group_mock = MagicMock()
        group_mock.name = "Courants Alimentations"
        group_mock.channels.return_value = [channel_mock]
        group_mock.as_dataframe.return_value = pd.DataFrame({"Courant_GR1": [1.0]})

        tdms_mock = MagicMock()
        tdms_mock.groups.return_value = [group_mock]

        with patch("nptdms.TdmsFile") as MockTdms:
            MockTdms.open.return_value = tdms_mock
            md = MagnetData.fromtdms(str(tdms_path))

        # wf_start_offset should be overwritten to 0.5 for Overview files
        assert md.Groups["Courants_Alimentations"]["Courant_GR1"]["wf_start_offset"] == 0.5


# ---------------------------------------------------------------------------
# getData — Type=0 (pandas)
# ---------------------------------------------------------------------------


class TestGetDataPandas:
    def test_none_key_returns_full_dataframe(self, simple_magnetdata: MagnetData) -> None:
        """getData(None) should return the complete DataFrame."""
        df = simple_magnetdata.getData(None)
        assert isinstance(df, pd.DataFrame)
        assert len(df.columns) == len(simple_magnetdata.Keys)

    def test_string_key_returns_single_column(self, simple_magnetdata: MagnetData) -> None:
        """getData('Field') should return a one-column DataFrame."""
        df = simple_magnetdata.getData("Field")
        assert "Field" in df.columns
        assert len(df.columns) == 1

    def test_list_key_returns_selected_columns(self, simple_magnetdata: MagnetData) -> None:
        """getData(['Field', 'Icoil1']) should return both columns."""
        df = simple_magnetdata.getData(["Field", "Icoil1"])
        assert set(df.columns) == {"Field", "Icoil1"}

    def test_invalid_key_raises(self, simple_magnetdata: MagnetData) -> None:
        """getData with a non-existent key should raise an Exception."""
        with pytest.raises(Exception, match="no such key"):
            simple_magnetdata.getData("NonExistent")

    def test_invalid_key_in_list_raises(self, simple_magnetdata: MagnetData) -> None:
        """getData with a list containing an unknown key should raise."""
        with pytest.raises(Exception, match="no such key"):
            simple_magnetdata.getData(["Field", "NonExistent"])

    def test_data_values_preserved(
        self, simple_df: pd.DataFrame, simple_magnetdata: MagnetData
    ) -> None:
        """getData should return the same numeric values as the source DataFrame."""
        result = simple_magnetdata.getData("Field")
        pd.testing.assert_series_equal(
            result["Field"].reset_index(drop=True),
            simple_df["Field"].reset_index(drop=True),
        )


# ---------------------------------------------------------------------------
# getData — Type=1 (TDMS / dict)
# ---------------------------------------------------------------------------


class TestGetDataTdms:
    @pytest.fixture()
    def tdms_magnetdata(self) -> MagnetData:
        group_df = pd.DataFrame(
            {
                "Courant_GR1": [100.0, 200.0, 300.0],
                "Tension_GR1": [1.0, 2.0, 3.0],
            }
        )
        keys = [
            "Courants_Alimentations/Courant_GR1",
            "Courants_Alimentations/Tension_GR1",
        ]
        data = {"Courants_Alimentations": group_df}
        groups = {
            "Courants_Alimentations": {
                "Courant_GR1": {"wf_increment": 1.0},
                "Tension_GR1": {"wf_increment": 1.0},
            }
        }
        return MagnetData("test.tdms", groups, keys, 1, data)

    def test_group_slash_channel(self, tdms_magnetdata: MagnetData) -> None:
        """getData('Group/Channel') should return the channel's DataFrame."""
        df = tdms_magnetdata.getData("Courants_Alimentations/Courant_GR1")
        assert isinstance(df, pd.DataFrame)

    def test_list_of_group_slash_channels(self, tdms_magnetdata: MagnetData) -> None:
        """getData([...]) should work for multiple channels in the same group."""
        df = tdms_magnetdata.getData(
            [
                "Courants_Alimentations/Courant_GR1",
                "Courants_Alimentations/Tension_GR1",
            ]
        )
        assert isinstance(df, pd.DataFrame)
        assert "Courant_GR1" in df.columns
        assert "Tension_GR1" in df.columns

    def test_multiple_groups_raises(self, tdms_magnetdata: MagnetData) -> None:
        """getData across multiple groups should raise RuntimeError."""
        with pytest.raises(RuntimeError, match="expect only one group"):
            tdms_magnetdata.getData(
                [
                    "Courants_Alimentations/Courant_GR1",
                    "AnotherGroup/Something",
                ]
            )

    def test_unsupported_type_raises(self) -> None:
        """getData on Type=2 should raise RuntimeError."""
        md = MagnetData("test.ensight", {}, [], 2, pd.DataFrame())
        with pytest.raises(RuntimeError, match="not implemented"):
            md.getData("anything")


# ---------------------------------------------------------------------------
# getKeys / getType
# ---------------------------------------------------------------------------


class TestGetKeysAndType:
    def test_getkeys_returns_list(self, simple_magnetdata: MagnetData) -> None:
        assert isinstance(simple_magnetdata.getKeys(), list)

    def test_getkeys_matches_dataframe_columns(self, simple_magnetdata: MagnetData) -> None:
        assert simple_magnetdata.getKeys() == simple_magnetdata.Data.columns.tolist()  # type: ignore[union-attr]

    def test_gettype_pandas(self, simple_magnetdata: MagnetData) -> None:
        assert simple_magnetdata.getType() == 0

    def test_gettype_tdms(self) -> None:
        md = MagnetData("x.tdms", {}, [], 1, {})
        assert md.getType() == 1

    def test_gettype_ensight(self) -> None:
        md = MagnetData("x.ensight", {}, [], 2, pd.DataFrame())
        assert md.getType() == 2


# ---------------------------------------------------------------------------
# renameData
# ---------------------------------------------------------------------------


class TestRenameData:
    def test_renames_existing_column(self, simple_magnetdata: MagnetData) -> None:
        """renameData should rename an existing column."""
        simple_magnetdata.renameData({"Field": "B"})
        assert "B" in simple_magnetdata.Keys
        assert "Field" not in simple_magnetdata.Keys
        assert "B" in simple_magnetdata.Data.columns  # type: ignore[union-attr]

    def test_keys_updated_after_rename(self, simple_magnetdata: MagnetData) -> None:
        """Keys list must stay in sync with DataFrame columns after rename."""
        simple_magnetdata.renameData({"Icoil1": "IH"})
        assert simple_magnetdata.Keys == simple_magnetdata.Data.columns.tolist()  # type: ignore[union-attr]

    def test_missing_key_is_skipped(self, simple_magnetdata: MagnetData) -> None:
        """renameData with a missing source key should not raise, just skip."""
        original_keys = list(simple_magnetdata.Keys)
        simple_magnetdata.renameData({"NonExistent": "NewName"})
        assert simple_magnetdata.Keys == original_keys

    def test_noop_for_type_1(self) -> None:
        """renameData on TDMS data (Type=1) should do nothing."""
        md = MagnetData("x.tdms", {}, ["GroupA/ch"], 1, {"GroupA": pd.DataFrame({"ch": [1]})})
        md.renameData({"GroupA/ch": "GroupA/new"})
        assert md.Keys == ["GroupA/ch"]


# ---------------------------------------------------------------------------
# removeData
# ---------------------------------------------------------------------------


class TestRemoveData:
    def test_removes_existing_column(self, simple_magnetdata: MagnetData) -> None:
        simple_magnetdata.removeData(["Field"])
        assert "Field" not in simple_magnetdata.Keys
        assert "Field" not in simple_magnetdata.Data.columns  # type: ignore[union-attr]

    def test_missing_key_is_skipped(self, simple_magnetdata: MagnetData) -> None:
        """removeData with an absent key should not raise."""
        original_len = len(simple_magnetdata.Keys)
        simple_magnetdata.removeData(["NonExistent"])
        assert len(simple_magnetdata.Keys) == original_len

    def test_keys_updated_after_remove(self, simple_magnetdata: MagnetData) -> None:
        simple_magnetdata.removeData(["Icoil1"])
        assert simple_magnetdata.Keys == simple_magnetdata.Data.columns.tolist()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# addData
# ---------------------------------------------------------------------------


class TestAddData:
    def test_adds_new_column_from_formula(self, simple_magnetdata: MagnetData) -> None:
        """addData should evaluate a formula and add the resulting column."""
        simple_magnetdata.addData("Field2", "Field2 = Field * 2")
        assert "Field2" in simple_magnetdata.Keys
        assert "Field2" in simple_magnetdata.Data.columns  # type: ignore[union-attr]

    def test_computed_values_correct(self, simple_magnetdata: MagnetData) -> None:
        """The new column should contain the correct computed values."""
        simple_magnetdata.addData("Field2", "Field2 = Field * 2")
        expected = simple_magnetdata.Data["Field"] * 2
        pd.testing.assert_series_equal(
            simple_magnetdata.Data["Field2"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_duplicate_key_is_skipped(self, simple_magnetdata: MagnetData) -> None:
        """addData with an already-existing key should not modify the column."""
        original_values = simple_magnetdata.Data["Field"].copy()
        simple_magnetdata.addData("Field", "Field = Field * 999")
        pd.testing.assert_series_equal(simple_magnetdata.Data["Field"], original_values)


# ---------------------------------------------------------------------------
# fromStringIO
# ---------------------------------------------------------------------------


class TestFromStringIO:
    def test_loads_from_string(self) -> None:
        """fromStringIO should create a valid MagnetData from a string."""
        content = "header line skipped\nA B C\n1 2 3\n4 5 6\n"
        md = MagnetData.fromStringIO(content)
        assert isinstance(md, MagnetData)
        assert "A" in md.Keys

    def test_type_is_zero(self) -> None:
        content = "skip\nX Y\n1 2\n"
        md = MagnetData.fromStringIO(content)
        assert md.getType() == 0

    def test_filename_is_stringio(self) -> None:
        content = "skip\nX Y\n1 2\n"
        md = MagnetData.fromStringIO(content)
        assert md.FileName == "stringIO"


# ---------------------------------------------------------------------------
# fromcsv
# ---------------------------------------------------------------------------


class TestFromCsv:
    @pytest.fixture()
    def csv_file(self, tmp_path: Path) -> Path:
        p = tmp_path / "data.csv"
        p.write_text("A,B,C\n1,2,3\n4,5,6\n")
        return p

    def test_loads_csv_file(self, csv_file: Path) -> None:
        md = MagnetData.fromcsv(str(csv_file))
        assert isinstance(md, MagnetData)

    def test_type_is_zero(self, csv_file: Path) -> None:
        md = MagnetData.fromcsv(str(csv_file))
        assert md.getType() == 0

    def test_keys_match_header(self, csv_file: Path) -> None:
        md = MagnetData.fromcsv(str(csv_file))
        assert md.getKeys() == ["A", "B", "C"]

    def test_data_is_dataframe(self, csv_file: Path) -> None:
        md = MagnetData.fromcsv(str(csv_file))
        assert isinstance(md.Data, pd.DataFrame)

    def test_filename_stored(self, csv_file: Path) -> None:
        md = MagnetData.fromcsv(str(csv_file))
        assert md.FileName == str(csv_file)

    def test_data_values_correct(self, csv_file: Path) -> None:
        md = MagnetData.fromcsv(str(csv_file))
        assert md.Data["A"].tolist() == [1, 4]


# ---------------------------------------------------------------------------
# fromensight
# ---------------------------------------------------------------------------


class TestFromEnsight:
    @pytest.fixture()
    def ensight_file(self, tmp_path: Path) -> Path:
        p = tmp_path / "result.csv"
        # Ensight CSV has 2 header rows before the column names
        p.write_text("Ensight header line 1\nEnsight header line 2\nX,Y,Z\n0.1,0.2,0.3\n0.4,0.5,0.6\n")
        return p

    def test_loads_ensight_file(self, ensight_file: Path) -> None:
        md = MagnetData.fromensight(str(ensight_file))
        assert isinstance(md, MagnetData)

    def test_type_is_two(self, ensight_file: Path) -> None:
        md = MagnetData.fromensight(str(ensight_file))
        assert md.getType() == 2

    def test_keys_match_header(self, ensight_file: Path) -> None:
        md = MagnetData.fromensight(str(ensight_file))
        assert md.getKeys() == ["X", "Y", "Z"]

    def test_data_is_dataframe(self, ensight_file: Path) -> None:
        md = MagnetData.fromensight(str(ensight_file))
        assert isinstance(md.Data, pd.DataFrame)

    def test_data_values_correct(self, ensight_file: Path) -> None:
        md = MagnetData.fromensight(str(ensight_file))
        assert len(md.Data) == 2
        assert round(md.Data["X"].iloc[0], 1) == 0.1


# ---------------------------------------------------------------------------
# getStartDate
# ---------------------------------------------------------------------------


class TestGetStartDate:
    def test_returns_four_tuple(self, simple_magnetdata: MagnetData) -> None:
        result = simple_magnetdata.getStartDate()
        assert len(result) == 4

    def test_correct_start_date(self, simple_magnetdata: MagnetData) -> None:
        result = simple_magnetdata.getStartDate()
        assert result[0] == "2022.03.30"

    def test_correct_start_time(self, simple_magnetdata: MagnetData) -> None:
        result = simple_magnetdata.getStartDate()
        assert result[1] == "21:55:17"

    def test_correct_end_time(self, simple_magnetdata: MagnetData) -> None:
        result = simple_magnetdata.getStartDate()
        assert result[3] == "21:55:19"

    def test_empty_when_no_date_columns(self) -> None:
        md = MagnetData("x.txt", {}, ["A", "B"], 0, pd.DataFrame({"A": [1], "B": [2]}))
        assert md.getStartDate() == ()


# ---------------------------------------------------------------------------
# addTime / getDuration / shiftTime
# ---------------------------------------------------------------------------


class TestAddTime:
    def test_adds_timestamp_column(self, simple_magnetdata: MagnetData) -> None:
        simple_magnetdata.addTime()
        assert "timestamp" in simple_magnetdata.Keys

    def test_adds_t_column(self, simple_magnetdata: MagnetData) -> None:
        simple_magnetdata.addTime()
        assert "t" in simple_magnetdata.Keys

    def test_drops_date_and_time(self, simple_magnetdata: MagnetData) -> None:
        simple_magnetdata.addTime()
        assert "Date" not in simple_magnetdata.Keys
        assert "Time" not in simple_magnetdata.Keys

    def test_t_starts_at_zero(self, simple_magnetdata: MagnetData) -> None:
        simple_magnetdata.addTime()
        assert simple_magnetdata.Data["t"].iloc[0] == pytest.approx(0.0)

    def test_t_values_one_second_apart(self, simple_magnetdata: MagnetData) -> None:
        """Sample data has 1-second intervals."""
        simple_magnetdata.addTime()
        diffs = simple_magnetdata.Data["t"].diff().dropna()
        assert all(abs(d - 1.0) < 1e-6 for d in diffs)

    def test_missing_date_raises(self) -> None:
        md = MagnetData("x.txt", {}, ["Field"], 0, pd.DataFrame({"Field": [1.0]}))
        with pytest.raises(RuntimeError, match="no Date or Time"):
            md.addTime()


class TestGetDuration:
    def test_duration_in_seconds(self, simple_magnetdata: MagnetData) -> None:
        """3-row fixture with 1-second intervals → duration = 2 s."""
        simple_magnetdata.addTime()
        assert simple_magnetdata.getDuration() == pytest.approx(2.0)

    def test_zero_when_no_timestamp(self, simple_magnetdata: MagnetData) -> None:
        """Without calling addTime first, getDuration returns 0."""
        assert simple_magnetdata.getDuration() == 0


class TestShiftTime:
    def test_shifts_t_column(self, simple_magnetdata: MagnetData) -> None:
        simple_magnetdata.addTime()
        t_before = simple_magnetdata.Data["t"].copy()
        simple_magnetdata.shiftTime(10.0)
        assert (simple_magnetdata.Data["t"] - t_before).round(6).eq(10.0).all()

    def test_missing_t_raises(self, simple_magnetdata: MagnetData) -> None:
        with pytest.raises(RuntimeError, match="no t column"):
            simple_magnetdata.shiftTime(5.0)


# ---------------------------------------------------------------------------
# addTdmsTime
# ---------------------------------------------------------------------------


class TestAddTdmsTime:
    @pytest.fixture()
    def tdms_md(self) -> MagnetData:
        group_df = pd.DataFrame({"Courant_GR1": [100.0, 200.0, 300.0]})
        groups = {
            "Courants_Alimentations": {
                "Courant_GR1": {"wf_increment": 0.5, "wf_start_offset": 1.0},
            }
        }
        keys = ["Courants_Alimentations/Courant_GR1"]
        return MagnetData("x.tdms", groups, keys, 1, {"Courants_Alimentations": group_df})

    def test_adds_t_column(self, tdms_md: MagnetData) -> None:
        tdms_md.addTdmsTime()
        assert "t" in tdms_md.Data["Courants_Alimentations"].columns

    def test_t_values_correct(self, tdms_md: MagnetData) -> None:
        tdms_md.addTdmsTime()
        t = tdms_md.Data["Courants_Alimentations"]["t"].tolist()
        # t = index * 0.5 + 1.0  →  [1.0, 1.5, 2.0]
        assert t == pytest.approx([1.0, 1.5, 2.0])

    def test_key_added(self, tdms_md: MagnetData) -> None:
        tdms_md.addTdmsTime()
        assert "Courants_Alimentations/t" in tdms_md.Keys

    def test_idempotent(self, tdms_md: MagnetData) -> None:
        tdms_md.addTdmsTime()
        tdms_md.addTdmsTime()
        assert tdms_md.Keys.count("Courants_Alimentations/t") == 1

    def test_raises_for_non_tdms(self, simple_magnetdata: MagnetData) -> None:
        with pytest.raises(RuntimeError, match="only applicable for Type=1"):
            simple_magnetdata.addTdmsTime()

    def test_raises_for_missing_group(self, tdms_md: MagnetData) -> None:
        with pytest.raises(RuntimeError, match="group 'Unknown' not found"):
            tdms_md.addTdmsTime(group="Unknown")


# ---------------------------------------------------------------------------
# extractData
# ---------------------------------------------------------------------------


class TestExtractData:
    def test_returns_selected_columns(self, simple_magnetdata: MagnetData) -> None:
        df = simple_magnetdata.extractData(["Field", "Icoil1"])
        assert list(df.columns) == ["Field", "Icoil1"]

    def test_values_match_original(self, simple_df: pd.DataFrame, simple_magnetdata: MagnetData) -> None:
        df = simple_magnetdata.extractData(["Field"])
        pd.testing.assert_series_equal(
            df["Field"].reset_index(drop=True),
            simple_df["Field"].reset_index(drop=True),
        )

    def test_unknown_key_raises(self, simple_magnetdata: MagnetData) -> None:
        with pytest.raises(RuntimeError):
            simple_magnetdata.extractData(["NonExistent"])

    def test_tdms_group_slash_channel(self) -> None:
        group_df = pd.DataFrame({"Courant_GR1": [1.0, 2.0], "Tension_GR1": [3.0, 4.0]})
        groups = {"G": {"Courant_GR1": {"wf_increment": 1.0}, "Tension_GR1": {"wf_increment": 1.0}}}
        md = MagnetData("x.tdms", groups, ["G/Courant_GR1", "G/Tension_GR1"], 1, {"G": group_df})
        df = md.extractData(["G/Courant_GR1", "G/Tension_GR1"])
        assert "Courant_GR1" in df.columns
        assert "Tension_GR1" in df.columns


# ---------------------------------------------------------------------------
# extractDataThreshold
# ---------------------------------------------------------------------------


class TestExtractDataThreshold:
    def test_filters_rows_at_or_above_threshold(self, simple_magnetdata: MagnetData) -> None:
        # Field values: [0.5, 0.6, 0.7] — threshold 0.6 keeps rows 1 and 2
        df = simple_magnetdata.extractDataThreshold("Field", 0.6)
        assert len(df) == 2
        assert (df["Field"] >= 0.6).all()

    def test_all_below_threshold_returns_empty(self, simple_magnetdata: MagnetData) -> None:
        df = simple_magnetdata.extractDataThreshold("Field", 999.0)
        assert len(df) == 0

    def test_all_at_or_above_threshold(self, simple_magnetdata: MagnetData) -> None:
        df = simple_magnetdata.extractDataThreshold("Field", 0.0)
        assert len(df) == 3

    def test_unknown_key_raises(self, simple_magnetdata: MagnetData) -> None:
        with pytest.raises(RuntimeError):
            simple_magnetdata.extractDataThreshold("NonExistent", 0.5)


# ---------------------------------------------------------------------------
# extractTimeData
# ---------------------------------------------------------------------------


class TestExtractTimeData:
    def test_filters_by_time_range(self, simple_magnetdata: MagnetData) -> None:
        # Time values: ["21:55:17", "21:55:18", "21:55:19"]
        df = simple_magnetdata.extractTimeData("21:55:17;21:55:18")
        assert len(df) == 2

    def test_inclusive_boundaries(self, simple_magnetdata: MagnetData) -> None:
        df = simple_magnetdata.extractTimeData("21:55:17;21:55:17")
        assert len(df) == 1
        assert df["Time"].iloc[0] == "21:55:17"

    def test_empty_range(self, simple_magnetdata: MagnetData) -> None:
        df = simple_magnetdata.extractTimeData("22:00:00;22:59:59")
        assert len(df) == 0


# ---------------------------------------------------------------------------
# saveData
# ---------------------------------------------------------------------------


class TestSaveData:
    def test_creates_file(self, simple_magnetdata: MagnetData, tmp_path: Path) -> None:
        out = tmp_path / "out.csv"
        simple_magnetdata.saveData(["Field", "Icoil1"], str(out))
        assert out.exists()

    def test_saved_file_has_correct_columns(self, simple_magnetdata: MagnetData, tmp_path: Path) -> None:
        out = tmp_path / "out.csv"
        simple_magnetdata.saveData(["Field", "Icoil1"], str(out))
        df = pd.read_csv(out, sep="\t")
        assert list(df.columns) == ["Field", "Icoil1"]

    def test_saved_values_match(self, simple_df: pd.DataFrame, simple_magnetdata: MagnetData, tmp_path: Path) -> None:
        out = tmp_path / "out.csv"
        simple_magnetdata.saveData(["Field"], str(out))
        df = pd.read_csv(out, sep="\t")
        pd.testing.assert_series_equal(
            df["Field"].reset_index(drop=True),
            simple_df["Field"].reset_index(drop=True),
        )


# ---------------------------------------------------------------------------
# computeData
# ---------------------------------------------------------------------------


class TestComputeData:
    def test_adds_column_from_function(self, simple_magnetdata: MagnetData) -> None:
        simple_magnetdata.computeData(lambda f, i: f * i, "Product", ["Field", "Icoil1"])
        assert "Product" in simple_magnetdata.Keys
        assert "Product" in simple_magnetdata.Data.columns

    def test_computed_values_correct(self, simple_df: pd.DataFrame, simple_magnetdata: MagnetData) -> None:
        simple_magnetdata.computeData(lambda f, i: f * i, "Product", ["Field", "Icoil1"])
        expected = simple_df["Field"] * simple_df["Icoil1"]
        pd.testing.assert_series_equal(
            simple_magnetdata.Data["Product"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_existing_key_is_skipped(self, simple_magnetdata: MagnetData) -> None:
        original = simple_magnetdata.Data["Field"].copy()
        simple_magnetdata.computeData(lambda f: f * 999, "Field", ["Field"])
        pd.testing.assert_series_equal(simple_magnetdata.Data["Field"], original)


# ===========================================================================
# Realistic data from data/ directory
# ===========================================================================

DATA_DIR = Path(__file__).parent.parent / "data"

M9_TXT = DATA_DIR / "M9_2019.02.14-23_00_38.txt"
M10_TXT = DATA_DIR / "M10_2020.10.23---20_10_41.txt"
M9_CSV = DATA_DIR / "M9_2019.06.20-14_36_30.csv"
M9_TDMS = DATA_DIR / "M9_Default_200921-123303_Courants50Hz.tdms"
M9_PROFILES = DATA_DIR / "M9_profiles.txt"

needs_data = pytest.mark.skipif(
    not DATA_DIR.exists(),
    reason="data/ directory not present",
)


# ---------------------------------------------------------------------------
# Realistic fromtxt — M9 pupitre run
# ---------------------------------------------------------------------------


@needs_data
class TestRealisticM9Txt:
    @pytest.fixture(scope="class")
    def m9(self) -> MagnetData:
        return MagnetData.fromtxt(str(M9_TXT))

    def test_row_count(self, m9: MagnetData) -> None:
        assert len(m9.Data) == 26723

    def test_standard_keys_present(self, m9: MagnetData) -> None:
        for col in ("Date", "Time", "Field", "Tin1", "Tin2", "Tout"):
            assert col in m9.Keys

    def test_type_is_zero(self, m9: MagnetData) -> None:
        assert m9.getType() == 0

    def test_start_date(self, m9: MagnetData) -> None:
        start_date, start_time, *_ = m9.getStartDate()
        assert start_date == "2019.02.14"
        assert start_time == "23:00:39"

    def test_end_date(self, m9: MagnetData) -> None:
        *_, end_date, end_time = m9.getStartDate()
        assert end_date == "2019.02.15"
        assert end_time == "06:26:11"

    def test_field_max(self, m9: MagnetData) -> None:
        assert m9.Data["Field"].max() == pytest.approx(25.6514, abs=0.01)

    def test_field_values_nonnegative(self, m9: MagnetData) -> None:
        assert (m9.Data["Field"] >= 0).all()

    def test_add_time_creates_t(self, m9: MagnetData) -> None:
        """addTime adds t and timestamp, removes Date and Time."""
        md = MagnetData.fromtxt(str(M9_TXT))
        md.addTime()
        assert "t" in md.Keys
        assert "timestamp" in md.Keys
        assert "Date" not in md.Keys
        assert "Time" not in md.Keys

    def test_duration(self) -> None:
        """Duration of the ~7.4 h run should be ~26732 s."""
        md = MagnetData.fromtxt(str(M9_TXT))
        md.addTime()
        assert md.getDuration() == pytest.approx(26732, abs=5)

    def test_extract_threshold_field_20(self, m9: MagnetData) -> None:
        """Most rows have Field >= 20 T for this high-field run."""
        df = m9.extractDataThreshold("Field", 20.0)
        assert len(df) == 26123
        assert (df["Field"] >= 20.0).all()

    def test_extract_time_data(self, m9: MagnetData) -> None:
        df = m9.extractTimeData("23:00:39;23:00:41")
        assert len(df) == 3
        assert df["Time"].iloc[0] == "23:00:39"
        assert df["Time"].iloc[-1] == "23:00:41"

    def test_save_and_reload(self, m9: MagnetData, tmp_path: Path) -> None:
        out = tmp_path / "m9_subset.csv"
        m9.saveData(["Field", "Icoil1"], str(out))
        df = pd.read_csv(out, sep="\t")
        assert list(df.columns) == ["Field", "Icoil1"]
        assert len(df) == len(m9.Data)


# ---------------------------------------------------------------------------
# Realistic fromtxt — M10 pupitre run
# ---------------------------------------------------------------------------


@needs_data
class TestRealisticM10Txt:
    @pytest.fixture(scope="class")
    def m10(self) -> MagnetData:
        return MagnetData.fromtxt(str(M10_TXT))

    def test_row_count(self, m10: MagnetData) -> None:
        assert len(m10.Data) == 10328

    def test_standard_keys_present(self, m10: MagnetData) -> None:
        for col in ("Date", "Time", "Field", "Icoil1"):
            assert col in m10.Keys

    def test_type_is_zero(self, m10: MagnetData) -> None:
        assert m10.getType() == 0

    def test_start_date(self, m10: MagnetData) -> None:
        start_date, start_time, *_ = m10.getStartDate()
        assert start_date == "2020.10.23"
        assert start_time == "20:10:42"

    def test_different_column_count_from_m9(self, m10: MagnetData) -> None:
        """M10 and M9 have different column sets (different magnet configs)."""
        m9 = MagnetData.fromtxt(str(M9_TXT))
        assert len(m10.Keys) != len(m9.Keys)


# ---------------------------------------------------------------------------
# Realistic fromcsv — M9 comma-separated run file
# ---------------------------------------------------------------------------


@needs_data
class TestRealisticM9Csv:
    @pytest.fixture(scope="class")
    def m9csv(self) -> MagnetData:
        return MagnetData.fromcsv(str(M9_CSV))

    def test_row_count(self, m9csv: MagnetData) -> None:
        assert len(m9csv.Data) == 4037

    def test_type_is_zero(self, m9csv: MagnetData) -> None:
        assert m9csv.getType() == 0

    def test_standard_keys_present(self, m9csv: MagnetData) -> None:
        for col in ("Date", "Time", "Field", "Tin1"):
            assert col in m9csv.Keys

    def test_duration(self) -> None:
        md = MagnetData.fromcsv(str(M9_CSV))
        md.addTime()
        assert md.getDuration() == pytest.approx(4045, abs=5)

    def test_getData_field_column(self, m9csv: MagnetData) -> None:
        df = m9csv.getData("Field")
        assert "Field" in df.columns
        assert len(df) == 4037


# ---------------------------------------------------------------------------
# Realistic fromcsv — M9 profiles (bprofile-style indexed CSV)
# ---------------------------------------------------------------------------


@needs_data
class TestRealisticM9Profiles:
    @pytest.fixture(scope="class")
    def profiles(self) -> MagnetData:
        return MagnetData.fromcsv(str(M9_PROFILES))

    def test_row_count(self, profiles: MagnetData) -> None:
        assert len(profiles.Data) == 2999

    def test_type_is_zero(self, profiles: MagnetData) -> None:
        assert profiles.getType() == 0

    def test_keys(self, profiles: MagnetData) -> None:
        assert "Index" in profiles.Keys
        assert "Position (mm)" in profiles.Keys
        assert "Profile at Tr (%)" in profiles.Keys
        assert "Profile at max (%)" in profiles.Keys

    def test_position_range(self, profiles: MagnetData) -> None:
        pos = profiles.Data["Position (mm)"]
        assert pos.min() == pytest.approx(299.7, abs=0.1)

    def test_getData_position(self, profiles: MagnetData) -> None:
        df = profiles.getData("Position (mm)")
        assert len(df) == 2999


# ---------------------------------------------------------------------------
# Realistic fromtdms — M9 50 Hz Courants file
# ---------------------------------------------------------------------------


@needs_data
class TestRealisticM9Tdms:
    @pytest.fixture(scope="class")
    def tdms(self) -> MagnetData:
        return MagnetData.fromtdms(str(M9_TDMS))

    def test_type_is_one(self, tdms: MagnetData) -> None:
        assert tdms.getType() == 1

    def test_expected_groups(self, tdms: MagnetData) -> None:
        for grp in (
            "Courants_Alimentations",
            "Tensions_Alimentations",
            "Tensions_Aimant",
            "Courants_ponts",
        ):
            assert grp in tdms.Groups

    def test_courants_channels(self, tdms: MagnetData) -> None:
        channels = list(tdms.Groups["Courants_Alimentations"].keys())
        assert "Courant_A1" in channels
        assert "Courant_A2" in channels

    def test_data_row_count(self, tdms: MagnetData) -> None:
        """432 000 samples at 4800 Hz = 90 s."""
        assert len(tdms.Data["Courants_Alimentations"]) == 432000

    def test_wf_increment(self, tdms: MagnetData) -> None:
        dt = tdms.Groups["Courants_Alimentations"]["Courant_A1"]["wf_increment"]
        assert dt == pytest.approx(1 / 4800, rel=1e-4)

    def test_duration(self, tdms: MagnetData) -> None:
        assert tdms.getDuration("Courants_Alimentations") == pytest.approx(90.0, abs=0.1)

    def test_getData_single_channel(self, tdms: MagnetData) -> None:
        df = tdms.getData("Courants_Alimentations/Courant_A1")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 432000

    def test_getData_multiple_channels(self, tdms: MagnetData) -> None:
        df = tdms.getData(
            ["Courants_Alimentations/Courant_A1", "Courants_Alimentations/Courant_A2"]
        )
        assert "Courant_A1" in df.columns
        assert "Courant_A2" in df.columns

    def test_add_tdms_time(self) -> None:
        """addTdmsTime should produce t from 0 to ~90 s."""
        md = MagnetData.fromtdms(str(M9_TDMS))
        md.addTdmsTime(group="Courants_Alimentations")
        t = md.Data["Courants_Alimentations"]["t"]
        assert t.iloc[0] == pytest.approx(0.0, abs=1e-6)
        assert t.iloc[-1] == pytest.approx(90.0, abs=0.01)
        assert t.is_monotonic_increasing

    def test_extract_data_channel(self, tdms: MagnetData) -> None:
        df = tdms.extractData(
            ["Courants_Alimentations/Courant_A1", "Courants_Alimentations/Courant_A2"]
        )
        assert "Courant_A1" in df.columns
        assert len(df) == 432000
