"""Unit tests for magnetdata.MagnetData.

Covers: fromtxt, fromtdms (mocked), getData, column renaming, getKeys, getType.
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
