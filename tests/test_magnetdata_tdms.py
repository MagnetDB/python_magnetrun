"""Unit tests for TdmsMagnetData ETL methods.

Covers:
  - addTime() delegates to addTdmsTime()
  - cleanupData(keys_to_add=...) calls addData for each entry
  - cleanupData(keys_to_remove=...) drops column from Data and Keys
  - cleanupData(keys_to_rename=...) emits logger.warning without raising
"""

import logging

import pandas as pd

from python_magnetrun.magnetdata_tdms import TdmsMagnetData

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tdms(groups: dict | None = None) -> TdmsMagnetData:
    """Build a minimal TdmsMagnetData with one group containing two channels."""
    if groups is None:
        df = pd.DataFrame({"ChA": [1.0, 2.0, 3.0], "ChB": [4.0, 5.0, 6.0]})
        groups = {
            "GrpX": {
                "ChA": {"wf_increment": 0.1, "wf_start_offset": 0.0, "wf_samples": 3},
                "ChB": {"wf_increment": 0.1, "wf_start_offset": 0.0, "wf_samples": 3},
            }
        }
        data = {"GrpX": df}
        keys = ["GrpX/ChA", "GrpX/ChB"]
    else:
        data = {}
        keys = []
        for gname, channels in groups.items():
            cols = {ch: [float(i) for i in range(3)] for ch in channels}
            data[gname] = pd.DataFrame(cols)
            for ch in channels:
                keys.append(f"{gname}/{ch}")

    return TdmsMagnetData("test.tdms", groups, keys, data)


# ---------------------------------------------------------------------------
# addTime
# ---------------------------------------------------------------------------

class TestAddTime:
    def test_delegates_to_addTdmsTime(self):
        """addTime() must add a 't' column to every non-Infos group."""
        tdms = _make_tdms()
        ret = tdms.addTime()
        assert ret == 0
        assert "t" in tdms.Data["GrpX"].columns
        assert "GrpX/t" in tdms.Keys

    def test_addTime_idempotent(self):
        """Calling addTime() twice must not raise or duplicate the 't' key."""
        tdms = _make_tdms()
        tdms.addTime()
        keys_after_first = list(tdms.Keys)
        tdms.addTime()
        assert tdms.Keys == keys_after_first


# ---------------------------------------------------------------------------
# cleanupData — keys_to_add
# ---------------------------------------------------------------------------

class TestCleanupDataKeysToAdd:
    def test_adds_derived_column(self):
        """cleanupData(keys_to_add=...) computes and stores the new column."""
        tdms = _make_tdms()
        tdms.cleanupData(keys_to_add={"GrpX/ChC": "GrpX/ChC = ChA + ChB"})
        assert "GrpX/ChC" in tdms.Keys
        assert "ChC" in tdms.Data["GrpX"].columns
        expected = tdms.Data["GrpX"]["ChA"] + tdms.Data["GrpX"]["ChB"]
        pd.testing.assert_series_equal(
            tdms.Data["GrpX"]["ChC"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_skips_existing_key(self):
        """cleanupData must not call addData if the key already exists."""
        tdms = _make_tdms()
        # Pre-populate ChA as an existing key (already in Keys)
        assert "GrpX/ChA" in tdms.Keys
        original_values = tdms.Data["GrpX"]["ChA"].copy()
        # Try to overwrite via cleanupData — should be skipped
        tdms.cleanupData(keys_to_add={"GrpX/ChA": "GrpX/ChA = ChA * 0"})
        pd.testing.assert_series_equal(tdms.Data["GrpX"]["ChA"], original_values)

    def test_returns_zero(self):
        tdms = _make_tdms()
        assert tdms.cleanupData(keys_to_add={"GrpX/ChC": "GrpX/ChC = ChA + ChB"}) == 0


# ---------------------------------------------------------------------------
# cleanupData — keys_to_remove
# ---------------------------------------------------------------------------

class TestCleanupDataKeysToRemove:
    def test_removes_existing_column(self):
        """cleanupData(keys_to_remove=...) drops the column from Data and Keys."""
        tdms = _make_tdms()
        assert "GrpX/ChB" in tdms.Keys
        tdms.cleanupData(keys_to_remove=["GrpX/ChB"])
        assert "GrpX/ChB" not in tdms.Keys
        assert "ChB" not in tdms.Data["GrpX"].columns

    def test_silently_skips_missing_key(self):
        """Removing a non-existent key must not raise."""
        tdms = _make_tdms()
        tdms.cleanupData(keys_to_remove=["GrpX/NoSuchChannel"])  # must not raise

    def test_warns_on_key_without_separator(self, caplog):
        """Keys without '/' must emit a warning and be skipped."""
        tdms = _make_tdms()
        with caplog.at_level(logging.WARNING, logger="python_magnetrun.magnetdata_tdms"):
            tdms.cleanupData(keys_to_remove=["NoSlashKey"])
        assert any("no '/' separator" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# cleanupData — keys_to_rename
# ---------------------------------------------------------------------------

class TestCleanupDataKeysToRename:
    def test_warns_and_does_not_raise(self, caplog):
        """keys_to_rename must emit a warning but not raise."""
        tdms = _make_tdms()
        with caplog.at_level(logging.WARNING, logger="python_magnetrun.magnetdata_tdms"):
            result = tdms.cleanupData(keys_to_rename={"ChA": "ChARenamed"})
        assert result == 0
        assert any("keys_to_rename" in rec.message for rec in caplog.records)
        # Column must NOT have been renamed
        assert "ChA" in tdms.Data["GrpX"].columns
        assert "ChARenamed" not in tdms.Data["GrpX"].columns
