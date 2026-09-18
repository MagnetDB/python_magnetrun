"""Unit tests for TdmsMagnetData ETL methods.

Covers:
  - addTime() delegates to addTdmsTime()
  - cleanupData(keys_to_add=...) calls addData for each entry
  - cleanupData(keys_to_remove=...) drops column from Data and Keys
  - cleanupData(keys_to_rename=...) emits logger.warning without raising
  - cleanupData(keys_to_add=...) skips formulas targeting an absent group
  - load_units_from_json() derives units for combined-probe channels
"""

import json
import logging

import polars as pl

from python_magnetrun.magnetdata_tdms import TdmsMagnetData

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tdms(groups: dict | None = None) -> TdmsMagnetData:
    """Build a minimal TdmsMagnetData with one group containing two channels."""
    if groups is None:
        df = pl.DataFrame({"ChA": [1.0, 2.0, 3.0], "ChB": [4.0, 5.0, 6.0]})
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
            data[gname] = pl.DataFrame(cols)
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
        tdms.cleanupData(keys_to_add={"GrpX/ChC": {"formula": "GrpX/ChC = ChA + ChB", "symbol": "ChC", "unit": None, "label": "ChA + ChB", "description": "Sum of ChA and ChB"}})
        assert "GrpX/ChC" in tdms.Keys
        assert "ChC" in tdms.Data["GrpX"].columns
        expected = tdms.Data["GrpX"]["ChA"] + tdms.Data["GrpX"]["ChB"]
        assert tdms.Data["GrpX"]["ChC"].to_list() == expected.to_list()

    def test_skips_existing_key(self):
        """cleanupData must not call addData if the key already exists."""
        tdms = _make_tdms()
        # Pre-populate ChA as an existing key (already in Keys)
        assert "GrpX/ChA" in tdms.Keys
        original_values = tdms.Data["GrpX"]["ChA"].clone()
        # Try to overwrite via cleanupData — should be skipped
        tdms.cleanupData(keys_to_add={"GrpX/ChA": {"formula": "GrpX/ChA = ChA * 0", "symbol": "ChA", "unit": None, "label": "ChA zeroed", "description": "Should be skipped"}})
        assert tdms.Data["GrpX"]["ChA"].to_list() == original_values.to_list()

    def test_returns_zero(self):
        tdms = _make_tdms()
        assert tdms.cleanupData(keys_to_add={"GrpX/ChC": {"formula": "GrpX/ChC = ChA + ChB", "symbol": "ChC", "unit": None, "label": "ChA + ChB", "description": "Sum of ChA and ChB"}}) == 0

    def test_skips_formula_when_target_group_missing(self, caplog):
        """A formula targeting a group this file doesn't have must be skipped,
        not raise -- e.g. a housing's pigbrother_formula_map (which targets
        Courants_Alimentations) applied to a Stats file, which lacks that
        group entirely."""
        tdms = _make_tdms()  # only has "GrpX"
        with caplog.at_level(logging.DEBUG, logger="python_magnetrun.magnetdata_tdms"):
            status = tdms.cleanupData(
                keys_to_add={
                    "NoSuchGroup/Derived": {
                        "formula": "NoSuchGroup/Derived = ChA + ChB",
                        "symbol": "X",
                        "unit": None,
                        "label": "X",
                        "description": "",
                    }
                }
            )
        assert status == 0
        assert "NoSuchGroup/Derived" not in tdms.Keys
        assert any("not present in this file" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# load_units_from_json — combined-probe fallback
# ---------------------------------------------------------------------------


class TestLoadUnitsFromJsonCombinedProbe:
    """A combined-probe channel (e.g. Interne1-2: probe 1 expected for the
    assembly but unavailable, folded into probe 2's channel) has no defs
    entry of its own -- load_units_from_json() must derive one from the
    replaced probe's entry instead."""

    def test_derives_unit_from_replaced_probe(self, tmp_path):
        tdms = _make_tdms(groups={"Stats_moy": ["moy_Interne1-2", "moy_Interne2"]})
        defs_file = tmp_path / "test-defs.json"
        defs_file.write_text(
            json.dumps(
                {
                    "Stats_moy/moy_Interne2": {
                        "symbol": "U",
                        "unit": "volt",
                        "label": "U_int2_moy",
                        "description": "Internal coil voltage, probe 2 -- mean over 1s",
                    }
                }
            )
        )
        tdms.load_units_from_json(str(defs_file))

        assert "Stats_moy/moy_Interne1-2" in tdms.units
        symbol, unit = tdms.units["Stats_moy/moy_Interne1-2"]
        assert symbol == "U"
        assert str(unit) == "volt"
        description = tdms.field_meta["Stats_moy/moy_Interne1-2"].description
        assert "probe 1" in description
        assert "unavailable" in description

    def test_no_base_entry_leaves_channel_unresolved(self, tmp_path):
        """If the replaced probe's own entry isn't defined either, the
        combined channel is simply left unresolved -- not an error."""
        tdms = _make_tdms(groups={"Stats_moy": ["moy_Interne1-2"]})
        defs_file = tmp_path / "test-defs.json"
        defs_file.write_text(json.dumps({}))
        tdms.load_units_from_json(str(defs_file))
        assert "Stats_moy/moy_Interne1-2" not in tdms.units


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
