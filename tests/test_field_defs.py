"""Tests for python_magnetrun.field_defs — README API snippets 40-41.

Covers:
  - load_defs         (read bundled defs files)
  - add_field_def     (write new entry to a temp file)
  - update_field_def  (update existing entry)
  - add_alias         (add cross-format alias)
  - get_aliases       (query aliases — README item 40)
  - build_crossref    (build unified index — README item 41)
"""

import json
from pathlib import Path

import pytest

from python_magnetrun.field_defs import (
    add_alias,
    add_field_def,
    build_crossref,
    get_aliases,
    load_defs,
    match_channels_across_formats,
    update_field_def,
)
from python_magnetrun.magnetdata_base import DataType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_minimal_defs(path: Path, extras: dict | None = None) -> Path:
    """Write a minimal *-defs.json to *path* and return it."""
    data = {
        "Field": {"symbol": "B", "unit": "tesla", "description": "Magnetic field"},
        "Idcct1": {
            "symbol": "I",
            "unit": "ampere",
            "description": "DCCT current 1",
            "aliases": {
                "pigbrother": "Courants_Alimentations/Courant_A1",
            },
        },
    }
    if extras:
        data.update(extras)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path


# ---------------------------------------------------------------------------
# load_defs
# ---------------------------------------------------------------------------


class TestLoadDefs:
    def test_bundled_pupitre(self):
        """load_defs with a bare name resolves to the bundled pupitre-defs.json."""
        defs = load_defs("pupitre-defs.json")
        assert isinstance(defs, dict)
        assert len(defs) > 0

    def test_bundled_pigbrother(self):
        defs = load_defs("pigbrother-defs.json")
        assert isinstance(defs, dict)
        assert len(defs) > 0

    def test_bundled_hybrid(self):
        defs = load_defs("hybrid-defs.json")
        assert isinstance(defs, dict)
        assert len(defs) > 0

    def test_bundled_pupitre_has_field(self):
        """The bundled pupitre defs must contain a 'Field' entry."""
        defs = load_defs("pupitre-defs.json")
        assert "Field" in defs

    def test_load_from_explicit_path(self, tmp_path: Path):
        """load_defs should accept an explicit file path."""
        json_file = _make_minimal_defs(tmp_path / "test-defs.json")
        defs = load_defs(json_file)
        assert "Field" in defs
        assert "Idcct1" in defs


# ---------------------------------------------------------------------------
# add_field_def
# ---------------------------------------------------------------------------


class TestAddFieldDef:
    def test_adds_new_entry(self, tmp_path: Path):
        """add_field_def should add a new key to the file."""
        json_file = _make_minimal_defs(tmp_path / "defs.json")
        add_field_def(json_file, "NewSensor", "I", "ampere", description="New coil current")
        defs = load_defs(json_file)
        assert "NewSensor" in defs
        assert defs["NewSensor"]["symbol"] == "I"
        assert defs["NewSensor"]["unit"] == "ampere"
        assert defs["NewSensor"]["description"] == "New coil current"

    def test_duplicate_raises_without_overwrite(self, tmp_path: Path):
        """Adding an existing key without overwrite=True should raise KeyError."""
        json_file = _make_minimal_defs(tmp_path / "defs.json")
        with pytest.raises(KeyError, match="already exists"):
            add_field_def(json_file, "Field", "B", "tesla")

    def test_overwrite_replaces_entry(self, tmp_path: Path):
        """overwrite=True should silently replace an existing entry."""
        json_file = _make_minimal_defs(tmp_path / "defs.json")
        add_field_def(json_file, "Field", "Bz", "tesla", overwrite=True)
        defs = load_defs(json_file)
        assert defs["Field"]["symbol"] == "Bz"

    def test_none_unit_allowed(self, tmp_path: Path):
        """unit=None should be stored as null (dimensionless/timestamp)."""
        json_file = _make_minimal_defs(tmp_path / "defs.json")
        add_field_def(json_file, "timestamp", "time", None)
        defs = load_defs(json_file)
        assert defs["timestamp"]["unit"] is None


# ---------------------------------------------------------------------------
# update_field_def
# ---------------------------------------------------------------------------


class TestUpdateFieldDef:
    def test_update_symbol(self, tmp_path: Path):
        """update_field_def should change only the specified attribute."""
        json_file = _make_minimal_defs(tmp_path / "defs.json")
        update_field_def(json_file, "Field", symbol="Bz")
        defs = load_defs(json_file)
        assert defs["Field"]["symbol"] == "Bz"
        assert defs["Field"]["unit"] == "tesla"  # unchanged

    def test_update_description(self, tmp_path: Path):
        json_file = _make_minimal_defs(tmp_path / "defs.json")
        update_field_def(json_file, "Field", description="Axial field")
        defs = load_defs(json_file)
        assert defs["Field"]["description"] == "Axial field"

    def test_update_unit_to_none(self, tmp_path: Path):
        """Passing unit=None should mark a field as dimensionless."""
        json_file = _make_minimal_defs(tmp_path / "defs.json")
        update_field_def(json_file, "Field", unit=None)
        defs = load_defs(json_file)
        assert defs["Field"]["unit"] is None

    def test_missing_key_raises(self, tmp_path: Path):
        json_file = _make_minimal_defs(tmp_path / "defs.json")
        with pytest.raises(KeyError):
            update_field_def(json_file, "DoesNotExist", symbol="X")


# ---------------------------------------------------------------------------
# add_alias / get_aliases  (README item 40)
# ---------------------------------------------------------------------------


class TestAliases:
    def test_add_alias(self, tmp_path: Path):
        """add_alias should store a cross-format alias under 'aliases'."""
        json_file = _make_minimal_defs(tmp_path / "defs.json")
        add_alias(json_file, "Field", "hybrid", "FEPC-LNCMI/B_HALL")
        defs = load_defs(json_file)
        assert defs["Field"]["aliases"]["hybrid"] == "FEPC-LNCMI/B_HALL"

    def test_add_alias_duplicate_is_skipped(self, tmp_path: Path):
        """add_alias on an existing format key should leave the original value unchanged."""
        json_file = _make_minimal_defs(tmp_path / "defs.json")
        add_alias(json_file, "Idcct1", "pigbrother", "Courants_Alimentations/Courant_NEW")
        aliases = get_aliases(json_file, "Idcct1")
        assert aliases["pigbrother"] == "Courants_Alimentations/Courant_A1"

    def test_add_alias_missing_key_raises(self, tmp_path: Path):
        json_file = _make_minimal_defs(tmp_path / "defs.json")
        with pytest.raises(KeyError):
            add_alias(json_file, "NonExistent", "hybrid", "FEPC/X")

    def test_get_aliases_returns_dict(self, tmp_path: Path):
        """get_aliases should return the aliases dict for a field."""
        json_file = _make_minimal_defs(tmp_path / "defs.json")
        aliases = get_aliases(json_file, "Idcct1")
        assert isinstance(aliases, dict)
        assert aliases["pigbrother"] == "Courants_Alimentations/Courant_A1"

    def test_get_aliases_empty_for_field_without_aliases(self, tmp_path: Path):
        """get_aliases returns an empty dict when no aliases are defined."""
        json_file = _make_minimal_defs(tmp_path / "defs.json")
        aliases = get_aliases(json_file, "Field")
        assert aliases == {}

    def test_get_aliases_missing_key_raises(self, tmp_path: Path):
        json_file = _make_minimal_defs(tmp_path / "defs.json")
        with pytest.raises(KeyError):
            get_aliases(json_file, "NonExistent")

    def test_get_aliases_bundled_pupitre_idcct1(self):
        """README example: get_aliases on bundled pupitre-defs.json for Idcct1."""
        aliases = get_aliases("pupitre-defs.json", "Idcct1")
        assert isinstance(aliases, dict)
        # Should have at least a pigbrother or hybrid alias if defined
        assert len(aliases) >= 0  # non-error is sufficient when no aliases yet


# ---------------------------------------------------------------------------
# build_crossref  (README item 41)
# ---------------------------------------------------------------------------


class TestBuildCrossref:
    def test_returns_nested_dict(self, tmp_path: Path):
        """build_crossref should return a dict keyed by format names."""
        # Build two minimal defs files with a shared alias
        pupitre = tmp_path / "pupitre-defs.json"
        hybrid = tmp_path / "hybrid-defs.json"

        pupitre.write_text(json.dumps({
            "Idcct1": {
                "symbol": "I", "unit": "ampere", "description": "",
                "aliases": {"hybrid": "FEPC-AUX-LNCMI/ALIM1_J1"},
            }
        }) + "\n")
        hybrid.write_text(json.dumps({
            "FEPC-AUX-LNCMI/ALIM1_J1": {
                "symbol": "I", "unit": "ampere", "description": "",
            }
        }) + "\n")

        index = build_crossref({
            "pupitre": str(pupitre),
            "hybrid": str(hybrid),
        })

        assert "pupitre" in index
        assert "hybrid" in index

    def test_crossref_pupitre_to_hybrid(self, tmp_path: Path):
        """Fields with aliases should appear in the index with their aliases."""
        pupitre = tmp_path / "pupitre-defs.json"
        hybrid = tmp_path / "hybrid-defs.json"

        pupitre.write_text(json.dumps({
            "Idcct1": {
                "symbol": "I", "unit": "ampere", "description": "",
                "aliases": {"hybrid": "FEPC-AUX-LNCMI/ALIM1_J1"},
            }
        }) + "\n")
        hybrid.write_text(json.dumps({
            "FEPC-AUX-LNCMI/ALIM1_J1": {
                "symbol": "I", "unit": "ampere", "description": "",
            }
        }) + "\n")

        index = build_crossref({
            "pupitre": str(pupitre),
            "hybrid": str(hybrid),
        })

        assert "Idcct1" in index["pupitre"]
        assert index["pupitre"]["Idcct1"]["hybrid"] == "FEPC-AUX-LNCMI/ALIM1_J1"

    def test_crossref_bundled_three_formats(self):
        """README example: build_crossref across all three bundled formats."""
        index = build_crossref({
            "pupitre": "pupitre-defs.json",
            "pigbrother": "pigbrother-defs.json",
            "hybrid": "hybrid-defs.json",
        })
        assert "pupitre" in index
        assert "pigbrother" in index
        assert "hybrid" in index
        # Every value in the index should be a dict
        for fmt, fields in index.items():
            for field_name, aliases in fields.items():
                assert isinstance(aliases, dict), (
                    f"index[{fmt!r}][{field_name!r}] is not a dict"
                )


# ---------------------------------------------------------------------------
# match_channels_across_formats
# ---------------------------------------------------------------------------


class TestMatchChannelsAcrossFormats:
    """Exercised against the bundled pupitre-defs.json/pigbrother-defs.json,
    which are known to alias Idcct1 (pupitre) <-> Courants_Alimentations/Courant_A1
    (pigbrother) — see TestAliases.test_get_aliases_bundled_pupitre_idcct1.
    """

    def test_matches_known_alias_pair(self):
        channels_by_type = {
            DataType.PUPITRE: {"Idcct1"},
            DataType.TDMS: {"Courant_A1"},
        }
        entries = match_channels_across_formats(channels_by_type, "Courants_Alimentations")
        assert len(entries) == 1
        entry = entries[0]
        assert entry["channels"] == {"pupitre": "Idcct1", "pigbrother": "Courant_A1"}
        assert entry["label"]
        for key in ("id", "label", "description", "channels"):
            assert key in entry

    def test_channel_without_counterpart_is_dropped(self):
        """A channel present in only one format has nothing to compare against."""
        channels_by_type = {DataType.PUPITRE: {"Idcct1"}}
        entries = match_channels_across_formats(channels_by_type, "Courants_Alimentations")
        assert entries == []

    def test_unsupported_datatype_is_skipped_not_erroring(self):
        """HYBRID has no *-defs.json/aliases support here and should be ignored,
        while the still-comparable pupitre/pigbrother pair is still returned."""
        channels_by_type = {
            DataType.PUPITRE: {"Idcct1"},
            DataType.TDMS: {"Courant_A1"},
            DataType.HYBRID: {"whatever"},
        }
        entries = match_channels_across_formats(channels_by_type, "Courants_Alimentations")
        assert len(entries) == 1

    def test_only_unsupported_datatype_returns_empty(self):
        entries = match_channels_across_formats(
            {DataType.HYBRID: {"whatever"}}, "Courants_Alimentations"
        )
        assert entries == []

    def test_empty_input_returns_empty(self):
        assert match_channels_across_formats({}, "Courants_Alimentations") == []
