"""Tests for python_magnetrun.housing_config — README API snippets 38-39.

Covers:
  - get_housing_config(housing)                              basic default
  - get_housing_config(housing, json_file=...)               load from file
  - get_housing_config(housing, overrides={...})             runtime override
  - load_housing_config(json_file)                           read from file
  - save_housing_config(json_file, config)                   write to file
  - update_housing_config(json_file, overrides)              update in place
"""

import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from python_magnetrun.housing_config import (
    HOUSING_CONFIGS,
    HousingConfig,
    get_bundled_housing_config_path,
    get_housing_config,
    load_housing_config,
    save_housing_config,
    update_housing_config,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def m9_json_file(tmp_path: Path) -> Path:
    """Write the built-in M9 config to a temp file and return its path."""
    cfg = get_housing_config("M9")
    dest = tmp_path / "M9-housing-config.json"
    save_housing_config(dest, cfg)
    return dest


# ---------------------------------------------------------------------------
# get_housing_config — basic defaults
# ---------------------------------------------------------------------------


class TestGetHousingConfigDefaults:
    def test_m9_returns_housing_config(self):
        """get_housing_config('M9') should return a HousingConfig instance."""
        cfg = get_housing_config("M9")
        assert isinstance(cfg, HousingConfig)

    def test_m9_name(self):
        cfg = get_housing_config("M9")
        assert cfg.name == "M9"

    def test_m9_gr1_current(self):
        """M9 GR1 reference current should be IH."""
        cfg = get_housing_config("M9")
        assert cfg.reference_gr1_current == "IH"

    def test_m9_gr2_current(self):
        """M9 GR2 reference current should be IB."""
        cfg = get_housing_config("M9")
        assert cfg.reference_gr2_current == "IB"

    def test_m8_gr1_gr2_swapped(self):
        """M8 swaps GR1/GR2 currents relative to M9."""
        cfg = get_housing_config("M8")
        assert cfg.reference_gr1_current == "IB"
        assert cfg.reference_gr2_current == "IH"

    def test_m10_same_as_m8(self):
        m8 = get_housing_config("M8")
        m10 = get_housing_config("M10")
        assert m10.reference_gr1_current == m8.reference_gr1_current
        assert m10.reference_gr2_current == m8.reference_gr2_current

    def test_unknown_housing_raises(self):
        with pytest.raises(ValueError, match="Unknown housing"):
            get_housing_config("INVALID")

    def test_config_is_frozen(self):
        """HousingConfig is a frozen dataclass — assignment should raise."""
        cfg = get_housing_config("M9")
        from dataclasses import FrozenInstanceError

        with pytest.raises(FrozenInstanceError):
            cfg.reference_gr1_current = "IB"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# get_housing_config — load from json_file
# ---------------------------------------------------------------------------


class TestGetHousingConfigFromFile:
    def test_load_from_json_file(self, m9_json_file: Path):
        """get_housing_config with json_file= should load from the given file."""
        cfg = get_housing_config("M9", json_file=str(m9_json_file))
        assert cfg.name == "M9"
        assert cfg.reference_gr1_current == "IH"

    def test_file_mismatch_raises(self, m9_json_file: Path):
        """Requesting M8 but providing an M9 file should raise ValueError."""
        with pytest.raises(ValueError):
            get_housing_config("M8", json_file=str(m9_json_file))


# ---------------------------------------------------------------------------
# get_housing_config — runtime overrides (README item 38)
# ---------------------------------------------------------------------------


class TestGetHousingConfigOverrides:
    def test_override_gr1_current(self):
        """override gr1_current should be reflected in the returned config."""
        cfg = get_housing_config("M9", overrides={"gr1_current": "IB"})
        assert cfg.reference_gr1_current == "IB"
        # gr2 unchanged
        assert cfg.reference_gr2_current == "IB"

    def test_override_gr1_and_gr2_current(self):
        """Swapping both GR1/GR2 currents at runtime."""
        cfg = get_housing_config("M9", overrides={"gr1_current": "IB", "gr2_current": "IH"})
        assert cfg.reference_gr1_current == "IB"
        assert cfg.reference_gr2_current == "IH"

    def test_override_does_not_mutate_default(self):
        """Overriding should not affect subsequent calls without overrides."""
        get_housing_config("M9", overrides={"gr1_current": "IB"})
        cfg_default = get_housing_config("M9")
        assert cfg_default.reference_gr1_current == "IH"

    def test_invalid_override_key_raises(self):
        with pytest.raises(ValueError):
            get_housing_config("M9", overrides={"not_a_role": "IB"})


# ---------------------------------------------------------------------------
# load_housing_config / save_housing_config
# ---------------------------------------------------------------------------


class TestLoadSaveHousingConfig:
    def test_save_then_load(self, tmp_path: Path):
        """save_housing_config followed by load_housing_config should be idempotent."""
        original = get_housing_config("M9")
        dest = tmp_path / "M9-housing-config.json"
        save_housing_config(dest, original)
        loaded = load_housing_config(dest)

        assert loaded.name == original.name
        assert loaded.reference_gr1_current == original.reference_gr1_current
        assert loaded.reference_gr2_current == original.reference_gr2_current
        assert loaded.reference_gr1_flow == original.reference_gr1_flow

    def test_save_produces_valid_json(self, tmp_path: Path):
        """The file written by save_housing_config should be parseable JSON."""
        cfg = get_housing_config("M9")
        dest = tmp_path / "M9-housing-config.json"
        save_housing_config(dest, cfg)
        with open(dest) as fh:
            data = json.load(fh)
        assert data["name"] == "M9"

    def test_load_bundled_m9(self):
        """load_housing_config on the bundled M9 file should work."""
        bundled = get_bundled_housing_config_path("M9")
        if bundled.exists():
            cfg = load_housing_config(bundled)
            assert cfg.name == "M9"
        else:
            pytest.skip("Bundled M9-housing-config.json not found")


# ---------------------------------------------------------------------------
# update_housing_config  (README item 39)
# ---------------------------------------------------------------------------


class TestUpdateHousingConfig:
    def test_update_gr1_current(self, m9_json_file: Path):
        """update_housing_config should change gr1_current in the file."""
        new_cfg = update_housing_config(str(m9_json_file), {"gr1_current": "IB"})
        assert new_cfg.reference_gr1_current == "IB"

    def test_update_persists_to_file(self, m9_json_file: Path):
        """After update_housing_config, reloading the file should reflect the change."""
        update_housing_config(str(m9_json_file), {"gr1_current": "IB"})
        reloaded = load_housing_config(m9_json_file)
        assert reloaded.reference_gr1_current == "IB"

    def test_update_leaves_other_fields_unchanged(self, m9_json_file: Path):
        """Updating gr1_current should not affect gr2_current."""
        original = load_housing_config(m9_json_file)
        update_housing_config(str(m9_json_file), {"gr1_current": "IB"})
        updated = load_housing_config(m9_json_file)
        assert updated.reference_gr2_current == original.reference_gr2_current

    def test_update_invalid_role_raises(self, m9_json_file: Path):
        with pytest.raises(ValueError):
            update_housing_config(str(m9_json_file), {"not_a_role": "IB"})

    def test_update_returns_housing_config(self, m9_json_file: Path):
        """update_housing_config should return the updated HousingConfig."""
        result = update_housing_config(str(m9_json_file), {"gr2_current": "IH"})
        assert isinstance(result, HousingConfig)


# ---------------------------------------------------------------------------
# Time-versioned wiring (HousingConfig.at / get_housing_config(at=...))
# ---------------------------------------------------------------------------

PARIS = ZoneInfo("Europe/Paris")
M8_UCOIL1_13 = tuple(f"Ucoil{i}" for i in range(1, 14))


def _versioned(**version) -> HousingConfig:
    """Return a minimal config holding one version (defaults filled in)."""
    v = {
        "name": "X-1",
        "valid_from": "2026-03-05T17:21:09+01:00",
        "overrides": {"voltage_channels_gr1": ["Ucoil14"]},
    }
    v.update(version)
    return HousingConfig(name="X", voltage_channels_gr1=("Ucoil15",), versions=(v,))


class TestHousingConfigVersions:
    def test_m8_default_is_base(self):
        """Without a date, M8 keeps today's wiring and no version name."""
        cfg = HOUSING_CONFIGS["M8"]
        assert cfg.name == "M8"
        assert cfg.version == ""
        assert cfg.voltage_channels_gr1 == ("Ucoil15", "Ucoil16")

    def test_m8_before_change_is_base_version(self):
        cfg = HOUSING_CONFIGS["M8"].at(datetime(2026, 3, 5, 15, 56, 42, tzinfo=PARIS))
        assert cfg.version == "M8-230913"
        assert cfg.voltage_channels_gr1 == ("Ucoil15", "Ucoil16")

    def test_m8_before_first_version_is_unnamed_base(self):
        """Data older than the first commissioned assembly: base, no version name."""
        cfg = HOUSING_CONFIGS["M8"].at(datetime(2015, 6, 1, tzinfo=PARIS))
        assert cfg == HOUSING_CONFIGS["M8"]

    @pytest.mark.parametrize(
        "housing, name, valid_from, overrides",
        [
            ("M5", "M5-140101", "2014-01-01T08:00:00+01:00", {}),
            ("M7", "M7-140101", "2014-01-01T08:00:00+01:00", {}),
            # M8's first assembly ran on a 12-slot pupitre (M8Bitters on Ucoil11/12)
            ("M8", "M8-211225", "2021-12-25T08:00:00+01:00", {
                "voltage_channels_gr1": ["Ucoil11", "Ucoil12"],
                "voltage_channels_gr2": [f"Ucoil{i}" for i in range(1, 11)],
            }),
            ("M9", "M9-181109", "2018-11-09T08:00:00+01:00", {}),
            ("M10", "M10-180220", "2018-02-20T08:00:00+01:00", {}),
        ],
    )
    def test_dated_base_version(self, housing, name, valid_from, overrides):
        """Every bundled housing starts with a dated version from its first assembly."""
        first = HOUSING_CONFIGS[housing].versions[0]
        assert (first["name"], first["valid_from"], first["overrides"]) == (name, valid_from, overrides)

    def test_m8_12_slot_pupitre_until_2023_09_13(self):
        """M8_A211225_00 (Hybrid + M8Bitters): 12-slot pupitre, Bitter on Ucoil11/12."""
        cfg = HOUSING_CONFIGS["M8"].at(datetime(2022, 11, 28, 13, 40, 49, tzinfo=PARIS))
        assert cfg.version == "M8-211225"
        assert cfg.voltage_channels_gr1 == ("Ucoil11", "Ucoil12")
        assert cfg.voltage_channels_gr2 == tuple(f"Ucoil{i}" for i in range(1, 11))

    def test_m8_16_slot_pupitre_from_2023_09_13(self):
        """From the first 16-slot file (M23012001 header): back to the base wiring."""
        cfg = HOUSING_CONFIGS["M8"].at(datetime(2023, 9, 14, 10, 27, 45, tzinfo=PARIS))
        assert cfg.version == "M8-230913"
        assert cfg.voltage_channels_gr1 == ("Ucoil15", "Ucoil16")
        assert cfg.voltage_channels_gr2 == HOUSING_CONFIGS["M8"].voltage_channels_gr2

    def test_m8_from_change_uses_version(self):
        cfg = HOUSING_CONFIGS["M8"].at(datetime(2026, 3, 5, 17, 21, 9, tzinfo=PARIS))
        assert cfg.name == "M8"
        assert cfg.version == "M8-260305"
        assert cfg.voltage_channels_gr1 == ("Ucoil14",)
        assert cfg.voltage_channels_gr2 == M8_UCOIL1_13
        # fields not overridden are kept
        assert cfg.reference_gr1_current == "IB"

    def test_m8_version_from_utc_time(self):
        """The same instant given in UTC resolves to the same version."""
        cfg = HOUSING_CONFIGS["M8"].at(datetime(2026, 3, 5, 16, 21, 9, tzinfo=timezone.utc))
        assert cfg.version == "M8-260305"

    def test_m8_version_formulas(self):
        """UB sums Ucoil14 only, UH no longer contains it."""
        cfg = HOUSING_CONFIGS["M8"].at(datetime(2026, 10, 30, 9, 10, 3, tzinfo=PARIS))
        formulas = cfg.get_pupitre_voltage_formulas()
        assert formulas["UB"]["formula"] == "UB = Ucoil14"
        assert "Ucoil14" not in formulas["UH"]["formula"]

    @pytest.mark.parametrize(
        "housing, when, version, ih, ib",
        [
            # M7: insert fed by A1 + A3 in 2014 (PLAN_pupitre_checks_file.md, data issues)
            ("M7", datetime(2014, 1, 2, tzinfo=PARIS), "M7-140101", "IH = Idcct1 + Idcct2", None),
            ("M7", datetime(2014, 4, 1, 22, 2, tzinfo=PARIS), "M7-140317", "IH = Idcct1 + Idcct3", None),
            ("M7", datetime(2016, 7, 27, tzinfo=PARIS), "M7-160726", "IH = Idcct1 + Idcct2", None),
            # M10: insert fed by A1 + A2, Bitters by A3 + A4, 2021-03-25 -> 2021-07-10
            ("M10", datetime(2021, 3, 9, 9, 17, tzinfo=PARIS), "M10-180220",
             "IH = Idcct3 + Idcct4", "IB = Idcct1 + Idcct2"),
            ("M10", datetime(2021, 5, 1, tzinfo=PARIS), "M10-210325",
             "IH = Idcct1 + Idcct2", "IB = Idcct3 + Idcct4"),
            ("M10", datetime(2021, 10, 1, tzinfo=PARIS), "M10-210929",
             "IH = Idcct3 + Idcct4", "IB = Idcct1 + Idcct2"),
        ],
    )
    def test_swapped_supply_versions(self, housing, when, version, ih, ib):
        cfg = HOUSING_CONFIGS[housing].at(when)
        assert cfg.version == version
        assert cfg.pupitre_formula_map["IH"]["formula"] == ih
        if ib:
            assert cfg.pupitre_formula_map["IB"]["formula"] == ib
        # only the current formulas change: wiring and the rest of the map are kept
        base = HOUSING_CONFIGS[housing]
        assert (cfg.voltage_channels_gr1, cfg.voltage_channels_gr2) == (
            base.voltage_channels_gr1, base.voltage_channels_gr2)
        assert set(cfg.pupitre_formula_map) == set(base.pupitre_formula_map)

    def test_m9_base_version(self):
        """M9 keeps its base wiring; only the version name is set."""
        cfg = HOUSING_CONFIGS["M9"].at(datetime(2026, 10, 29, 18, 3, 45, tzinfo=PARIS))
        assert cfg.version == "M9-181109"
        assert dataclasses.replace(cfg, version="") == HOUSING_CONFIGS["M9"]

    def test_get_housing_config_at(self):
        when = datetime(2026, 3, 6, 9, 11, 2, tzinfo=PARIS)
        assert get_housing_config("M8", at=when) == HOUSING_CONFIGS["M8"].at(when)

    def test_get_housing_config_at_then_overrides(self):
        """Runtime overrides apply on top of the resolved version."""
        when = datetime(2026, 3, 6, 9, 11, 2, tzinfo=PARIS)
        cfg = get_housing_config("M8", at=when, overrides={"gr1_current": "IH"})
        assert cfg.version == "M8-260305"
        assert cfg.reference_gr1_current == "IH"

    def test_latest_version_wins(self):
        v1 = {"name": "X-1", "valid_from": "2026-01-01T00:00:00+00:00",
              "overrides": {"voltage_channels_gr1": ["Ucoil14"]}}
        v2 = {"name": "X-2", "valid_from": "2026-06-01T00:00:00+00:00",
              "overrides": {"voltage_channels_gr1": ["Ucoil15", "Ucoil16"]}}
        cfg = HousingConfig(name="X", versions=(v2, v1))
        assert cfg.at(datetime(2026, 3, 1, tzinfo=timezone.utc)).version == "X-1"
        assert cfg.at(datetime(2026, 7, 1, tzinfo=timezone.utc)).version == "X-2"

    def test_naive_at_raises(self):
        with pytest.raises(ValueError, match="timezone"):
            HOUSING_CONFIGS["M8"].at(datetime(2026, 3, 6))

    def test_at_on_version_raises(self):
        """Versions apply to the base config only."""
        cfg = _versioned().at(datetime(2026, 4, 1, tzinfo=timezone.utc))
        with pytest.raises(ValueError, match="base"):
            cfg.at(datetime(2026, 4, 1, tzinfo=timezone.utc))

    def test_naive_valid_from_raises(self):
        with pytest.raises(ValueError, match="offset"):
            _versioned(valid_from="2026-03-05T17:21:09")

    def test_unknown_override_field_raises(self):
        with pytest.raises(ValueError, match="not_a_field"):
            _versioned(overrides={"not_a_field": "x"})

    def test_override_of_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            _versioned(overrides={"name": "Y"})

    def test_unknown_version_key_raises(self):
        with pytest.raises(ValueError, match="valid_form"):
            _versioned(valid_form="2026-03-05T17:21:09+01:00")

    def test_duplicate_valid_from_raises(self):
        v = {"name": "X-1", "valid_from": "2026-03-05T17:21:09+01:00", "overrides": {}}
        w = dict(v, name="X-2")
        with pytest.raises(ValueError, match="valid_from"):
            HousingConfig(name="X", versions=(v, w))

    def test_round_trip_keeps_versions(self, tmp_path: Path):
        dest = tmp_path / "M8-housing-config.json"
        save_housing_config(dest, HOUSING_CONFIGS["M8"])
        loaded = load_housing_config(dest)
        assert loaded == HOUSING_CONFIGS["M8"]
        assert [v["name"] for v in loaded.versions] == ["M8-211225", "M8-230913", "M8-260305"]
