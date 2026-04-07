"""Tests for python_magnetrun.site_config — README API snippets 38-39.

Covers:
  - get_site_config(housing)                              basic default
  - get_site_config(housing, json_file=...)               load from file
  - get_site_config(housing, overrides={...})             runtime override
  - load_site_config(json_file)                           read from file
  - save_site_config(json_file, config)                   write to file
  - update_site_config(json_file, overrides)              update in place
"""

import json
from pathlib import Path

import pytest

from python_magnetrun.site_config import (
    SiteConfig,
    get_bundled_site_config_path,
    get_site_config,
    load_site_config,
    save_site_config,
    update_site_config,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def m9_json_file(tmp_path: Path) -> Path:
    """Write the built-in M9 config to a temp file and return its path."""
    cfg = get_site_config("M9")
    dest = tmp_path / "M9-site-config.json"
    save_site_config(dest, cfg)
    return dest


# ---------------------------------------------------------------------------
# get_site_config — basic defaults
# ---------------------------------------------------------------------------


class TestGetSiteConfigDefaults:
    def test_m9_returns_site_config(self):
        """get_site_config('M9') should return a SiteConfig instance."""
        cfg = get_site_config("M9")
        assert isinstance(cfg, SiteConfig)

    def test_m9_name(self):
        cfg = get_site_config("M9")
        assert cfg.name == "M9"

    def test_m9_gr1_current(self):
        """M9 GR1 reference current should be IH."""
        cfg = get_site_config("M9")
        assert cfg.reference_gr1_current == "IH"

    def test_m9_gr2_current(self):
        """M9 GR2 reference current should be IB."""
        cfg = get_site_config("M9")
        assert cfg.reference_gr2_current == "IB"

    def test_m8_gr1_gr2_swapped(self):
        """M8 swaps GR1/GR2 currents relative to M9."""
        cfg = get_site_config("M8")
        assert cfg.reference_gr1_current == "IB"
        assert cfg.reference_gr2_current == "IH"

    def test_m10_same_as_m8(self):
        m8 = get_site_config("M8")
        m10 = get_site_config("M10")
        assert m10.reference_gr1_current == m8.reference_gr1_current
        assert m10.reference_gr2_current == m8.reference_gr2_current

    def test_unknown_housing_raises(self):
        with pytest.raises(ValueError, match="Unknown housing"):
            get_site_config("INVALID")

    def test_config_is_frozen(self):
        """SiteConfig is a frozen dataclass — assignment should raise."""
        cfg = get_site_config("M9")
        from dataclasses import FrozenInstanceError

        with pytest.raises(FrozenInstanceError):
            cfg.reference_gr1_current = "IB"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# get_site_config — load from json_file
# ---------------------------------------------------------------------------


class TestGetSiteConfigFromFile:
    def test_load_from_json_file(self, m9_json_file: Path):
        """get_site_config with json_file= should load from the given file."""
        cfg = get_site_config("M9", json_file=str(m9_json_file))
        assert cfg.name == "M9"
        assert cfg.reference_gr1_current == "IH"

    def test_file_mismatch_raises(self, m9_json_file: Path):
        """Requesting M8 but providing an M9 file should raise ValueError."""
        with pytest.raises(ValueError):
            get_site_config("M8", json_file=str(m9_json_file))


# ---------------------------------------------------------------------------
# get_site_config — runtime overrides (README item 38)
# ---------------------------------------------------------------------------


class TestGetSiteConfigOverrides:
    def test_override_gr1_current(self):
        """override gr1_current should be reflected in the returned config."""
        cfg = get_site_config("M9", overrides={"gr1_current": "IB"})
        assert cfg.reference_gr1_current == "IB"
        # gr2 unchanged
        assert cfg.reference_gr2_current == "IB"

    def test_override_gr1_and_gr2_current(self):
        """Swapping both GR1/GR2 currents at runtime."""
        cfg = get_site_config("M9", overrides={"gr1_current": "IB", "gr2_current": "IH"})
        assert cfg.reference_gr1_current == "IB"
        assert cfg.reference_gr2_current == "IH"

    def test_override_does_not_mutate_default(self):
        """Overriding should not affect subsequent calls without overrides."""
        get_site_config("M9", overrides={"gr1_current": "IB"})
        cfg_default = get_site_config("M9")
        assert cfg_default.reference_gr1_current == "IH"

    def test_invalid_override_key_raises(self):
        with pytest.raises(ValueError):
            get_site_config("M9", overrides={"not_a_role": "IB"})


# ---------------------------------------------------------------------------
# load_site_config / save_site_config
# ---------------------------------------------------------------------------


class TestLoadSaveSiteConfig:
    def test_save_then_load(self, tmp_path: Path):
        """save_site_config followed by load_site_config should be idempotent."""
        original = get_site_config("M9")
        dest = tmp_path / "M9-site-config.json"
        save_site_config(dest, original)
        loaded = load_site_config(dest)

        assert loaded.name == original.name
        assert loaded.reference_gr1_current == original.reference_gr1_current
        assert loaded.reference_gr2_current == original.reference_gr2_current
        assert loaded.reference_gr1_flow == original.reference_gr1_flow

    def test_save_produces_valid_json(self, tmp_path: Path):
        """The file written by save_site_config should be parseable JSON."""
        cfg = get_site_config("M9")
        dest = tmp_path / "M9-site-config.json"
        save_site_config(dest, cfg)
        with open(dest) as fh:
            data = json.load(fh)
        assert data["name"] == "M9"

    def test_load_bundled_m9(self):
        """load_site_config on the bundled M9 file should work."""
        bundled = get_bundled_site_config_path("M9")
        if bundled.exists():
            cfg = load_site_config(bundled)
            assert cfg.name == "M9"
        else:
            pytest.skip("Bundled M9-site-config.json not found")


# ---------------------------------------------------------------------------
# update_site_config  (README item 39)
# ---------------------------------------------------------------------------


class TestUpdateSiteConfig:
    def test_update_gr1_current(self, m9_json_file: Path):
        """update_site_config should change gr1_current in the file."""
        new_cfg = update_site_config(str(m9_json_file), {"gr1_current": "IB"})
        assert new_cfg.reference_gr1_current == "IB"

    def test_update_persists_to_file(self, m9_json_file: Path):
        """After update_site_config, reloading the file should reflect the change."""
        update_site_config(str(m9_json_file), {"gr1_current": "IB"})
        reloaded = load_site_config(m9_json_file)
        assert reloaded.reference_gr1_current == "IB"

    def test_update_leaves_other_fields_unchanged(self, m9_json_file: Path):
        """Updating gr1_current should not affect gr2_current."""
        original = load_site_config(m9_json_file)
        update_site_config(str(m9_json_file), {"gr1_current": "IB"})
        updated = load_site_config(m9_json_file)
        assert updated.reference_gr2_current == original.reference_gr2_current

    def test_update_invalid_role_raises(self, m9_json_file: Path):
        with pytest.raises(ValueError):
            update_site_config(str(m9_json_file), {"not_a_role": "IB"})

    def test_update_returns_site_config(self, m9_json_file: Path):
        """update_site_config should return the updated SiteConfig."""
        result = update_site_config(str(m9_json_file), {"gr2_current": "IH"})
        assert isinstance(result, SiteConfig)
