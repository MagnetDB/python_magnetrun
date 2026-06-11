"""Tests for python_magnetrun.data_dirs — data_dirs.json config file feature.

Covers:
  - load_data_dirs_json()          valid file, missing file, invalid JSON,
                                   non-dict root, empty-string values
  - _resolve()                     env var > JSON > default priority chain
  - Module-level constants         reloaded with a custom JSON file via monkeypatch
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

import python_magnetrun.data_dirs as data_dirs_mod
from python_magnetrun.data_dirs import _resolve, load_data_dirs_json

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_json(path: Path, data: object) -> Path:
    path.write_text(json.dumps(data))
    return path


# ---------------------------------------------------------------------------
# load_data_dirs_json — file content
# ---------------------------------------------------------------------------


class TestLoadDataDirsJson:
    def test_returns_dict_from_valid_file(self, tmp_path: Path):
        p = _write_json(tmp_path / "data_dirs.json", {"pupitre": "/data/pupitre"})
        result = load_data_dirs_json(p)
        assert result == {"pupitre": "/data/pupitre"}

    def test_returns_empty_dict_when_file_missing(self, tmp_path: Path):
        result = load_data_dirs_json(tmp_path / "nonexistent.json")
        assert result == {}

    def test_returns_empty_dict_on_invalid_json(self, tmp_path: Path):
        p = tmp_path / "data_dirs.json"
        p.write_text("{ not valid json }")
        result = load_data_dirs_json(p)
        assert result == {}

    def test_returns_empty_dict_when_root_is_list(self, tmp_path: Path):
        p = _write_json(tmp_path / "data_dirs.json", ["/data/pupitre"])
        result = load_data_dirs_json(p)
        assert result == {}

    def test_skips_empty_string_values(self, tmp_path: Path):
        p = _write_json(tmp_path / "data_dirs.json", {"pupitre": "/data/pupitre", "hybrid": ""})
        result = load_data_dirs_json(p)
        assert "hybrid" not in result
        assert result["pupitre"] == "/data/pupitre"

    def test_skips_non_string_values(self, tmp_path: Path):
        p = _write_json(tmp_path / "data_dirs.json", {"pupitre": "/data/pupitre", "bprofile": 42})
        result = load_data_dirs_json(p)
        assert "bprofile" not in result

    def test_all_recognised_keys(self, tmp_path: Path):
        data = {
            "pupitre": "/d/pupitre",
            "pigbrother": "/d/pigbrother",
            "hybrid": "/d/hybrid",
            "bprofile": "/d/bprofile",
            "feelpp": "/d/feelpp",
            "magnettools": "/d/magnettools",
            "pigbrother_runlog": "/d/pb_runlog",
            "pupitre_runlog": "/d/pu_runlog",
            "hybrid_runlog": "/d/hy_runlog",
        }
        p = _write_json(tmp_path / "data_dirs.json", data)
        result = load_data_dirs_json(p)
        assert result == data

    def test_unknown_keys_are_preserved(self, tmp_path: Path):
        """Unknown keys pass through — callers decide what to do with them."""
        p = _write_json(tmp_path / "data_dirs.json", {"unknown_key": "/some/path"})
        result = load_data_dirs_json(p)
        assert result.get("unknown_key") == "/some/path"

    def test_uses_default_path_when_none_given(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        """With no argument, load_data_dirs_json reads from _DATA_DIRS_JSON_PATH."""
        json_dir = tmp_path / ".config" / "python_magnetrun"
        json_dir.mkdir(parents=True)
        _write_json(json_dir / "data_dirs.json", {"pupitre": "/from/default"})
        # Redirect Path.home() to tmp_path so the default path points to our file.
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr(data_dirs_mod, "_DATA_DIRS_JSON_PATH", json_dir / "data_dirs.json")
        result = load_data_dirs_json()
        assert result == {"pupitre": "/from/default"}


# ---------------------------------------------------------------------------
# _resolve — priority chain
# ---------------------------------------------------------------------------


class TestResolve:
    def test_returns_default_when_nothing_set(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(data_dirs_mod, "_JSON_DIRS", {})
        monkeypatch.delenv("MAGNETRUN_PUPITRE_DATA_DIR", raising=False)
        result = _resolve("MAGNETRUN_PUPITRE_DATA_DIR", json_key="pupitre", default="/hard/default")
        assert result == "/hard/default"

    def test_json_overrides_default(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(data_dirs_mod, "_JSON_DIRS", {"pupitre": "/from/json"})
        monkeypatch.delenv("MAGNETRUN_PUPITRE_DATA_DIR", raising=False)
        result = _resolve("MAGNETRUN_PUPITRE_DATA_DIR", json_key="pupitre", default="/hard/default")
        assert result == "/from/json"

    def test_env_var_overrides_json(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(data_dirs_mod, "_JSON_DIRS", {"pupitre": "/from/json"})
        monkeypatch.setenv("MAGNETRUN_PUPITRE_DATA_DIR", "/from/env")
        result = _resolve("MAGNETRUN_PUPITRE_DATA_DIR", json_key="pupitre", default="/hard/default")
        assert result == "/from/env"

    def test_first_env_var_wins_over_second(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(data_dirs_mod, "_JSON_DIRS", {})
        monkeypatch.setenv("MAGNETRUN_PUPITRE_DATA_DIR", "/first")
        monkeypatch.setenv("PUPITRE_DATADIR", "/second")
        result = _resolve("MAGNETRUN_PUPITRE_DATA_DIR", "PUPITRE_DATADIR", json_key="pupitre", default="/default")
        assert result == "/first"

    def test_second_env_var_used_when_first_absent(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(data_dirs_mod, "_JSON_DIRS", {})
        monkeypatch.delenv("MAGNETRUN_PUPITRE_DATA_DIR", raising=False)
        monkeypatch.setenv("PUPITRE_DATADIR", "/second")
        result = _resolve("MAGNETRUN_PUPITRE_DATA_DIR", "PUPITRE_DATADIR", json_key="pupitre", default="/default")
        assert result == "/second"

    def test_no_json_key_falls_straight_to_default(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(data_dirs_mod, "_JSON_DIRS", {"pupitre": "/from/json"})
        monkeypatch.delenv("MAGNETRUN_PUPITRE_DATA_DIR", raising=False)
        result = _resolve("MAGNETRUN_PUPITRE_DATA_DIR", default="/default")
        assert result == "/default"


# ---------------------------------------------------------------------------
# Module-level constants — reload integration test
# ---------------------------------------------------------------------------


def _make_fake_home(tmp_path: Path, data: dict) -> Path:
    """Create a fake HOME dir with data_dirs.json and return the HOME path."""
    json_dir = tmp_path / ".config" / "python_magnetrun"
    json_dir.mkdir(parents=True)
    _write_json(json_dir / "data_dirs.json", data)
    return tmp_path


class TestModuleLevelConstants:
    """Integration tests: reload the module with a fake HOME to pick up JSON."""

    def test_pupitre_data_dir_from_json(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        """PUPITRE_DATA_DIR picks up the JSON value when the module is reloaded."""
        fake_home = _make_fake_home(tmp_path, {"pupitre": "/json/pupitre"})

        monkeypatch.delenv("MAGNETRUN_PUPITRE_DATA_DIR", raising=False)
        monkeypatch.delenv("PUPITRE_DATADIR", raising=False)
        monkeypatch.setenv("HOME", str(fake_home))

        reloaded = importlib.reload(data_dirs_mod)
        try:
            assert reloaded.PUPITRE_DATA_DIR == "/json/pupitre"
        finally:
            importlib.reload(data_dirs_mod)  # restore original state

    def test_pigbrother_data_dir_from_json(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        fake_home = _make_fake_home(tmp_path, {"pigbrother": "/json/pb"})

        monkeypatch.delenv("MAGNETRUN_PIGBROTHER_DATA_DIR", raising=False)
        monkeypatch.delenv("PIGBROTHER_DATADIR", raising=False)
        monkeypatch.setenv("HOME", str(fake_home))

        reloaded = importlib.reload(data_dirs_mod)
        try:
            assert reloaded.PIGBROTHER_DATA_DIR == "/json/pb"
        finally:
            importlib.reload(data_dirs_mod)

    def test_env_var_beats_json_in_constants(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        """Env var still takes priority over JSON after module reload."""
        fake_home = _make_fake_home(tmp_path, {"pupitre": "/json/pupitre"})

        monkeypatch.setenv("MAGNETRUN_PUPITRE_DATA_DIR", "/env/pupitre")
        monkeypatch.setenv("HOME", str(fake_home))

        reloaded = importlib.reload(data_dirs_mod)
        try:
            assert reloaded.PUPITRE_DATA_DIR == "/env/pupitre"
        finally:
            importlib.reload(data_dirs_mod)

    def test_json_missing_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        """When JSON has no entry for a key, the hard-coded default is used."""
        fake_home = _make_fake_home(tmp_path, {})

        monkeypatch.delenv("MAGNETRUN_PUPITRE_DATA_DIR", raising=False)
        monkeypatch.delenv("PUPITRE_DATADIR", raising=False)
        monkeypatch.setenv("HOME", str(fake_home))

        reloaded = importlib.reload(data_dirs_mod)
        try:
            assert reloaded.PUPITRE_DATA_DIR == "/mnt/LNCMIG-Data/records/srv-data-install"
        finally:
            importlib.reload(data_dirs_mod)
