"""CLI tests for ``magnetrun-config`` — the unified config entrypoint.

Covers the argument-parsing and dispatch layer for all three domains:
  - plot    {init, show, validate}
  - housing {show, create, update}
  - field   {list, add, update, delete, alias-add, alias-show, crossref}

Each test drives the CLI through ``sys.argv`` patching so the full
argparse → dispatch path is exercised without spawning a subprocess.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

from python_magnetrun.config_cli import main as config_main
from python_magnetrun.field_defs import _dispatch_field, _populate_field_parser
from python_magnetrun.housing_config import (
    _dispatch_housing,
    _populate_housing_parser,
    get_housing_config,
    save_housing_config,
)
from python_magnetrun.plotting.cli import _dispatch_plot, _populate_plot_parser

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(argv: list[str]) -> int:
    """Run ``magnetrun-config`` with the given argv and return the exit code."""
    old = sys.argv
    sys.argv = ["magnetrun-config"] + argv
    try:
        with pytest.raises(SystemExit) as exc:
            config_main()
        return exc.value.code  # type: ignore[return-value]
    finally:
        sys.argv = old


def _make_plot_args(argv: list[str]) -> argparse.Namespace:
    import argparse

    p = argparse.ArgumentParser()
    _populate_plot_parser(p)
    return p.parse_args(argv)


def _make_housing_args(argv: list[str]) -> argparse.Namespace:
    import argparse

    p = argparse.ArgumentParser()
    _populate_housing_parser(p)
    return p.parse_args(argv)


def _make_field_args(argv: list[str]) -> argparse.Namespace:
    import argparse

    p = argparse.ArgumentParser()
    _populate_field_parser(p)
    return p.parse_args(argv)


def _minimal_defs(path: Path) -> Path:
    data = {
        "Field": {"symbol": "B", "unit": "tesla", "description": "Magnetic field"},
        "Idcct1": {
            "symbol": "I",
            "unit": "ampere",
            "description": "DCCT current 1",
            "aliases": {"pigbrother": "Courants_Alimentations/Courant_A1"},
        },
    }
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path


# ---------------------------------------------------------------------------
# Top-level routing
# ---------------------------------------------------------------------------


class TestTopLevelRouting:
    def test_no_domain_exits_nonzero(self):
        code = _run([])
        assert code != 0

    def test_unknown_domain_exits_nonzero(self):
        code = _run(["unknown"])
        assert code != 0

    def test_help_exits_zero(self):
        code = _run(["--help"])
        assert code == 0

    def test_plot_help_exits_zero(self):
        code = _run(["plot", "--help"])
        assert code == 0

    def test_housing_help_exits_zero(self):
        code = _run(["housing", "--help"])
        assert code == 0

    def test_field_help_exits_zero(self):
        code = _run(["field", "--help"])
        assert code == 0


# ---------------------------------------------------------------------------
# plot domain
# ---------------------------------------------------------------------------


class TestPlotInit:
    def test_init_creates_file(self, tmp_path: Path):
        out = tmp_path / "plot_config.json"
        args = _make_plot_args(["init", "--output", str(out)])
        assert _dispatch_plot(args) == 0
        assert out.exists()

    def test_init_does_not_overwrite_without_force(self, tmp_path: Path):
        out = tmp_path / "plot_config.json"
        out.write_text("{}")
        args = _make_plot_args(["init", "--output", str(out)])
        assert _dispatch_plot(args) != 0

    def test_init_overwrites_with_force(self, tmp_path: Path):
        out = tmp_path / "plot_config.json"
        out.write_text("{}")
        args = _make_plot_args(["init", "--output", str(out), "--force"])
        assert _dispatch_plot(args) == 0


class TestPlotShow:
    def test_show_defaults_exits_zero(self):
        args = _make_plot_args(["show"])
        assert _dispatch_plot(args) == 0

    def test_show_valid_file_exits_zero(self, tmp_path: Path):
        out = tmp_path / "plot_config.json"
        _dispatch_plot(_make_plot_args(["init", "--output", str(out)]))
        args = _make_plot_args(["show", str(out)])
        assert _dispatch_plot(args) == 0

    def test_show_invalid_file_exits_nonzero(self, tmp_path: Path):
        bad = tmp_path / "bad.json"
        bad.write_text("not-json{{{")
        args = _make_plot_args(["show", str(bad)])
        assert _dispatch_plot(args) != 0


class TestPlotValidate:
    def test_validate_bundled_config_ok(self, tmp_path: Path):
        out = tmp_path / "plot_config.json"
        _dispatch_plot(_make_plot_args(["init", "--output", str(out)]))
        args = _make_plot_args(["validate", str(out)])
        assert _dispatch_plot(args) == 0

    def test_validate_unknown_field_fails(self, tmp_path: Path):
        cfg = tmp_path / "bad_config.json"
        cfg.write_text(json.dumps({"style": {"not_a_field": 42}}))
        args = _make_plot_args(["validate", str(cfg)])
        assert _dispatch_plot(args) != 0


# ---------------------------------------------------------------------------
# housing domain
# ---------------------------------------------------------------------------


class TestHousingShow:
    def test_show_m9(self, tmp_path: Path):
        dest = tmp_path / "M9-housing-config.json"
        save_housing_config(dest, get_housing_config("M9"))
        args = _make_housing_args([str(dest), "show"])
        assert _dispatch_housing(args) == 0


class TestHousingCreate:
    def test_create_new_housing(self, tmp_path: Path):
        dest = tmp_path / "M11-housing-config.json"
        args = _make_housing_args([str(dest), "create", "M11"])
        assert _dispatch_housing(args) == 0
        assert dest.exists()
        data = json.loads(dest.read_text())
        assert data["name"] == "M11"

    def test_create_from_builtin(self, tmp_path: Path):
        dest = tmp_path / "M11-housing-config.json"
        args = _make_housing_args([str(dest), "create", "M11", "--from-builtin", "M9"])
        assert _dispatch_housing(args) == 0

    def test_create_from_unknown_builtin_fails(self, tmp_path: Path):
        dest = tmp_path / "X-housing-config.json"
        args = _make_housing_args([str(dest), "create", "X", "--from-builtin", "NOPE"])
        assert _dispatch_housing(args) != 0


class TestHousingUpdate:
    def test_update_gr1_current(self, tmp_path: Path):
        dest = tmp_path / "M9-housing-config.json"
        save_housing_config(dest, get_housing_config("M9"))
        args = _make_housing_args([str(dest), "update", "--gr1-current", "IB"])
        assert _dispatch_housing(args) == 0
        data = json.loads(dest.read_text())
        assert data["reference_gr1_current"] == "IB"

    def test_update_no_fields_fails(self, tmp_path: Path):
        dest = tmp_path / "M9-housing-config.json"
        save_housing_config(dest, get_housing_config("M9"))
        args = _make_housing_args([str(dest), "update"])
        assert _dispatch_housing(args) != 0


# ---------------------------------------------------------------------------
# field domain
# ---------------------------------------------------------------------------


class TestFieldList:
    def test_list_bundled(self, tmp_path: Path):
        defs = _minimal_defs(tmp_path / "defs.json")
        args = _make_field_args([str(defs), "list"])
        assert _dispatch_field(args) == 0


class TestFieldAdd:
    def test_add_new_field(self, tmp_path: Path):
        defs = _minimal_defs(tmp_path / "defs.json")
        args = _make_field_args([str(defs), "add", "NewSensor", "I", "ampere"])
        assert _dispatch_field(args) == 0
        data = json.loads(defs.read_text())
        assert "NewSensor" in data


class TestFieldUpdate:
    def test_update_symbol(self, tmp_path: Path):
        defs = _minimal_defs(tmp_path / "defs.json")
        args = _make_field_args([str(defs), "update", "Field", "--symbol", "Bz"])
        assert _dispatch_field(args) == 0
        data = json.loads(defs.read_text())
        assert data["Field"]["symbol"] == "Bz"


class TestFieldDelete:
    def test_delete_field(self, tmp_path: Path):
        defs = _minimal_defs(tmp_path / "defs.json")
        args = _make_field_args([str(defs), "delete", "Field"])
        assert _dispatch_field(args) == 0
        data = json.loads(defs.read_text())
        assert "Field" not in data

    def test_remove_alias(self, tmp_path: Path):
        defs = _minimal_defs(tmp_path / "defs.json")
        args = _make_field_args([str(defs), "remove", "Field"])
        assert _dispatch_field(args) == 0


class TestFieldAlias:
    def test_alias_add(self, tmp_path: Path):
        defs = _minimal_defs(tmp_path / "defs.json")
        args = _make_field_args(
            [str(defs), "alias-add", "Field", "hybrid", "FEPC-LNCMI/B_HALL"]
        )
        assert _dispatch_field(args) == 0
        data = json.loads(defs.read_text())
        assert data["Field"]["aliases"]["hybrid"] == "FEPC-LNCMI/B_HALL"

    def test_alias_show(self, tmp_path: Path, capsys: pytest.CaptureFixture):
        defs = _minimal_defs(tmp_path / "defs.json")
        args = _make_field_args([str(defs), "alias-show", "Idcct1"])
        assert _dispatch_field(args) == 0
        captured = capsys.readouterr()
        assert "pigbrother" in captured.out

    def test_alias_remove(self, tmp_path: Path):
        defs = _minimal_defs(tmp_path / "defs.json")
        args = _make_field_args(
            [str(defs), "alias-remove", "Idcct1", "pigbrother"]
        )
        assert _dispatch_field(args) == 0
        data = json.loads(defs.read_text())
        assert "pigbrother" not in data["Idcct1"].get("aliases", {})


class TestFieldCrossref:
    def test_crossref_two_files(self, tmp_path: Path):
        f1 = _minimal_defs(tmp_path / "a-defs.json")
        f2 = _minimal_defs(tmp_path / "b-defs.json")
        args = _make_field_args(
            [
                str(f1),
                "crossref",
                "--format", f"a={f1}",
                "--format", f"b={f2}",
            ]
        )
        assert _dispatch_field(args) == 0

    def test_crossref_bad_format_spec_fails(self, tmp_path: Path):
        defs = _minimal_defs(tmp_path / "defs.json")
        args = _make_field_args(
            [str(defs), "crossref", "--format", "no-equals-sign"]
        )
        assert _dispatch_field(args) != 0


# ---------------------------------------------------------------------------
# end-to-end via config_main (smoke tests)
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_plot_show_via_main(self):
        assert _run(["plot", "show"]) == 0

    def test_field_list_via_main(self, tmp_path: Path):
        defs = _minimal_defs(tmp_path / "defs.json")
        assert _run(["field", str(defs), "list"]) == 0

    def test_housing_show_via_main(self, tmp_path: Path):
        dest = tmp_path / "M9-housing-config.json"
        save_housing_config(dest, get_housing_config("M9"))
        assert _run(["housing", str(dest), "show"]) == 0
