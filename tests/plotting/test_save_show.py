"""Tests for _default_save_path, _handle_output, and --save/--show argparse semantics."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from python_magnetrun.cli_args import create_managed_plots_parser
from python_magnetrun.commands.plot import _default_save_path, _handle_output


class TestDefaultSavePath:
    def test_returns_path_under_cwd(self):
        path = _default_save_path(["/data/run.txt"], ["B"], "matplotlib")
        assert path.parent == Path.cwd()

    def test_png_suffix_for_matplotlib(self):
        path = _default_save_path(["/data/run.txt"], ["B"], "matplotlib")
        assert path.suffix == ".png"

    def test_html_suffix_for_plotly(self):
        path = _default_save_path(["/data/run.txt"], ["B"], "plotly")
        assert path.suffix == ".html"

    def test_html_suffix_for_plotly_resampler(self):
        path = _default_save_path(["/data/run.txt"], ["B"], "plotly-resampler")
        assert path.suffix == ".html"

    def test_stem_from_last_input_file(self):
        path = _default_save_path(["/data/M9_260331.txt"], ["B"], "matplotlib")
        assert "M9_260331" in path.stem

    def test_empty_input_files_uses_output_stem(self):
        path = _default_save_path([], ["B"], "matplotlib")
        assert "output" in path.stem

    def test_up_to_two_fields_in_name(self):
        path = _default_save_path(["/data/run.txt"], ["B", "I", "T"], "matplotlib")
        assert "B" in path.stem
        assert "I" in path.stem


class TestHandleOutput:
    def _make_backend(self):
        b = MagicMock()
        b.save = MagicMock()
        b.show = MagicMock()
        return b

    def _args(self, save=None, show=False):
        ns = argparse.Namespace()
        ns.save = save
        ns.show = show
        return ns

    def test_neither_flag_defaults_to_show(self):
        b = self._make_backend()
        fig = MagicMock()
        _handle_output(fig, self._args(save=None, show=False), b, ["f.txt"], ["B"], "matplotlib")
        b.show.assert_called_once_with(fig)
        b.save.assert_not_called()

    def test_show_true_calls_show(self):
        b = self._make_backend()
        fig = MagicMock()
        _handle_output(fig, self._args(save=None, show=True), b, ["f.txt"], ["B"], "matplotlib")
        b.show.assert_called_once_with(fig)
        b.save.assert_not_called()

    def test_save_empty_string_uses_default_name(self):
        b = self._make_backend()
        fig = MagicMock()
        _handle_output(fig, self._args(save="", show=False), b, ["f.txt"], ["B"], "matplotlib")
        b.save.assert_called_once()
        saved_path = b.save.call_args[0][1]
        assert saved_path.parent == Path.cwd()
        b.show.assert_not_called()

    def test_save_explicit_filename_uses_that_path(self, tmp_path):
        b = self._make_backend()
        fig = MagicMock()
        explicit = str(tmp_path / "out.png")
        _handle_output(fig, self._args(save=explicit, show=False), b, ["f.txt"], ["B"], "matplotlib")
        b.save.assert_called_once()
        saved_path = b.save.call_args[0][1]
        assert saved_path == Path(explicit)
        b.show.assert_not_called()

    def test_save_does_not_call_show(self):
        b = self._make_backend()
        fig = MagicMock()
        _handle_output(fig, self._args(save="out.png", show=False), b, ["f.txt"], ["B"], "matplotlib")
        b.show.assert_not_called()

    def test_dpi_forwarded_to_save(self):
        b = self._make_backend()
        fig = MagicMock()
        _handle_output(fig, self._args(save=""), b, ["f.txt"], ["B"], "matplotlib", dpi=150)
        _, kwargs = b.save.call_args
        assert kwargs.get("dpi") == 150


class TestSaveShowMutuallyExclusive:
    def _parser(self):
        return argparse.ArgumentParser(parents=[create_managed_plots_parser()])

    def test_save_and_show_are_mutually_exclusive(self):
        parser = self._parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--save", "out.png", "--show"])

    def test_save_alone_is_accepted(self):
        args = self._parser().parse_args(["--save", "out.png"])
        assert args.save == "out.png"
        assert args.show is False

    def test_save_no_filename_is_accepted(self):
        args = self._parser().parse_args(["--save"])
        assert args.save == ""

    def test_show_alone_is_accepted(self):
        args = self._parser().parse_args(["--show"])
        assert args.show is True
        assert args.save is None

    def test_neither_flag_gives_defaults(self):
        args = self._parser().parse_args([])
        assert args.save is None
        assert args.show is False
