"""Tests for unit resolution helpers and ylabel rules in timeseries plotting."""

from __future__ import annotations

import warnings

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

pint = pytest.importorskip("pint")

from python_magnetrun.plotting.timeseries import (  # noqa: E402
    _check_dimension_consistency,
    _resolve_units,
    plot_overlay,
    plot_subplots,
)


@pytest.fixture()
def ureg():
    return pint.UnitRegistry()


@pytest.fixture()
def df_no_attrs():
    t = np.linspace(0, 1, 50)
    return pd.DataFrame({"t": t, "x": t * 2.0, "y": t * 3.0})


@pytest.fixture()
def df_with_tesla(ureg):
    t = np.linspace(0, 1, 50)
    df = pd.DataFrame({"t": t, "B": t * 1.0})
    df.attrs["units"] = {"B": ("T", ureg.tesla)}
    return df


@pytest.fixture()
def df_with_millitesla(ureg):
    t = np.linspace(0, 1, 50)
    # 1000 mT = 1 T
    df = pd.DataFrame({"t": t, "B": np.ones(50) * 1000.0})
    df.attrs["units"] = {"B": ("mT", ureg.millitesla)}
    return df


@pytest.fixture()
def df_with_mixed_units(ureg):
    t = np.linspace(0, 1, 50)
    df = pd.DataFrame({"t": t, "B": t, "I": t * 100})
    df.attrs["units"] = {"B": ("T", ureg.tesla), "I": ("A", ureg.ampere)}
    return df


class TestResolveUnits:
    def test_no_display_units_values_unchanged(self, df_with_tesla):
        result = _resolve_units(df_with_tesla, ["B"], None)
        np.testing.assert_allclose(result["B"][0], df_with_tesla["B"].to_numpy())

    def test_no_display_units_symbol_from_stored(self, df_with_tesla):
        result = _resolve_units(df_with_tesla, ["B"], None)
        assert result["B"][1] == "T"

    def test_with_conversion_millitesla_to_tesla_magnitude(self, df_with_millitesla):
        result = _resolve_units(df_with_millitesla, ["B"], {"B": "tesla"})
        # 1000 mT = 1 T
        np.testing.assert_allclose(
            result["B"][0], df_with_millitesla["B"].to_numpy() * 1e-3, rtol=1e-4
        )

    def test_with_conversion_millitesla_to_tesla_symbol(self, df_with_millitesla):
        result = _resolve_units(df_with_millitesla, ["B"], {"B": "tesla"})
        assert result["B"][1] == "T"

    def test_missing_attrs_values_unchanged(self, df_no_attrs):
        result = _resolve_units(df_no_attrs, ["x"], None)
        np.testing.assert_allclose(result["x"][0], df_no_attrs["x"].to_numpy())

    def test_missing_attrs_symbol_falls_back_to_field_name(self, df_no_attrs):
        result = _resolve_units(df_no_attrs, ["x"], None)
        assert result["x"][1] == "x"

    def test_display_unit_as_symbol_override_no_conversion(self, df_no_attrs):
        # No stored pint unit → display_units string used as symbol only, no conversion
        result = _resolve_units(df_no_attrs, ["x"], {"x": "m/s"})
        np.testing.assert_allclose(result["x"][0], df_no_attrs["x"].to_numpy())
        assert result["x"][1] == "m/s"


class TestCheckDimensionConsistency:
    def test_mixed_dims_emits_user_warning(self, ureg):
        stored = {"B": ("T", ureg.tesla), "I": ("A", ureg.ampere)}
        with pytest.warns(UserWarning, match="incompatible dimensions"):
            _check_dimension_consistency(["B", "I"], stored)

    def test_same_dim_no_warning(self, ureg):
        # millitesla and tesla share the same dimensionality — no warning.
        stored = {"B1": ("T", ureg.tesla), "B2": ("mT", ureg.millitesla)}
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            _check_dimension_consistency(["B1", "B2"], stored)

    def test_missing_entries_silently_ignored(self):
        stored = {}
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            _check_dimension_consistency(["A", "B"], stored)


class TestYlabelRules:
    def test_subplots_ylabel_per_ax_contains_unit(self, df_with_mixed_units):
        fig = plot_subplots(df_with_mixed_units, ["B", "I"], backend="matplotlib")
        axs = fig._magnetrun_axes
        assert "[T]" in axs[0].get_ylabel()
        assert "[A]" in axs[1].get_ylabel()

    def test_overlay_shared_ylabel_when_same_unit(self, df_with_tesla):
        fig = plot_overlay(df_with_tesla, ["B"], backend="matplotlib")
        ax = fig._magnetrun_axes[0]
        assert ax.get_ylabel() == "[T]"

    def test_overlay_no_ylabel_when_mixed_units(self, df_with_mixed_units):
        with pytest.warns(UserWarning):
            fig = plot_overlay(df_with_mixed_units, ["B", "I"], backend="matplotlib")
        ax = fig._magnetrun_axes[0]
        assert ax.get_ylabel() == ""

    def test_overlay_mixed_units_appear_in_legend_labels(self, df_with_mixed_units):
        with pytest.warns(UserWarning):
            fig = plot_overlay(df_with_mixed_units, ["B", "I"], backend="matplotlib")
        ax = fig._magnetrun_axes[0]
        labels = [ln.get_label() for ln in ax.lines]
        assert any("[T]" in lbl for lbl in labels)
        assert any("[A]" in lbl for lbl in labels)

    def test_normalize_and_display_units_compose(self, df_with_millitesla):
        # normalize + display_units are independent and compose (no ValueError).
        fig = plot_overlay(
            df_with_millitesla,
            ["B"],
            normalize=True,
            display_units={"B": "tesla"},
            backend="matplotlib",
        )
        ax = fig._magnetrun_axes[0]
        y = ax.lines[0].get_ydata()
        assert float(np.abs(y).max()) <= 1.0 + 1e-9
        assert "max=" in ax.lines[0].get_label()
