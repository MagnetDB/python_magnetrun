"""Tests for plotting/utils.py — format_axis_label, format_legend_label,
resolve_legend_labels, and _extract_suffix."""

from __future__ import annotations

import pytest

pint = pytest.importorskip("pint")

from python_magnetrun.magnetdata_base import FieldMeta  # noqa: E402
from python_magnetrun.plotting.utils import (  # noqa: E402
    _extract_suffix,
    format_axis_label,
    format_legend_label,
    resolve_legend_labels,
)


@pytest.fixture(scope="module")
def ureg():
    return pint.UnitRegistry()


# ---------------------------------------------------------------------------
# format_axis_label
# ---------------------------------------------------------------------------


class TestFormatAxisLabel:
    def test_with_unit(self, ureg):
        assert format_axis_label("B", ureg.tesla) == "B [T]"

    def test_with_ampere(self, ureg):
        assert format_axis_label("I", ureg.ampere) == "I [A]"

    def test_with_kilampere(self, ureg):
        assert format_axis_label("I", ureg.kiloampere) == "I [kA]"

    def test_without_unit(self):
        assert format_axis_label("Timestamp", None) == "Timestamp"

    def test_time_seconds(self, ureg):
        assert format_axis_label("t", ureg.second) == "t [s]"


# ---------------------------------------------------------------------------
# format_legend_label
# ---------------------------------------------------------------------------


class TestFormatLegendLabel:
    def test_single_file_no_unit(self):
        assert format_legend_label("Champ_magn") == "Champ_magn"

    def test_multi_file_no_unit(self):
        assert (
            format_legend_label("Champ_magn", basename="M9_260331")
            == "M9_260331: Champ_magn"
        )

    def test_mixed_unit_overlay(self, ureg):
        label = format_legend_label("Field_B", unit=ureg.tesla)
        assert label == "Field_B [T]"

    def test_mixed_unit_overlay_with_basename(self, ureg):
        label = format_legend_label("Field_B", basename="run1", unit=ureg.tesla)
        assert label == "run1: Field_B [T]"

    def test_normalized_with_unit(self, ureg):
        label = format_legend_label("I_GR1", unit=ureg.ampere, max_val=1234.5)
        assert label == "I_GR1 [A]  (max = 1.23e+03 [A])"

    def test_normalized_without_unit(self):
        label = format_legend_label("I_GR1", max_val=1234.5)
        assert label == "I_GR1  (max = 1.23e+03)"

    def test_normalized_with_basename_and_unit(self, ureg):
        label = format_legend_label(
            "I_GR1", basename="run1", unit=ureg.ampere, max_val=100.0
        )
        assert label == "run1: I_GR1 [A]  (max = 100 [A])"


# ---------------------------------------------------------------------------
# _extract_suffix
# ---------------------------------------------------------------------------


class TestExtractSuffix:
    def test_simple_suffix(self):
        assert _extract_suffix("Group/Courant_GR1") == "GR1"

    def test_nested_path(self):
        assert _extract_suffix("kHz/FEPC-AUX-LNCMI/ALIM1_J1") == "J1"

    def test_no_slash_with_underscore(self):
        assert _extract_suffix("Courant_GR1") == "GR1"

    def test_no_underscore_in_base(self):
        # No underscore in the last segment → empty string
        assert _extract_suffix("Field") == ""

    def test_only_slash_no_underscore(self):
        assert _extract_suffix("kHz/ALIM1") == ""


# ---------------------------------------------------------------------------
# resolve_legend_labels
# ---------------------------------------------------------------------------


def _meta(symbol, unit=None, label="", description=""):
    return FieldMeta(symbol=symbol, unit=unit, label=label, description=description)


class TestResolveLegendLabels:
    # --- alias priority ---

    def test_alias_wins_over_everything(self, ureg):
        fields = ["B_field"]
        metas = {"B_field": _meta("B", ureg.tesla, label="magnetic field")}
        result = resolve_legend_labels(
            fields, metas, aliases={"B_field": "my override"}
        )
        assert result["B_field"] == "my override"

    # --- explicit label from meta ---

    def test_meta_label_used_when_no_alias(self):
        fields = ["Courants/Courant_GR1"]
        metas = {"Courants/Courant_GR1": _meta("I", label="I_{GR1}")}
        result = resolve_legend_labels(fields, metas)
        assert result["Courants/Courant_GR1"] == "I_{GR1}"

    # --- unique symbol — no disambiguation needed ---

    def test_unique_symbol_returned_as_is(self, ureg):
        fields = ["B_field", "temp"]
        metas = {
            "B_field": _meta("B", ureg.tesla),
            "temp": _meta("T", ureg.degC),
        }
        result = resolve_legend_labels(fields, metas)
        assert result["B_field"] == "B"
        assert result["temp"] == "T"

    # --- symbol clash — disambiguation by suffix ---

    def test_clashing_symbols_get_suffix(self):
        fields = ["Courant_GR1", "Courant_GR2"]
        metas = {
            "Courant_GR1": _meta("I"),
            "Courant_GR2": _meta("I"),
        }
        result = resolve_legend_labels(fields, metas)
        assert result["Courant_GR1"] == "I_GR1"
        assert result["Courant_GR2"] == "I_GR2"

    def test_clashing_symbols_nested_path(self):
        fields = ["kHz/FEPC/ALIM1_J1", "kHz/FEPC/ALIM1_J2"]
        metas = {
            "kHz/FEPC/ALIM1_J1": _meta("I"),
            "kHz/FEPC/ALIM1_J2": _meta("I"),
        }
        result = resolve_legend_labels(fields, metas)
        assert result["kHz/FEPC/ALIM1_J1"] == "I_J1"
        assert result["kHz/FEPC/ALIM1_J2"] == "I_J2"

    def test_clashing_symbol_no_suffix_distinct_labels(self):
        # Clash + no suffix, but each field has a distinct meta.label → use labels
        fields = ["CurrentA", "CurrentB"]
        metas = {
            "CurrentA": _meta("I", label="I_supply"),
            "CurrentB": _meta("I", label="I_return"),
        }
        result = resolve_legend_labels(fields, metas)
        assert result["CurrentA"] == "I_supply"
        assert result["CurrentB"] == "I_return"

    def test_clashing_symbol_no_suffix_same_labels_falls_back_to_field_name(self):
        # Clash + no suffix + same label → labels cannot disambiguate → field name
        fields = ["CurrentA", "CurrentB"]
        metas = {
            "CurrentA": _meta("I", label="current"),
            "CurrentB": _meta("I", label="current"),
        }
        result = resolve_legend_labels(fields, metas)
        assert result["CurrentA"] == "CurrentA"
        assert result["CurrentB"] == "CurrentB"

    def test_clashing_symbol_no_suffix_no_labels_falls_back_to_field_name(self):
        # Clash + no suffix + no labels at all → field name
        fields = ["CurrentA", "CurrentB"]
        metas = {
            "CurrentA": _meta("I"),
            "CurrentB": _meta("I"),
        }
        result = resolve_legend_labels(fields, metas)
        assert result["CurrentA"] == "CurrentA"
        assert result["CurrentB"] == "CurrentB"

    # --- missing meta ---

    def test_missing_meta_falls_back_to_field_name(self):
        fields = ["unknown_field"]
        result = resolve_legend_labels(fields, {})
        assert result["unknown_field"] == "unknown_field"

    def test_none_meta_falls_back_to_field_name(self):
        fields = ["some_field"]
        result = resolve_legend_labels(fields, {"some_field": None})
        assert result["some_field"] == "some_field"

    # --- all three priorities in one call ---

    def test_mixed_priorities(self, ureg):
        fields = ["B", "I_GR1", "I_GR2"]
        metas = {
            "B": _meta("B", ureg.tesla),
            "I_GR1": _meta("I"),
            "I_GR2": _meta("I"),
        }
        result = resolve_legend_labels(fields, metas, aliases={"B": "field alias"})
        assert result["B"] == "field alias"  # alias wins
        assert result["I_GR1"] == "I_GR1"  # clash → suffix
        assert result["I_GR2"] == "I_GR2"  # clash → suffix

    # --- empty input ---

    def test_empty_fields(self):
        assert resolve_legend_labels([], {}) == {}
