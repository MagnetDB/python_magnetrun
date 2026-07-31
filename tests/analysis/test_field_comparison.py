"""
Tests for python_magnetrun.analysis.field_comparison.

These tests use synthetic DataFrames and monkeypatched loaders — no real
Overview/Archive/pupitre files are available in this repository.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from python_magnetrun.analysis import field_comparison  # noqa: E402
from python_magnetrun.analysis.field_comparison import (  # noqa: E402
    AliasedField,
    compare_all_fields,
    compare_field,
    compute_reference_lag,
    discover_pupitre_pigbrother_fields,
    print_comparison_summary,
)
from python_magnetrun.analysis.processing import OverviewRecord  # noqa: E402
from python_magnetrun.analysis.synchronization import LagResult  # noqa: E402

ORIGIN = pd.Timestamp(datetime(2024, 1, 1, 0, 0, 0))


def _bump_df(t: np.ndarray, columns_to_lag: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Build a DataFrame with one Gaussian-bump column per entry in *columns_to_lag*.

    ``columns_to_lag[name] = (amplitude, lag)`` — the bump for *name* is
    centered at ``t=10 + lag`` with the given amplitude, so that a "pupitre"
    column built with ``lag=true_lag`` is a delayed copy of a "pigbrother"
    column built with ``lag=0``.
    """
    data = {"timestamp": ORIGIN + pd.to_timedelta(t, unit="s")}
    for name, (amplitude, lag) in columns_to_lag.items():
        data[name] = amplitude * np.exp(-((t - 10.0 - lag) ** 2) / (2 * 1.0**2))
    return pd.DataFrame(data)


# =============================================================================
# discover_pupitre_pigbrother_fields
# =============================================================================
class TestDiscoverPupitrePigbrotherFields:
    def test_discovers_known_fields(self):
        fields = discover_pupitre_pigbrother_fields()
        by_key = {f.pupitre_key: f for f in fields}

        assert by_key["Idcct1"] == AliasedField(
            "Idcct1", "Courants_Alimentations", "Courant_A1",
            pupitre_unit="ampere", pigbrother_unit="ampere", pupitre_symbol="I",
        )
        assert by_key["Idcct3"] == AliasedField(
            "Idcct3", "Courants_Alimentations", "Courant_A3",
            pupitre_unit="ampere", pigbrother_unit="ampere", pupitre_symbol="I",
        )
        assert by_key["Field"] == AliasedField(
            "Field", "Courants_Alimentations", "Champ_magn",
            pupitre_unit="tesla", pigbrother_unit="millitesla", pupitre_symbol="Bz",
        )
        assert by_key["Ucoil1"] == AliasedField(
            "Ucoil1", "Tensions_Aimant", "Interne1",
            pupitre_unit="volt", pigbrother_unit="volt", pupitre_symbol="U",
        )
        assert by_key["UH"] == AliasedField(
            "UH", "Tensions_Aimant", "ALL_internes",
            pupitre_unit="volt", pigbrother_unit="volt", pupitre_symbol="U_H",
        )
        assert by_key["UB"] == AliasedField(
            "UB", "Tensions_Aimant", "ALL_externes",
            pupitre_unit="volt", pigbrother_unit="volt", pupitre_symbol="U",
        )

    def test_expected_count(self):
        # Field, Idcct1-4, Ucoil1-7, Ucoil15, Ucoil16, UH, UB
        fields = discover_pupitre_pigbrother_fields()
        assert len(fields) == 16

    def test_no_group_prefix_in_pupitre_key(self):
        fields = discover_pupitre_pigbrother_fields()
        for f in fields:
            assert "/" not in f.pupitre_key


# =============================================================================
# _find_combined_tap_channel
# =============================================================================
class TestFindCombinedTapChannel:
    def test_finds_merged_channel_for_high_tap(self):
        result = field_comparison._find_combined_tap_channel("Interne2", {"Interne1-2", "Interne3"})
        assert result == "Interne1-2"

    def test_no_match_for_unrelated_tap(self):
        result = field_comparison._find_combined_tap_channel("Interne3", {"Interne1-2"})
        assert result is None

    def test_no_match_for_non_tap_channel(self):
        result = field_comparison._find_combined_tap_channel("ALL_internes", {"Interne1-2"})
        assert result is None


# =============================================================================
# compute_reference_lag
# =============================================================================
class TestComputeReferenceLag:
    REFERENCE_FIELDS = {
        "Idcct1": AliasedField("Idcct1", "Courants_Alimentations", "Courant_A1"),
        "Idcct3": AliasedField("Idcct3", "Courants_Alimentations", "Courant_A3"),
    }

    def _record(self) -> OverviewRecord:
        return OverviewRecord(filename="test", housing="M9")

    def test_prefers_idcct1_when_both_present(self):
        t = np.arange(0, 20, 1.0)
        pupitre_df = _bump_df(t, {"Idcct1": (1.0, 2.0), "Idcct3": (1.0, 2.0)})
        pigbrother_df = _bump_df(t, {"Courant_A1": (1.0, 0.0), "Courant_A3": (1.0, 0.0)})

        result, field = compute_reference_lag(
            self._record(), "overview", pupitre_df, pigbrother_df, self.REFERENCE_FIELDS
        )
        assert abs(result.lag.total_seconds() - 2.0) < 0.5
        assert field.pupitre_key == "Idcct1"

    def test_falls_back_to_idcct3_when_idcct1_missing(self):
        t = np.arange(0, 20, 1.0)
        pupitre_df = _bump_df(t, {"Idcct3": (1.0, 3.0)})  # no Idcct1 column
        pigbrother_df = _bump_df(t, {"Courant_A3": (1.0, 0.0)})  # no Courant_A1 column

        result, field = compute_reference_lag(
            self._record(), "overview", pupitre_df, pigbrother_df, self.REFERENCE_FIELDS
        )
        assert abs(result.lag.total_seconds() - 3.0) < 0.5
        assert field.pupitre_key == "Idcct3"

    def test_falls_back_when_pigbrother_channel_missing(self):
        t = np.arange(0, 20, 1.0)
        pupitre_df = _bump_df(t, {"Idcct1": (1.0, 1.0), "Idcct3": (1.0, 1.0)})
        pigbrother_df = _bump_df(t, {"Courant_A3": (1.0, 0.0)})  # Courant_A1 absent

        result, field = compute_reference_lag(
            self._record(), "overview", pupitre_df, pigbrother_df, self.REFERENCE_FIELDS
        )
        assert abs(result.lag.total_seconds() - 1.0) < 0.5
        assert field.pupitre_key == "Idcct3"

    def test_raises_when_neither_pair_available(self):
        t = np.arange(0, 20, 1.0)
        pupitre_df = _bump_df(t, {"SomethingElse": (1.0, 0.0)})
        pigbrother_df = _bump_df(t, {"SomethingElse": (1.0, 0.0)})

        with pytest.raises(ValueError, match="no reference channel pair available"):
            compute_reference_lag(
                self._record(), "overview", pupitre_df, pigbrother_df, self.REFERENCE_FIELDS
            )


# =============================================================================
# compare_field
# =============================================================================
class TestCompareField:
    FIELD = AliasedField("Ucoil1", "Tensions_Aimant", "Interne1")

    def test_available_with_high_correlation(self):
        t = np.arange(0, 20, 1.0)
        pigbrother_df = _bump_df(t, {"Interne1": (2.0, 0.0)})
        pupitre_corrected = _bump_df(t, {"Ucoil1": (2.0, 0.0)})  # already "corrected" == aligned

        result = compare_field(self.FIELD, "overview", pigbrother_df, pupitre_corrected)

        assert result.available
        assert result.reason is None
        assert result.n_points > 0
        assert result.metrics["distances"].correlation > 0.95

    def test_unavailable_missing_pigbrother_channel(self):
        t = np.arange(0, 20, 1.0)
        pigbrother_df = _bump_df(t, {"SomethingElse": (1.0, 0.0)})
        pupitre_corrected = _bump_df(t, {"Ucoil1": (1.0, 0.0)})

        result = compare_field(self.FIELD, "overview", pigbrother_df, pupitre_corrected)

        assert not result.available
        assert "Interne1" in result.reason

    def test_unavailable_missing_pupitre_channel(self):
        t = np.arange(0, 20, 1.0)
        pigbrother_df = _bump_df(t, {"Interne1": (1.0, 0.0)})
        pupitre_corrected = _bump_df(t, {"SomethingElse": (1.0, 0.0)})

        result = compare_field(self.FIELD, "overview", pigbrother_df, pupitre_corrected)

        assert not result.available
        assert "Ucoil1" in result.reason

    def test_unavailable_no_time_overlap(self):
        t1 = np.arange(0, 5, 1.0)
        t2 = np.arange(100, 105, 1.0)
        pigbrother_df = _bump_df(t1, {"Interne1": (1.0, 0.0)})
        pupitre_corrected = _bump_df(t2, {"Ucoil1": (1.0, 0.0)})

        result = compare_field(self.FIELD, "overview", pigbrother_df, pupitre_corrected)

        assert not result.available
        assert "overlap" in result.reason


# =============================================================================
# compare_field -- unit conversion
# =============================================================================
class TestCompareFieldUnitConversion:
    def test_converts_pigbrother_unit_to_pupitre_unit(self, caplog):
        t = np.arange(0, 20, 1.0)
        field = AliasedField(
            "Field", "Courants_Alimentations", "Champ_magn",
            pupitre_unit="tesla", pigbrother_unit="millitesla",
        )
        # Same physical field, same shape/timing -- 1.0 tesla == 1000.0 millitesla.
        pigbrother_df = _bump_df(t, {"Champ_magn": (1000.0, 0.0)})
        pupitre_df = _bump_df(t, {"Field": (1.0, 0.0)})

        with caplog.at_level("WARNING", logger="python_magnetrun.analysis.field_comparison"):
            result = compare_field(field, "overview", pigbrother_df, pupitre_df)

        assert result.available
        # Without the conversion, MAPE would be ~99900% (1000x scale mismatch).
        assert result.metrics["distances"].correlation > 0.999
        assert result.metrics["distances"].mape < 1.0
        assert any("unit mismatch" in r.message for r in caplog.records)

    def test_no_conversion_when_units_match(self):
        t = np.arange(0, 20, 1.0)
        field = AliasedField(
            "Ucoil1", "Tensions_Aimant", "Interne1",
            pupitre_unit="volt", pigbrother_unit="volt",
        )
        pigbrother_df = _bump_df(t, {"Interne1": (2.0, 0.0)})
        pupitre_df = _bump_df(t, {"Ucoil1": (2.0, 0.0)})

        result = compare_field(field, "overview", pigbrother_df, pupitre_df)

        assert result.available
        assert result.metrics["distances"].correlation > 0.999

    def test_raises_on_undefined_unit(self):
        t = np.arange(0, 20, 1.0)
        field = AliasedField(
            "Field", "Courants_Alimentations", "Champ_magn",
            pupitre_unit="tesla", pigbrother_unit="bogus_unit_xyz",
        )
        pigbrother_df = _bump_df(t, {"Champ_magn": (1000.0, 0.0)})
        pupitre_df = _bump_df(t, {"Field": (1.0, 0.0)})

        with pytest.raises(ValueError, match="cannot convert"):
            compare_field(field, "overview", pigbrother_df, pupitre_df)

    def test_raises_on_incompatible_dimensions(self):
        t = np.arange(0, 20, 1.0)
        field = AliasedField(
            "Ucoil1", "Tensions_Aimant", "Interne1",
            pupitre_unit="volt", pigbrother_unit="ampere",
        )
        pigbrother_df = _bump_df(t, {"Interne1": (2.0, 0.0)})
        pupitre_df = _bump_df(t, {"Ucoil1": (2.0, 0.0)})

        with pytest.raises(ValueError, match="cannot convert"):
            compare_field(field, "overview", pigbrother_df, pupitre_df)


# =============================================================================
# compare_field -- plotting (lag_method/lag value in filename & title, ylabel)
# =============================================================================
class TestCompareFieldPlotting:
    def _capture_plot_comparison(self, monkeypatch):
        captured = {}

        def fake_plot_comparison(df1, df2, x_col, y_col1, y_col2, **kwargs):
            captured["df1"] = df1
            captured["y_col1"] = y_col1
            captured["kwargs"] = kwargs
            return None

        monkeypatch.setattr(field_comparison, "plot_comparison", fake_plot_comparison)
        return captured

    def test_filename_and_title_include_lag_method_and_value(self, monkeypatch, tmp_path):
        captured = self._capture_plot_comparison(monkeypatch)
        t = np.arange(0, 20, 1.0)
        field = AliasedField("Ucoil1", "Tensions_Aimant", "Interne1")
        pigbrother_df = _bump_df(t, {"Interne1": (2.0, 0.0)})
        pupitre_df = _bump_df(t, {"Ucoil1": (2.0, 0.0)})
        lag = LagResult(lag=timedelta(seconds=0.523), method="interpolated")

        compare_field(
            field, "overview", pigbrother_df, pupitre_df,
            plot=True, output_dir=str(tmp_path), reference_lag=lag,
        )

        kwargs = captured["kwargs"]
        assert kwargs["output_path"] == f"{tmp_path}/Ucoil1_overview_interpolated_comparison.png"
        assert "lag_method=interpolated" in kwargs["title"]
        assert "lag=0.523s" in kwargs["title"]

    def test_filename_and_title_say_none_without_reference_lag(self, monkeypatch, tmp_path):
        captured = self._capture_plot_comparison(monkeypatch)
        t = np.arange(0, 20, 1.0)
        field = AliasedField("Ucoil1", "Tensions_Aimant", "Interne1")
        pigbrother_df = _bump_df(t, {"Interne1": (2.0, 0.0)})
        pupitre_df = _bump_df(t, {"Ucoil1": (2.0, 0.0)})

        compare_field(
            field, "overview", pigbrother_df, pupitre_df,
            plot=True, output_dir=str(tmp_path),
        )

        kwargs = captured["kwargs"]
        assert kwargs["output_path"] == f"{tmp_path}/Ucoil1_overview_none_comparison.png"
        assert "lag_method=none" in kwargs["title"]
        assert "lag=n/a" in kwargs["title"]

    def test_ylabel_uses_pupitre_symbol_and_unit(self, monkeypatch, tmp_path):
        captured = self._capture_plot_comparison(monkeypatch)
        t = np.arange(0, 20, 1.0)
        field = AliasedField(
            "Field", "Courants_Alimentations", "Champ_magn",
            pupitre_unit="tesla", pigbrother_unit="millitesla", pupitre_symbol="Bz",
        )
        pigbrother_df = _bump_df(t, {"Champ_magn": (1000.0, 0.0)})
        pupitre_df = _bump_df(t, {"Field": (1.0, 0.0)})

        compare_field(
            field, "overview", pigbrother_df, pupitre_df,
            plot=True, output_dir=str(tmp_path),
        )

        assert captured["kwargs"]["ylabel"] == "Bz [T]"
        # Plotted pigbrother data was converted into the pupitre unit too --
        # amplitude should read ~1.0 (tesla), not 1000.0 (millitesla).
        assert captured["df1"][captured["y_col1"]].max() == pytest.approx(1.0, rel=1e-6)

    def test_ylabel_falls_back_to_key_without_symbol_or_unit(self, monkeypatch, tmp_path):
        captured = self._capture_plot_comparison(monkeypatch)
        t = np.arange(0, 20, 1.0)
        field = AliasedField("Ucoil1", "Tensions_Aimant", "Interne1")
        pigbrother_df = _bump_df(t, {"Interne1": (2.0, 0.0)})
        pupitre_df = _bump_df(t, {"Ucoil1": (2.0, 0.0)})

        compare_field(
            field, "overview", pigbrother_df, pupitre_df,
            plot=True, output_dir=str(tmp_path),
        )

        assert captured["kwargs"]["ylabel"] == "Ucoil1"

    def test_saves_real_png_end_to_end(self, tmp_path):
        t = np.arange(0, 20, 1.0)
        field = AliasedField(
            "Field", "Courants_Alimentations", "Champ_magn",
            pupitre_unit="tesla", pigbrother_unit="millitesla", pupitre_symbol="Bz",
        )
        pigbrother_df = _bump_df(t, {"Champ_magn": (1000.0, 0.0)})
        pupitre_df = _bump_df(t, {"Field": (1.0, 0.0)})
        lag = LagResult(lag=timedelta(seconds=0.0), method="resample_1s")

        result = compare_field(
            field, "overview", pigbrother_df, pupitre_df,
            plot=True, output_dir=str(tmp_path), reference_lag=lag,
        )

        expected_path = tmp_path / "Field_overview_resample_1s_comparison.png"
        assert result.plot_path == str(expected_path)
        assert expected_path.exists()


# =============================================================================
# compare_all_fields
# =============================================================================
class TestCompareAllFields:
    FIELDS = [
        AliasedField("Idcct1", "Courants_Alimentations", "Courant_A1"),
        AliasedField("Ucoil1", "Tensions_Aimant", "Interne1"),
    ]
    TRUE_LAG = 2.0

    def _record(self) -> OverviewRecord:
        return OverviewRecord(filename="test", housing="M9")

    def _patch_loaders(self, monkeypatch, include_archive: bool = False):
        t = np.arange(0, 20, 1.0)
        pupitre_df = _bump_df(
            t, {"Idcct1": (1.0, self.TRUE_LAG), "Idcct3": (1.0, self.TRUE_LAG), "Ucoil1": (2.0, self.TRUE_LAG)}
        )
        pigbrother_data = {
            ("overview", "Courants_Alimentations"): _bump_df(t, {"Courant_A1": (1.0, 0.0)}),
            ("overview", "Tensions_Aimant"): _bump_df(t, {"Interne1": (2.0, 0.0)}),
        }
        if include_archive:
            pigbrother_data[("archive", "Courants_Alimentations")] = _bump_df(
                t, {"Courant_A1": (1.0, 0.0)}
            )
            pigbrother_data[("archive", "Tensions_Aimant")] = _bump_df(
                t, {"Interne1": (2.0, 0.0)}
            )

        load_calls: list[tuple[str, str]] = []
        pupitre_key_requests: list[list[str]] = []

        def fake_load_pigbrother_group(record, source, group, cache):
            key = (source, group)
            if key in cache:
                return cache[key]
            load_calls.append(key)
            df = pigbrother_data.get(key, pd.DataFrame())
            cache[key] = df
            return df

        def fake_load_pupitre_fields(record, keys):
            pupitre_key_requests.append(list(keys))
            return pupitre_df

        monkeypatch.setattr(field_comparison, "_load_pigbrother_group", fake_load_pigbrother_group)
        monkeypatch.setattr(field_comparison, "_load_pupitre_fields", fake_load_pupitre_fields)
        return load_calls, pupitre_key_requests

    def test_returns_expected_structure_with_high_correlation(self, monkeypatch):
        self._patch_loaders(monkeypatch)
        record = self._record()

        results = compare_all_fields(record, fields=self.FIELDS, sources=("overview",))

        assert set(results.keys()) == {"Idcct1", "Ucoil1"}
        assert results["Idcct1"]["overview"].available
        assert results["Idcct1"]["overview"].metrics["distances"].correlation > 0.9
        assert results["Ucoil1"]["overview"].available
        assert results["Ucoil1"]["overview"].metrics["distances"].correlation > 0.9

        # Should not raise
        print_comparison_summary(results)

    def test_lag_method_none_skips_lag_correction(self, monkeypatch):
        self._patch_loaders(monkeypatch)
        record = self._record()

        corrected = compare_all_fields(record, fields=self.FIELDS, sources=("overview",))
        raw = compare_all_fields(
            record, fields=self.FIELDS, sources=("overview",), lag_method="none"
        )

        assert raw["Idcct1"]["overview"].reference_lag is None
        assert raw["Idcct1"]["overview"].reference_field is None
        assert raw["Idcct1"]["overview"].available

        # Without lag correction, the TRUE_LAG-shifted bumps line up worse
        # than the lag-corrected comparison -- proving no shift was applied.
        assert (
            raw["Idcct1"]["overview"].metrics["distances"].correlation
            < corrected["Idcct1"]["overview"].metrics["distances"].correlation
        )

        # Should not raise
        print_comparison_summary(raw)

    def test_archive_resample_1s_skips_lag_and_warns(self, monkeypatch, caplog):
        self._patch_loaders(monkeypatch, include_archive=True)
        record = self._record()

        with caplog.at_level("WARNING", logger="python_magnetrun.analysis.field_comparison"):
            raw = compare_all_fields(
                record, fields=self.FIELDS, sources=("archive",), lag_method="resample_1s"
            )
        none = compare_all_fields(
            record, fields=self.FIELDS, sources=("archive",), lag_method="none"
        )

        assert raw["Idcct1"]["archive"].reference_lag is None
        assert raw["Idcct1"]["archive"].reference_field is None
        assert raw["Idcct1"]["archive"].available

        # Same result as lag_method="none" -- no shift was computed or applied.
        assert (
            raw["Idcct1"]["archive"].metrics["distances"].correlation
            == none["Idcct1"]["archive"].metrics["distances"].correlation
        )

        assert any(
            "resample_1s" in r.message and "archive" in r.message for r in caplog.records
        )

        # Should not raise
        print_comparison_summary(raw)

    def test_reference_group_loaded_only_once(self, monkeypatch):
        load_calls, _ = self._patch_loaders(monkeypatch)
        record = self._record()

        compare_all_fields(record, fields=self.FIELDS, sources=("overview",))

        # Courants_Alimentations is needed both for the reference lag and for
        # the Idcct1 field comparison, but must be loaded only once.
        assert load_calls.count(("overview", "Courants_Alimentations")) == 1
        assert load_calls.count(("overview", "Tensions_Aimant")) == 1

    def test_always_requests_reference_keys_from_pupitre(self, monkeypatch):
        _, pupitre_key_requests = self._patch_loaders(monkeypatch)
        record = self._record()

        # Only ask to compare Ucoil1 -- Idcct1/Idcct3 aren't in `fields`, but
        # must still be requested since they're needed for the reference lag.
        compare_all_fields(
            record, fields=[AliasedField("Ucoil1", "Tensions_Aimant", "Interne1")], sources=("overview",)
        )

        assert "Idcct1" in pupitre_key_requests[0]
        assert "Idcct3" in pupitre_key_requests[0]

    def test_raises_when_reference_pair_unavailable(self, monkeypatch):
        t = np.arange(0, 20, 1.0)
        pupitre_df = _bump_df(t, {"SomethingElse": (1.0, 0.0)})
        pigbrother_data = {
            ("overview", "Courants_Alimentations"): _bump_df(t, {"SomethingElse": (1.0, 0.0)}),
        }

        def fake_load_pigbrother_group(record, source, group, cache):
            key = (source, group)
            if key not in cache:
                cache[key] = pigbrother_data.get(key, pd.DataFrame())
            return cache[key]

        def fake_load_pupitre_fields(record, keys):
            return pupitre_df

        monkeypatch.setattr(field_comparison, "_load_pigbrother_group", fake_load_pigbrother_group)
        monkeypatch.setattr(field_comparison, "_load_pupitre_fields", fake_load_pupitre_fields)

        with pytest.raises(ValueError, match="no reference channel pair available"):
            compare_all_fields(self._record(), fields=self.FIELDS, sources=("overview",))

    def test_skips_source_with_no_data(self, monkeypatch):
        self._patch_loaders(monkeypatch)
        record = self._record()

        # "archive" has no data in the fake loaders (empty DataFrame), so it
        # should be skipped entirely rather than raising.
        results = compare_all_fields(record, fields=self.FIELDS, sources=("overview", "archive"))

        assert "archive" not in results["Idcct1"]
        assert "overview" in results["Idcct1"]

    def test_no_channel_map_skips_filtering(self, monkeypatch):
        # record.sources is None (default OverviewRecord), so
        # _load_overview_channel_map returns {} and no field is dropped.
        self._patch_loaders(monkeypatch)
        record = self._record()

        results = compare_all_fields(record, fields=self.FIELDS, sources=("overview",))

        assert set(results.keys()) == {"Idcct1", "Ucoil1"}

    def test_drops_field_absent_from_overview_channel_map(self, monkeypatch):
        self._patch_loaders(monkeypatch)
        record = self._record()
        monkeypatch.setattr(
            field_comparison,
            "_load_overview_channel_map",
            lambda record: {"Courants_Alimentations": {"Courant_A1", "Courant_A3"}},
        )

        results = compare_all_fields(record, fields=self.FIELDS, sources=("overview",))

        # Ucoil1 -> Tensions_Aimant/Interne1 isn't in the fake channel map,
        # so it should be dropped entirely rather than marked unavailable.
        assert set(results.keys()) == {"Idcct1"}
        assert results["Idcct1"]["overview"].available

    def test_redirects_field_to_merged_tap_channel(self, monkeypatch):
        t = np.arange(0, 20, 1.0)
        pupitre_df = _bump_df(
            t, {"Idcct1": (1.0, self.TRUE_LAG), "Idcct3": (1.0, self.TRUE_LAG), "Ucoil2": (2.0, self.TRUE_LAG)}
        )
        pigbrother_data = {
            ("overview", "Courants_Alimentations"): _bump_df(t, {"Courant_A1": (1.0, 0.0)}),
            ("overview", "Tensions_Aimant"): _bump_df(t, {"Interne1-2": (2.0, 0.0)}),
        }

        def fake_load_pigbrother_group(record, source, group, cache):
            key = (source, group)
            if key not in cache:
                cache[key] = pigbrother_data.get(key, pd.DataFrame())
            return cache[key]

        def fake_load_pupitre_fields(record, keys):
            return pupitre_df

        monkeypatch.setattr(field_comparison, "_load_pigbrother_group", fake_load_pigbrother_group)
        monkeypatch.setattr(field_comparison, "_load_pupitre_fields", fake_load_pupitre_fields)
        monkeypatch.setattr(
            field_comparison,
            "_load_overview_channel_map",
            lambda record: {
                "Courants_Alimentations": {"Courant_A1", "Courant_A3"},
                "Tensions_Aimant": {"Interne1-2"},
            },
        )

        record = self._record()
        fields = [
            AliasedField("Ucoil1", "Tensions_Aimant", "Interne1"),
            AliasedField("Ucoil2", "Tensions_Aimant", "Interne2"),
        ]
        results = compare_all_fields(record, fields=fields, sources=("overview",))

        # Ucoil1 has no standalone reading (tap 1 isn't independently
        # wired) and is dropped entirely.
        assert "Ucoil1" not in results
        # Ucoil2 is redirected to the merged "Interne1-2" channel.
        assert results["Ucoil2"]["overview"].available
        assert results["Ucoil2"]["overview"].metrics["distances"].correlation > 0.9
