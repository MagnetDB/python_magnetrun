"""
Tests for python_magnetrun.analysis.field_comparison.

These tests use synthetic DataFrames and monkeypatched loaders — no real
Overview/Archive/pupitre files are available in this repository.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from python_magnetrun.analysis import field_comparison
from python_magnetrun.analysis.field_comparison import (
    AliasedField,
    compare_all_fields,
    compare_field,
    compute_reference_lag,
    discover_pupitre_pigbrother_fields,
    print_comparison_summary,
)
from python_magnetrun.analysis.processing import OverviewRecord

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

        assert by_key["Idcct1"] == AliasedField("Idcct1", "Courants_Alimentations", "Courant_A1")
        assert by_key["Idcct3"] == AliasedField("Idcct3", "Courants_Alimentations", "Courant_A3")
        assert by_key["Field"] == AliasedField("Field", "Courants_Alimentations", "Champ_magn")
        assert by_key["Ucoil1"] == AliasedField("Ucoil1", "Tensions_Aimant", "Interne1")
        assert by_key["UH"] == AliasedField("UH", "Tensions_Aimant", "ALL_internes")
        assert by_key["UB"] == AliasedField("UB", "Tensions_Aimant", "ALL_externes")

    def test_expected_count(self):
        # Field, Idcct1-4, Ucoil1-7, Ucoil15, Ucoil16, UH, UB
        fields = discover_pupitre_pigbrother_fields()
        assert len(fields) == 16

    def test_no_group_prefix_in_pupitre_key(self):
        fields = discover_pupitre_pigbrother_fields()
        for f in fields:
            assert "/" not in f.pupitre_key


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

        result = compute_reference_lag(
            self._record(), "overview", pupitre_df, pigbrother_df, self.REFERENCE_FIELDS
        )
        assert abs(result.lag.total_seconds() - 2.0) < 0.5

    def test_falls_back_to_idcct3_when_idcct1_missing(self):
        t = np.arange(0, 20, 1.0)
        pupitre_df = _bump_df(t, {"Idcct3": (1.0, 3.0)})  # no Idcct1 column
        pigbrother_df = _bump_df(t, {"Courant_A3": (1.0, 0.0)})  # no Courant_A1 column

        result = compute_reference_lag(
            self._record(), "overview", pupitre_df, pigbrother_df, self.REFERENCE_FIELDS
        )
        assert abs(result.lag.total_seconds() - 3.0) < 0.5

    def test_falls_back_when_pigbrother_channel_missing(self):
        t = np.arange(0, 20, 1.0)
        pupitre_df = _bump_df(t, {"Idcct1": (1.0, 1.0), "Idcct3": (1.0, 1.0)})
        pigbrother_df = _bump_df(t, {"Courant_A3": (1.0, 0.0)})  # Courant_A1 absent

        result = compute_reference_lag(
            self._record(), "overview", pupitre_df, pigbrother_df, self.REFERENCE_FIELDS
        )
        assert abs(result.lag.total_seconds() - 1.0) < 0.5

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

    def _patch_loaders(self, monkeypatch):
        t = np.arange(0, 20, 1.0)
        pupitre_df = _bump_df(
            t, {"Idcct1": (1.0, self.TRUE_LAG), "Idcct3": (1.0, self.TRUE_LAG), "Ucoil1": (2.0, self.TRUE_LAG)}
        )
        pigbrother_data = {
            ("overview", "Courants_Alimentations"): _bump_df(t, {"Courant_A1": (1.0, 0.0)}),
            ("overview", "Tensions_Aimant"): _bump_df(t, {"Interne1": (2.0, 0.0)}),
        }

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
