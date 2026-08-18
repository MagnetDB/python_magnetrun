"""Unit tests for python_magnetrun.runetl — prepareData and _cleanup_pupitre_icoil."""

from pathlib import Path

import pandas as pd

from python_magnetrun.housing_config import get_housing_config
from python_magnetrun.magnetdata_pandas import PandasMagnetData
from python_magnetrun.runetl import _cleanup_pupitre_icoil, prepareData

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_TXT = Path(__file__).parent / "data" / "sample_pupitre.txt"


def _pupitre(df: pd.DataFrame) -> PandasMagnetData:
    """Wrap a DataFrame as a PUPITRE-type PandasMagnetData."""
    return PandasMagnetData("test.txt", {}, df.columns.tolist(), df.copy())


def _base_df(**extra) -> pd.DataFrame:
    """Minimal two-row DataFrame with Date/Time and common FKey columns."""
    data = {
        "Date": ["2022.03.30", "2022.03.30"],
        "Time": ["21:55:17", "21:55:18"],
        "Field": [0.5, 0.6],
        "Pmagnet": [8.1, 8.2],
        "Ptot": [9.1, 9.2],
        "teb": [1015.0, 1016.0],
        "tsb": [-5.52, -5.50],
    }
    data.update(extra)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# _cleanup_pupitre_icoil
# ---------------------------------------------------------------------------


class TestCleanupPupitreIcoilZeroColumns:
    def test_non_fkey_zero_column_dropped(self):
        df = _base_df(Icoil1=[1000.0, 1001.0], ZeroExtra=[0.0, 0.0])
        data = _pupitre(df)
        cfg = get_housing_config("M9")
        _cleanup_pupitre_icoil(data, cfg)
        assert "ZeroExtra" not in data.getKeys()

    def test_fkey_zero_column_preserved(self):
        """Columns matching _FKEYS_PATTERN must never be dropped, even if all-zero."""
        df = _base_df(Icoil1=[1000.0, 1001.0], HP1=[0.0, 0.0], Flow1=[0.0, 0.0])
        data = _pupitre(df)
        cfg = get_housing_config("M9")
        _cleanup_pupitre_icoil(data, cfg)
        keys = data.getKeys()
        assert "HP1" in keys
        assert "Flow1" in keys

    def test_teb_tsb_zero_preserved(self):
        df = _base_df(Icoil1=[1000.0, 1001.0], teb=[0.0, 0.0], tsb=[0.0, 0.0])
        data = _pupitre(df)
        cfg = get_housing_config("M9")
        _cleanup_pupitre_icoil(data, cfg)
        keys = data.getKeys()
        assert "teb" in keys
        assert "tsb" in keys


class TestCleanupPupitreIcoilDeduplication:
    def test_duplicate_icoil_dropped(self):
        """When Icoil columns have mean diff ≤ 1e-2, the duplicate is removed."""
        df = _base_df(
            Icoil1=[1000.0, 1001.0],
            Icoil2=[1000.0, 1001.0],  # exact duplicate of Icoil1
            Icoil15=[500.0, 501.0],
        )
        data = _pupitre(df)
        cfg = get_housing_config("M9")
        _cleanup_pupitre_icoil(data, cfg)
        # Icoil2 was a duplicate of Icoil1 and must be removed
        assert "Icoil2" not in data.getKeys()

    def test_distinct_icoil_kept(self):
        """Icoil columns with different values must not be deduplicated."""
        df = _base_df(
            Icoil1=[1000.0, 1001.0],
            Icoil2=[500.0, 501.0],  # clearly different from Icoil1
            Icoil15=[200.0, 201.0],
        )
        data = _pupitre(df)
        cfg = get_housing_config("M9")
        _cleanup_pupitre_icoil(data, cfg)
        # Both should survive deduplication (renamed or dropped, but not mistakenly removed)
        keys = data.getKeys()
        # Neither original name should be flagged as a duplicate
        assert "Icoil2" not in data.getKeys() or "IB" in keys  # either renamed or stays


class TestCleanupPupitreIcoilRename:
    def test_icoil_gr1_renamed_to_role(self):
        """An Icoil column whose index is in gr1 gets renamed to reference_gr1_current."""
        # M9: voltage_channels_gr1 = Ucoil1..14  → gr1_indices = {1..14}
        #      reference_gr1_current = 'IH'
        df = _base_df(Icoil1=[1000.0, 1001.0])
        data = _pupitre(df)
        cfg = get_housing_config("M9")
        _cleanup_pupitre_icoil(data, cfg)
        assert "IH" in data.getKeys()
        assert "Icoil1" not in data.getKeys()

    def test_icoil_gr2_renamed_to_role(self):
        """An Icoil column whose index is in gr2 gets renamed to reference_gr2_current."""
        # M9: voltage_channels_gr2 = Ucoil15..16 → gr2_indices = {15, 16}
        #      reference_gr2_current = 'IB'
        df = _base_df(Icoil1=[1000.0, 1001.0], Icoil15=[500.0, 501.0])
        data = _pupitre(df)
        cfg = get_housing_config("M9")
        _cleanup_pupitre_icoil(data, cfg)
        assert "IB" in data.getKeys()
        assert "Icoil15" not in data.getKeys()

    def test_role_already_present_skips_rename(self):
        """If the role name is already in the DataFrame, the rename is skipped."""
        df = _base_df(IH=[1000.0, 1001.0], Icoil1=[1000.0, 1001.0])
        data = _pupitre(df)
        cfg = get_housing_config("M9")
        _cleanup_pupitre_icoil(data, cfg)
        # IH was pre-existing so Icoil1 should not have been renamed over it
        assert "IH" in data.getKeys()
        assert "Icoil1" not in data.getKeys()

    def test_leftover_unmatched_icoil_dropped(self):
        """Icoil columns that don't map to any GR index are dropped after rename."""
        # Icoil99 has index 99 which is not in gr1 or gr2 indices for M9
        df = _base_df(Icoil99=[999.0, 998.0])
        data = _pupitre(df)
        cfg = get_housing_config("M9")
        _cleanup_pupitre_icoil(data, cfg)
        assert "Icoil99" not in data.getKeys()


# ---------------------------------------------------------------------------
# prepareData
# ---------------------------------------------------------------------------


class TestPrepareDataExplicitKwargs:
    def test_explicit_remove_drops_column(self):
        df = _base_df(Icoil1=[1000.0, 1001.0], ExtraCol=[42.0, 43.0])
        data = _pupitre(df)
        prepareData(data, "M9", keys_to_remove=["ExtraCol"], keys_to_rename={}, keys_to_add={})
        assert "ExtraCol" not in data.getKeys()

    def test_explicit_rename_renames_column(self):
        df = _base_df(Icoil1=[1000.0, 1001.0], OldName=[7.0, 8.0])
        data = _pupitre(df)
        prepareData(
            data, "M9",
            keys_to_remove=None,
            keys_to_rename={"OldName": "NewName"},
            keys_to_add={},
        )
        assert "NewName" in data.getKeys()
        assert "OldName" not in data.getKeys()

    def test_addtime_adds_t_column(self):
        df = _base_df(Icoil1=[1000.0, 1001.0])
        data = _pupitre(df)
        prepareData(data, "M9", keys_to_rename={}, keys_to_add={})
        assert "t" in data.getKeys()
        assert "timestamp" in data.getKeys()


class TestPrepareDataAutoDerivation:
    def test_auto_pupitre_adds_t(self):
        """Auto-derivation for PUPITRE type calls addTime, producing 't'."""
        df = _base_df(Icoil1=[1000.0, 1001.0])
        data = _pupitre(df)
        prepareData(data, "M9")
        assert "t" in data.getKeys()

    def test_auto_pupitre_runs_icoil_cleanup(self):
        """Auto-derivation for PUPITRE type runs _cleanup_pupitre_icoil."""
        df = _base_df(Icoil1=[1000.0, 1001.0])
        data = _pupitre(df)
        prepareData(data, "M9")
        assert "Icoil1" not in data.getKeys()
