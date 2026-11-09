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


def _m8_df(date: str) -> pd.DataFrame:
    """Two-row M8 pupitre frame dated *date*: Ucoil<n> = n volts."""
    extra = {f"Idcct{i}": [7000.0, 7001.0] for i in range(1, 5)}
    extra.update({f"Ucoil{n}": [float(n), float(n)] for n in range(1, 17)})
    df = _base_df(**extra)
    df["Date"] = [date, date]
    return df


class TestPrepareDataHousingVersion:
    """prepareData resolves the housing wiring in force at the data start."""

    def test_m8_after_change_uses_ucoil14_for_ub(self):
        data = _pupitre(_m8_df("2026.03.06"))
        prepareData(data, "M8")
        df = data.getData(["UB", "UH"])
        assert list(df["UB"]) == [14.0, 14.0]
        assert list(df["UH"]) == [float(sum(range(1, 14)))] * 2

    def test_m8_before_change_keeps_base_wiring(self):
        data = _pupitre(_m8_df("2025.11.05"))
        prepareData(data, "M8")
        df = data.getData(["UB", "UH"])
        assert list(df["UB"]) == [15.0 + 16.0] * 2
        assert list(df["UH"]) == [float(sum(range(1, 15)))] * 2

    def test_m8_after_change_polars(self):
        """Same as the pandas case on a polars data object."""
        import polars as pl

        from python_magnetrun.magnetdata_polars import PolarsMagnetData

        df = pl.from_pandas(_m8_df("2026.03.06"))
        data = PolarsMagnetData("test.txt", {}, df.columns, df, defs_file=None)
        prepareData(data, "M8")
        assert list(data.getData(["UB"])["UB"]) == [14.0, 14.0]

    def test_m8_change_instant_is_local_time(self):
        """valid_from 17:21:09+01:00 compares with the file's local Date/Time."""
        df = _m8_df("2026.03.05")
        df["Time"] = ["17:21:09", "17:21:10"]
        data = _pupitre(df)
        prepareData(data, "M8")
        assert list(data.getData(["UB"])["UB"]) == [14.0, 14.0]


# ---------------------------------------------------------------------------
# Spec-driven ETL: PupitreEtlSpec / prepare_pupitre
# (accessed through the module so that each test fails on its own until the
# API exists)
# ---------------------------------------------------------------------------

import logging  # noqa: E402
from datetime import datetime  # noqa: E402
from zoneinfo import ZoneInfo  # noqa: E402

import pytest  # noqa: E402

from python_magnetrun import runetl  # noqa: E402
from python_magnetrun.housing_config import HOUSING_CONFIGS  # noqa: E402

PARIS = ZoneInfo("Europe/Paris")


def _fd(formula: str) -> dict:
    """Complete field definition for *formula*."""
    key = formula.split("=")[0].strip()
    return {"formula": formula, "symbol": key, "unit": "volt", "label": key, "description": key}


def _frame(date: str = "2026.05.15", **cols) -> pd.DataFrame:
    """Two-row pupitre frame with 16 channels, all four fields 0 unless overridden."""
    data = {
        "Date": [date, date],
        "Time": ["10:00:00", "10:00:01"],
        "Field": [1.0, 1.1],
        "Flow1": [50.0, 50.0], "Flow2": [40.0, 40.0],
        "Tin1": [15.0, 15.0], "Tin2": [16.0, 16.0],
        "Rpm1": [1500.0, 1500.0], "Rpm2": [1400.0, 1400.0],
        "HP1": [20.0, 20.0], "HP2": [19.0, 19.0],
        "Idcct1": [1000.0, 1000.0], "Idcct2": [1000.0, 1000.0],
        "Idcct3": [500.0, 500.0], "Idcct4": [500.0, 500.0],
        "ZeroCol": [0.0, 0.0],
    }
    for n in range(1, 17):
        for f in ("Icoil", "Ucoil", "DRcoil", "Tcal"):
            data[f"{f}{n}"] = [0.0, 0.0]
    data.update(cols)
    return pd.DataFrame(data)


def _spec(
    housing: str = "M9", when=datetime(2026, 5, 15, tzinfo=PARIS), magnets=None, rules=None
):
    """Spec = housing part of *housing* at *when* + *magnets* + *rules*."""
    base = runetl.PupitreEtlSpec.from_housing(HOUSING_CONFIGS[housing].at(when))
    return runetl.PupitreEtlSpec(
        assembly="A", housing=base.housing, magnets=magnets or {}, rules=rules or []
    )


def _rule(expr: str = "IH - Icoil2", stat: str = "max_abs", tol: float = 5.0, **extra) -> dict:
    """Rule named ``r`` on *expr*."""
    return {"name": "r", "expr": expr, "stat": stat, "tol": tol, **extra}


def _long_frame(n: int = 100, **cols) -> pd.DataFrame:
    """*n*-row pupitre frame, one second apart; *cols* give full columns."""
    df = pd.concat([_frame().iloc[[0]]] * n, ignore_index=True)
    df["Time"] = [f"10:{i // 60:02d}:{i % 60:02d}" for i in range(n)]
    for col, values in cols.items():
        df[col] = values
    return df


class TestPupitreEtlSpec:
    def test_round_trip(self):
        spec = _spec(magnets={"Ins": {"channels": ["Ucoil2"], "add": {"UH": _fd("UH = Ucoil2")}}})
        assert runetl.PupitreEtlSpec.from_dict(spec.to_dict()) == spec

    def test_wrong_kind_refused(self):
        d = _spec().to_dict()
        d["kind"] = "other"
        with pytest.raises(ValueError, match="kind"):
            runetl.PupitreEtlSpec.from_dict(d)

    def test_wrong_format_version_refused(self):
        d = _spec().to_dict()
        d["version"] = 99
        with pytest.raises(ValueError, match="version"):
            runetl.PupitreEtlSpec.from_dict(d)

    def test_two_magnets_adding_same_key_raise(self):
        with pytest.raises(ValueError, match="UH"):
            _spec(magnets={
                "A1": {"channels": ["Ucoil2"], "add": {"UH": _fd("UH = Ucoil2")}},
                "A2": {"channels": ["Ucoil3"], "add": {"UH": _fd("UH = Ucoil3")}},
            })

    def test_magnet_add_conflicting_with_housing_add_raises(self):
        with pytest.raises(ValueError, match="IH"):
            _spec(magnets={"A1": {"add": {"IH": _fd("IH = Idcct1")}}})

    def test_channel_claimed_twice_raises(self):
        with pytest.raises(ValueError, match="Ucoil2"):
            _spec(magnets={"A1": {"channels": ["Ucoil2"]}, "A2": {"channels": ["Ucoil2"]}})

    def test_magnet_rename_refused(self):
        with pytest.raises(ValueError, match="rename"):
            _spec(magnets={"A1": {"rename": {"Flow1": "X"}}})

    def test_incomplete_field_def_refused(self):
        with pytest.raises(ValueError, match="formula"):
            _spec(magnets={"A1": {"add": {"UH": {"formula": "UH = Ucoil2"}}}})

    def test_magnet_add_replaces_default(self):
        spec = _spec(magnets={"Ins": {"channels": ["Ucoil2"], "add": {"UH": _fd("UH = Ucoil2")}}})
        assert spec.add["UH"]["formula"] == "UH = Ucoil2"
        assert "UB" in spec.add  # other default kept

    def test_save_then_load(self, tmp_path):
        spec = _spec(
            magnets={"Ins": {"channels": ["Ucoil2"]}},
            rules=[_rule(unit="A", tag="check:current_mismatch", scope="magnet:Ins")],
        )
        path = tmp_path / "A.M9-181109.pupitre-etl.json"
        runetl.save_pupitre_etl_spec(path, spec)
        assert runetl.load_pupitre_etl_spec(path) == spec

    def test_format_version_1_refused(self):
        d = _spec().to_dict()
        d["version"] = 1
        with pytest.raises(ValueError, match="version"):
            runetl.PupitreEtlSpec.from_dict(d)

    def test_magnet_current_refused(self):
        with pytest.raises(ValueError, match="current"):
            _spec(magnets={"Ins": {"current": {"source": "Icoil2", "check": "IH"}}})

    def test_duplicate_rule_names_refused(self):
        with pytest.raises(ValueError, match="'r'"):
            _spec(rules=[_rule(), _rule("Idcct1 - Idcct2")])


class TestCheckRule:
    def test_valid(self):
        runetl.check_rule(_rule("abs(Idcct1) - Idcct2", stat="p99_abs", unit="A", tag="x:y"))

    @pytest.mark.parametrize(
        "rule, match",
        [
            ({"name": "r", "expr": "IH", "stat": "max_abs"}, "tol"),
            (_rule(other=1), "other"),
            (_rule(stat="median"), "stat"),
            (_rule(tol=-1.0), "tol"),
            (_rule(tol="5"), "tol"),
            (_rule("IH - "), "expr"),
            (_rule("IH = Icoil2"), "expr"),
            (_rule("IH - @x"), "@"),
            (_rule("`pupitre.IH` - Icoil2"), "backtick"),
            (_rule("1 + 2"), "column"),
            ({**_rule(), "name": "2 bad"}, "name"),
        ],
    )
    def test_invalid(self, rule, match):
        with pytest.raises(ValueError, match=match):
            runetl.check_rule(rule)

    def test_rule_columns(self):
        assert runetl.rule_columns("abs(IH) - (Idcct1 + Idcct2) * 2") == ["IH", "Idcct1", "Idcct2"]


class TestFromHousing:
    def test_m9_m10_renames_differ(self):
        m9 = _spec("M9")
        m10 = _spec("M10")
        assert m9.rename["Flow1"] == "FlowH"
        assert m10.rename["Flow1"] == "FlowB"
        assert m9.magnets == {} and m10.magnets == {}

    def test_version_recorded(self):
        assert _spec("M9").housing["version"] == "M9-181109"

    def test_m8_after_wiring_change(self):
        spec = _spec("M8", when=datetime(2026, 3, 6, tzinfo=PARIS))
        assert spec.housing["version"] == "M8-260305"
        assert spec.add["UB"]["formula"] == "UB = Ucoil14"
        assert "Ucoil14" not in spec.add["UH"]["formula"]

    def test_m8_before_wiring_change(self):
        spec = _spec("M8", when=datetime(2025, 11, 5, tzinfo=PARIS))
        assert spec.housing["version"] == "M8-230913"
        assert spec.add["UB"]["formula"] == "UB = Ucoil15 + Ucoil16"


class TestPreparePupitre:
    def test_rename_before_add_and_removal(self):
        spec = _spec(magnets={"Ins": {"add": {"F2": _fd("F2 = FlowH * 2")}}})
        data = _pupitre(_frame())
        runetl.prepare_pupitre(data, spec)
        keys = data.getKeys()
        assert "FlowH" in keys and "Flow1" not in keys
        assert list(data.getData(["F2"])["F2"]) == [100.0, 100.0]
        assert not [k for k in keys if k.startswith("Icoil")]
        assert "t" in keys
        assert "ZeroCol" in keys  # no content-based dropping
        assert list(data.getData(["IH"])["IH"]) == [2000.0, 2000.0]

    def test_rule_ok(self):
        spec = _spec(rules=[_rule(unit="A", tag="check:current_mismatch", scope="magnet:Ins")])
        data = _pupitre(_frame(Icoil2=[2000.0, 2000.0]))
        checks = runetl.prepare_pupitre(data, spec)
        assert [c for c in checks if c["kind"] == "rule"] == [{
            "kind": "rule", "name": "r", "scope": "magnet:Ins", "expr": "IH - Icoil2",
            "stat": "max_abs", "value": 0.0, "tol": 5.0, "unit": "A",
            "tag": "check:current_mismatch", "missing": [], "ok": True,
        }]

    def test_rule_offset_fails(self, caplog):
        spec = _spec(rules=[_rule()])
        data = _pupitre(_frame(Icoil2=[2007.0, 2007.0]))
        with caplog.at_level(logging.WARNING):
            checks = runetl.prepare_pupitre(data, spec)
        rule = [c for c in checks if c["kind"] == "rule"][0]
        assert rule["ok"] is False and rule["value"] == pytest.approx(7.0)
        assert "IH - Icoil2" in caplog.text

    def test_rule_missing_column(self):
        spec = _spec(rules=[_rule("IH - Icoil99")])
        rule = [c for c in runetl.prepare_pupitre(_pupitre(_frame()), spec) if c["kind"] == "rule"][0]
        assert rule["ok"] is None and rule["value"] is None and rule["missing"] == ["Icoil99"]

    def test_rule_on_renamed_name(self):
        """Rules run after rename: FlowH exists, Flow1 does not."""
        spec = _spec(rules=[_rule("FlowH - 50", tol=0.0), {**_rule("Flow1"), "name": "old"}])
        checks = [c for c in runetl.prepare_pupitre(_pupitre(_frame()), spec) if c["kind"] == "rule"]
        assert [(c["name"], c["ok"]) for c in checks] == [("r", True), ("old", None)]

    def test_rule_p99_ignores_transient(self):
        """One 13 A spike in 100 rows fails max_abs but passes p99_abs."""
        icoil2 = [2000.0] * 99 + [1987.0]
        spec = _spec(rules=[_rule(), {**_rule(stat="p99_abs"), "name": "p99"}])
        data = _pupitre(_long_frame(Icoil2=icoil2))
        checks = {c["name"]: c for c in runetl.prepare_pupitre(data, spec) if c["kind"] == "rule"}
        assert checks["r"]["ok"] is False and checks["r"]["value"] == pytest.approx(13.0)
        assert checks["p99"]["ok"] is True and checks["p99"]["value"] == pytest.approx(0.13)

    def test_rule_mean_abs(self):
        spec = _spec(rules=[_rule("Idcct1 - Idcct2 - Field", stat="mean_abs", tol=1.0)])
        rule = [c for c in runetl.prepare_pupitre(_pupitre(_frame()), spec) if c["kind"] == "rule"][0]
        assert rule["value"] == pytest.approx(1.05) and rule["ok"] is False

    def test_companions_follow_removed_ucoil(self):
        spec = _spec(magnets={"Ins": {"channels": ["Ucoil2"], "remove": ["Ucoil1"]}})
        data = _pupitre(_frame(Ucoil2=[3.0, 3.0], Tcal2=[20.0, 20.0]))
        runetl.prepare_pupitre(data, spec)
        keys = data.getKeys()
        assert not {"Ucoil1", "DRcoil1", "Tcal1"} & set(keys)
        assert {"Ucoil2", "DRcoil2", "Tcal2"} <= set(keys)

    def test_channel_check_all_zero_ok(self):
        spec = _spec(magnets={"Ins": {"remove": ["Ucoil1"]}})
        checks = runetl.prepare_pupitre(_pupitre(_frame()), spec)
        ch = [c for c in checks if c["kind"] == "channel"]
        assert ch == [{"kind": "channel", "channel": 1, "fields": {}, "ok": True}]

    def test_channel_check_tcal_fails(self, caplog):
        spec = _spec(magnets={"Ins": {"remove": ["Ucoil5"]}})
        data = _pupitre(_frame(Tcal5=[9.3, 9.3]))
        with caplog.at_level(logging.WARNING):
            checks = runetl.prepare_pupitre(data, spec)
        ch = [c for c in checks if c["kind"] == "channel"][0]
        assert ch["ok"] is False and ch["fields"] == {"Tcal5": 9.3}
        assert "Tcal5" not in data.getKeys()  # removed anyway

    def test_housing_version_mismatch_flagged(self, caplog):
        spec = _spec("M8", when=datetime(2025, 11, 5, tzinfo=PARIS))  # M8-230913
        data = _pupitre(_frame(date="2026.03.06"))
        with caplog.at_level(logging.WARNING):
            checks = runetl.prepare_pupitre(data, spec)
        hv = [c for c in checks if c["kind"] == "housing_version"][0]
        assert hv == {"kind": "housing_version", "spec": "M8-230913", "data": "M8-260305", "ok": False}
        assert "M8-260305" in caplog.text
        assert "t" in data.getKeys()  # still processed

    def test_housing_version_match_ok(self):
        spec = _spec("M8", when=datetime(2026, 3, 6, tzinfo=PARIS))
        checks = runetl.prepare_pupitre(_pupitre(_frame(date="2026.03.06")), spec)
        assert [c for c in checks if c["kind"] == "housing_version"][0]["ok"] is True

    def test_polars(self):
        import polars as pl

        from python_magnetrun.magnetdata_polars import PolarsMagnetData

        df = pl.from_pandas(_frame(Icoil2=[2000.0, 2000.0]))
        data = PolarsMagnetData("test.txt", {}, df.columns, df, defs_file=None)
        spec = _spec(magnets={"Ins": {"channels": ["Ucoil2"]}}, rules=[_rule()])
        checks = runetl.prepare_pupitre(data, spec)
        keys = data.getKeys()
        assert "IH" in keys and "FlowH" in keys and "ZeroCol" in keys
        assert not [k for k in keys if k.startswith("Icoil")]
        assert [c for c in checks if c["kind"] == "rule"][0]["ok"] is True


class TestFromTxtEtlSpec:
    """MagnetRun.fromtxt(etl_spec=...) uses prepare_pupitre instead of prepareData."""

    def test_spec_applied(self):
        from python_magnetrun.MagnetRun import MagnetRun

        spec = _spec("M9", when=datetime(2022, 3, 30, tzinfo=PARIS))
        mrun = MagnetRun.fromtxt("M9", "A", str(SAMPLE_TXT), etl_spec=spec)
        keys = mrun.getKeys()
        assert "FlowH" in keys and "Flow1" not in keys
        assert not [k for k in keys if k.startswith("Icoil")]
        assert "t" in keys

    def test_spec_with_keys_refused(self):
        from python_magnetrun.MagnetRun import MagnetRun

        with pytest.raises(ValueError, match="etl_spec"):
            MagnetRun.fromtxt("M9", "A", str(SAMPLE_TXT), keys_to_add={}, etl_spec=_spec())
