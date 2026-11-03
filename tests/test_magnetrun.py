"""Unit tests for python_magnetrun.MagnetRun and load_mrun."""

from pathlib import Path

import pandas as pd
import pytest

from python_magnetrun.magnetdata_base import DataType
from python_magnetrun.magnetdata_pandas import PandasMagnetData
from python_magnetrun.MagnetRun import MagnetRun, load_mrun

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_TXT = Path(__file__).parent / "data" / "sample_pupitre.txt"

_PUPITRE_STRING = (
    "16\tM22011801\t14440\t15380\t0\t4.24628e-06\t5.02238e-11\t0.0038\t1.30227\n"
    "Date\tTime\tField\tFlow1\tHP1\tIcoil1\tUcoil1\tPmagnet\tPtot\tteb\ttsb\n"
    "2022.03.30\t21:55:17\t0.5\t49.1\t6.1\t1000.0\t2.5\t8.1\t9.1\t1015.0\t-5.52\n"
    "2022.03.30\t21:55:18\t0.6\t49.2\t6.2\t1001.0\t2.6\t8.2\t9.2\t1016.0\t-5.50\n"
)


@pytest.fixture()
def simple_data() -> PandasMagnetData:
    df = pd.DataFrame({"Field": [0.5, 0.6], "IH": [1000.0, 1001.0]})
    return PandasMagnetData("test.txt", {}, df.columns.tolist(), df)


@pytest.fixture()
def mrun_with_data(simple_data) -> MagnetRun:
    return MagnetRun("M9", "test_assembly", simple_data)


@pytest.fixture()
def mrun_no_data() -> MagnetRun:
    return MagnetRun("M9", "test_assembly", None)


# ---------------------------------------------------------------------------
# Constructor and basic attributes
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_housing_stored(self, mrun_with_data):
        assert mrun_with_data.Housing == "M9"

    def test_assembly_stored(self, mrun_with_data):
        assert mrun_with_data.Assembly == "test_assembly"

    def test_data_stored(self, mrun_with_data, simple_data):
        assert mrun_with_data.MagnetData is simple_data

    def test_no_data_none(self, mrun_no_data):
        assert mrun_no_data.MagnetData is None

    def test_start_time_default_none(self, mrun_with_data):
        assert mrun_with_data.StartTime is None


# ---------------------------------------------------------------------------
# Getters / setters
# ---------------------------------------------------------------------------


class TestGettersSetters:
    def test_gethousing(self, mrun_with_data):
        assert mrun_with_data.getHousing() == "M9"

    def test_getassembly(self, mrun_with_data):
        assert mrun_with_data.getAssembly() == "test_assembly"

    def test_getdomain(self, mrun_with_data):
        assert mrun_with_data.getDomain() == "operational"

    def test_sethousing(self, mrun_with_data):
        mrun_with_data.setHousing("M10")
        assert mrun_with_data.getHousing() == "M10"

    def test_setassembly(self, mrun_with_data):
        mrun_with_data.setAssembly("other_assembly")
        assert mrun_with_data.getAssembly() == "other_assembly"

    def test_repr_contains_housing(self, mrun_with_data):
        assert "M9" in repr(mrun_with_data)

    def test_repr_contains_assembly(self, mrun_with_data):
        assert "test_assembly" in repr(mrun_with_data)


# ---------------------------------------------------------------------------
# Data delegation
# ---------------------------------------------------------------------------


class TestDataDelegation:
    def test_gettype_returns_pupitre(self, mrun_with_data):
        assert mrun_with_data.getType() == DataType.PUPITRE

    def test_gettype_no_data_raises(self, mrun_no_data):
        with pytest.raises(RuntimeError, match="no MagnetData"):
            mrun_no_data.getType()

    def test_getmdata_returns_data(self, mrun_with_data, simple_data):
        assert mrun_with_data.getMData() is simple_data

    def test_getmdata_no_data_raises(self, mrun_no_data):
        with pytest.raises(RuntimeError, match="no magnetdata"):
            mrun_no_data.getMData()

    def test_getkeys_returns_list(self, mrun_with_data):
        keys = mrun_with_data.getKeys()
        assert isinstance(keys, list)
        assert "Field" in keys

    def test_getkeys_no_data_raises(self, mrun_no_data):
        with pytest.raises(RuntimeError, match="no MagnetData"):
            mrun_no_data.getKeys()

    def test_getdata_full_returns_dataframe(self, mrun_with_data):
        df = mrun_with_data.getData(None)
        assert isinstance(df, pd.DataFrame)

    def test_getdata_key_returns_series(self, mrun_with_data):
        series = mrun_with_data.getData("Field")
        assert hasattr(series, "__len__")

    def test_getdata_no_data_raises(self, mrun_no_data):
        with pytest.raises(RuntimeError, match="no MagnetData"):
            mrun_no_data.getData()

    def test_getstats_no_data_raises(self, mrun_no_data):
        with pytest.raises(RuntimeError, match="no MagnetData"):
            mrun_no_data.getStats()

    def test_getdataframe_pupitre_returns_dataframe(self, mrun_with_data):
        df = mrun_with_data.getDataFrame()
        assert isinstance(df, pd.DataFrame)

    def test_getdataframe_no_data_raises(self, mrun_no_data):
        with pytest.raises(RuntimeError, match="no MagnetData"):
            mrun_no_data.getDataFrame()


# ---------------------------------------------------------------------------
# Factory classmethods
# ---------------------------------------------------------------------------


class TestFromStringIO:
    def test_returns_magnetrun(self):
        mrun = MagnetRun.fromStringIO("M9", "assembly", _PUPITRE_STRING)
        assert isinstance(mrun, MagnetRun)

    def test_housing_set(self):
        mrun = MagnetRun.fromStringIO("M9", "assembly", _PUPITRE_STRING)
        assert mrun.getHousing() == "M9"

    def test_assembly_set(self):
        mrun = MagnetRun.fromStringIO("M9", "assembly", _PUPITRE_STRING)
        assert mrun.getAssembly() == "assembly"

    def test_type_is_pupitre(self):
        mrun = MagnetRun.fromStringIO("M9", "assembly", _PUPITRE_STRING)
        assert mrun.getType() == DataType.PUPITRE

    def test_keys_non_empty(self):
        mrun = MagnetRun.fromStringIO("M9", "assembly", _PUPITRE_STRING)
        assert len(mrun.getKeys()) > 0

    def test_empty_string_raises(self):
        with pytest.raises((RuntimeError, ValueError, Exception)):
            MagnetRun.fromStringIO("M9", "assembly", "")


class TestFromTxt:
    def test_returns_magnetrun(self):
        mrun = MagnetRun.fromtxt("M9", "assembly", str(SAMPLE_TXT))
        assert isinstance(mrun, MagnetRun)

    def test_housing_set(self):
        mrun = MagnetRun.fromtxt("M9", "assembly", str(SAMPLE_TXT))
        assert mrun.getHousing() == "M9"

    def test_type_is_pupitre(self):
        mrun = MagnetRun.fromtxt("M9", "assembly", str(SAMPLE_TXT))
        assert mrun.getType() == DataType.PUPITRE

    def test_start_time_set(self):
        mrun = MagnetRun.fromtxt("M9", "assembly", str(SAMPLE_TXT))
        assert mrun.StartTime is not None

    def test_keys_non_empty(self):
        mrun = MagnetRun.fromtxt("M9", "assembly", str(SAMPLE_TXT))
        assert len(mrun.getKeys()) > 0

    def test_keys_to_add_forwarded(self):
        usum = {
            "formula": "Usum = Ucoil1 + Ucoil2",
            "symbol": "U",
            "unit": "volt",
            "label": "Usum",
            "description": "test sum",
        }
        mrun = MagnetRun.fromtxt(
            "M9", "assembly", str(SAMPLE_TXT), keys_to_rename={}, keys_to_add={"Usum": usum}
        )
        assert "Usum" in mrun.getKeys()

    def test_keys_to_remove_forwarded(self):
        mrun = MagnetRun.fromtxt(
            "M9", "assembly", str(SAMPLE_TXT),
            keys_to_remove=["Field"], keys_to_rename={}, keys_to_add={},
        )
        assert "Field" not in mrun.getKeys()

    def test_keys_to_rename_forwarded(self):
        mrun = MagnetRun.fromtxt(
            "M9", "assembly", str(SAMPLE_TXT),
            keys_to_rename={"Tout": "Tout_renamed"}, keys_to_add={},
        )
        assert "Tout_renamed" in mrun.getKeys()
        assert "Tout" not in mrun.getKeys()


class TestLoadMrun:
    def test_txt_dispatches_to_fromtxt(self):
        mrun = load_mrun(str(SAMPLE_TXT), housing="M9", auto_resolve=False)
        assert isinstance(mrun, MagnetRun)
        assert mrun.getType() == DataType.PUPITRE

    def test_unknown_extension_raises(self, tmp_path):
        bad = tmp_path / "file.xyz"
        bad.write_text("dummy")
        with pytest.raises(ValueError, match="unsupported file extension"):
            load_mrun(str(bad), auto_resolve=False)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_mrun(str(tmp_path / "nonexistent.txt"), auto_resolve=False)


# ---------------------------------------------------------------------------
# load_mrun: explicit ETL arguments (spec-driven ETL, never silently ignored)
# ---------------------------------------------------------------------------

from datetime import datetime  # noqa: E402
from zoneinfo import ZoneInfo  # noqa: E402

from python_magnetrun.housing_config import get_housing_config  # noqa: E402
from python_magnetrun.runetl import PupitreEtlSpec, save_pupitre_etl_spec  # noqa: E402

_ETL_COLS = (
    ["Date", "Time", "Field", "Flow1", "Flow2", "Tin1", "Tin2", "Rpm1", "Rpm2", "HP1", "HP2"]
    + [f"Idcct{i}" for i in range(1, 5)]
    + [f"{f}{n}" for n in range(1, 17) for f in ("Icoil", "Ucoil", "DRcoil", "Tcal")]
)


@pytest.fixture()
def m9_etl(tmp_path):
    """Synthetic M9 file starting 2026-05-15 10:00:00 and the spec dir with A.M9-181109."""
    values = {"Field": "1.0", "Flow1": "50", "Flow2": "40", "Idcct1": "1000", "Idcct2": "1000"}
    rows = [
        "\t".join({"Date": "2026.05.15", "Time": t, **values}.get(c, "0") for c in _ETL_COLS)
        for t in ("10:00:00", "10:00:01")
    ]
    path = tmp_path / "2026.05.15 - 10:00:00.txt"
    path.write_text("16\tM0\t1\t1\n" + "\t".join(_ETL_COLS) + "\n" + "\n".join(rows) + "\n")
    cfg = get_housing_config("M9", at=datetime(2026, 5, 15, 10, tzinfo=ZoneInfo("Europe/Paris")))
    spec = PupitreEtlSpec.from_housing(cfg, assembly="A")
    specs = tmp_path / "specs"
    specs.mkdir()
    save_pupitre_etl_spec(specs / "A.M9-181109.pupitre-etl.json", spec)
    return path, specs, spec


class TestLoadMrunEtl:
    @pytest.mark.parametrize("ext", [".tdms", ".csv"])
    @pytest.mark.parametrize("arg", ["etl_spec", "etl_dir", "keys_to_add"])
    def test_etl_arguments_refused_for_non_txt(self, m9_etl, tmp_path, ext, arg):
        value = {"etl_spec": m9_etl[2], "etl_dir": str(m9_etl[1]), "keys_to_add": {}}[arg]
        with pytest.raises(ValueError, match="txt"):
            load_mrun(str(tmp_path / f"x{ext}"), housing="M9", auto_resolve=False, **{arg: value})

    def test_etl_spec_applied(self, m9_etl):
        path, _, spec = m9_etl
        mrun = load_mrun(str(path), housing="M9", assembly="A", auto_resolve=False, etl_spec=spec)
        keys = mrun.getKeys()
        assert "FlowH" in keys and not [k for k in keys if k.startswith("Icoil")]
        assert "Ucoil1" in keys  # no content-based dropping

    def test_etl_dir_picks_version_file(self, m9_etl):
        path, specs, _ = m9_etl
        mrun = load_mrun(str(path), housing="M9", assembly="A", auto_resolve=False, etl_dir=str(specs))
        assert "FlowH" in mrun.getKeys() and "Ucoil1" in mrun.getKeys()

    def test_etl_dir_needs_assembly(self, m9_etl):
        path, specs, _ = m9_etl
        with pytest.raises(ValueError, match="assembly"):
            load_mrun(str(path), housing="M9", auto_resolve=False, etl_dir=str(specs))

    def test_etl_dir_missing_spec_file(self, m9_etl):
        path, specs, _ = m9_etl
        with pytest.raises(FileNotFoundError, match="B.M9-181109.pupitre-etl.json"):
            load_mrun(str(path), housing="M9", assembly="B", auto_resolve=False, etl_dir=str(specs))

    def test_etl_spec_and_dir_exclusive(self, m9_etl):
        path, specs, spec = m9_etl
        with pytest.raises(ValueError, match="etl_dir"):
            load_mrun(str(path), housing="M9", assembly="A", auto_resolve=False,
                      etl_spec=spec, etl_dir=str(specs))

    def test_spec_of_other_housing_refused(self, m9_etl):
        path, _, spec = m9_etl
        with pytest.raises(ValueError, match="M9"):
            MagnetRun.fromtxt("M8", "A", str(path), etl_spec=spec)
