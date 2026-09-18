"""Unit tests for PolarsMagnetData ETL methods.

Covers:
  - fromtxt() construction, lazy loading, start_timestamp derivation
  - Units() / getUnitKey() / Groups population
  - addTime() computes t/timestamp and drops Date/Time
  - cleanupData(keys_to_add=...) calls addData for each entry
  - cleanupData(): all-zero column drop, exact-duplicate column drop
  - addData() / computeData() / removeData() / renameData()
  - extractData() / getData() / getStartDate() / getDuration() / shiftTime()
  - extractTimeData() / saveData() / plotData() / stats() (Phase B methods
    exercised by real call sites: viewcsv.py, commands/select.py, MagnetRun.py)
  - header-only file rejected the same way as PandasMagnetData
  - full ETL parity against PandasMagnetData on real fixtures
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from python_magnetrun.housing_config import get_housing_config
from python_magnetrun.magnetdata_pandas import PandasMagnetData
from python_magnetrun.magnetdata_polars import PolarsMagnetData
from python_magnetrun.utils.validation import FileFormatError

DATA_DIR = Path(__file__).parent / "data"
SAMPLE_PUPITRE = DATA_DIR / "sample_pupitre.txt"
HEADER_ONLY = DATA_DIR / "pupitre_header_only.txt"
REAL_M10 = Path(__file__).parent.parent / "data" / "M10_2020.10.23---20_10_41.txt"
REAL_M9_MALFORMED = (
    Path(__file__).parent.parent / "data" / "M9_2019.02.14---23_00_38.txt"
)


def _make_polars(df: pl.DataFrame | None = None, keys: list[str] | None = None) -> PolarsMagnetData:
    """Build a minimal PolarsMagnetData without going through fromtxt()."""
    if df is None:
        df = pl.DataFrame({"ChA": [1.0, 2.0, 3.0], "ChB": [4.0, 5.0, 6.0]})
    if keys is None:
        keys = df.columns
    return PolarsMagnetData("test.txt", {}, keys, df, defs_file=None)


# ---------------------------------------------------------------------------
# fromtxt / construction
# ---------------------------------------------------------------------------


class TestFromtxt:
    def test_type_is_pupitre(self):
        from python_magnetrun.magnetdata_base import DataType

        data = PolarsMagnetData.fromtxt(str(SAMPLE_PUPITRE))
        assert data.Type == DataType.PUPITRE

    def test_keys_from_stub(self):
        data = PolarsMagnetData.fromtxt(str(SAMPLE_PUPITRE))
        assert "Date" in data.getKeys()
        assert "Field" in data.getKeys()

    def test_lazy_load_full_file(self):
        data = PolarsMagnetData.fromtxt(str(SAMPLE_PUPITRE))
        assert data.Data.height >= 1  # full file loaded on first .Data access

    def test_header_only_file_raises(self):
        with pytest.raises(FileFormatError):
            PolarsMagnetData.fromtxt(str(HEADER_ONLY))

    def test_start_timestamp_set(self):
        data = PolarsMagnetData.fromtxt(str(SAMPLE_PUPITRE))
        assert data.start_timestamp is not None


# ---------------------------------------------------------------------------
# Units / Groups
# ---------------------------------------------------------------------------


class TestUnits:
    def test_units_populated(self):
        data = PolarsMagnetData.fromtxt(str(SAMPLE_PUPITRE), defs_file="pupitre-defs.json")
        data.Units()
        assert "Field" in data.units
        symbol, unit = data.getUnitKey("Field")
        assert symbol
        assert unit is not None

    def test_groups_populated_from_defs(self):
        data = PolarsMagnetData.fromtxt(str(SAMPLE_PUPITRE), defs_file="pupitre-defs.json")
        data.Units()
        assert len(data.Groups) > 0


# ---------------------------------------------------------------------------
# addTime
# ---------------------------------------------------------------------------


class TestAddTime:
    def test_adds_t_and_timestamp_drops_date_time(self):
        data = PolarsMagnetData.fromtxt(str(SAMPLE_PUPITRE))
        ret = data.addTime()
        assert ret == 0
        assert "t" in data.getKeys()
        assert "timestamp" in data.getKeys()
        assert "Date" not in data.getKeys()
        assert "Time" not in data.getKeys()

    def test_t_starts_at_zero(self):
        data = PolarsMagnetData.fromtxt(str(SAMPLE_PUPITRE))
        data.addTime()
        assert data.Data["t"][0] == 0.0

    def test_missing_date_time_raises(self):
        data = _make_polars()
        with pytest.raises(RuntimeError):
            data.addTime()


# ---------------------------------------------------------------------------
# cleanupData
# ---------------------------------------------------------------------------


class TestCleanupData:
    def test_keys_to_add_calls_addData(self):
        data = _make_polars()
        data.cleanupData(
            keys_to_add={
                "Sum": {
                    "formula": "Sum = ChA + ChB",
                    "symbol": "S",
                    "unit": None,
                    "label": "",
                    "description": "",
                }
            }
        )
        assert "Sum" in data.getKeys()
        assert data.Data["Sum"].to_list() == [5.0, 7.0, 9.0]

    def test_keys_to_remove_drops_column(self):
        data = _make_polars()
        data.cleanupData(keys_to_remove=["ChB"])
        assert "ChB" not in data.getKeys()

    def test_keys_to_rename(self):
        data = _make_polars()
        data.cleanupData(keys_to_rename={"ChA": "Renamed"})
        assert "Renamed" in data.getKeys()
        assert "ChA" not in data.getKeys()

    def test_drops_all_zero_column(self):
        df = pl.DataFrame({"ChA": [1.0, 2.0], "Zero": [0.0, 0.0]})
        data = _make_polars(df)
        data.cleanupData()
        assert "Zero" not in data.getKeys()

    def test_all_zero_flow_column_protected(self):
        df = pl.DataFrame({"ChA": [1.0, 2.0], "Flow1": [0.0, 0.0]})
        data = _make_polars(df)
        data.cleanupData()
        assert "Flow1" in data.getKeys()

    def test_drops_duplicate_column(self):
        df = pl.DataFrame({"ChA": [1.0, 2.0], "Dup": [1.0, 2.0]})
        data = _make_polars(df)
        data.cleanupData()
        assert set(data.getKeys()) == {"ChA"}


# ---------------------------------------------------------------------------
# addData / computeData
# ---------------------------------------------------------------------------


class TestAddData:
    def test_simple_sum_formula(self):
        data = _make_polars()
        status = data.addData(
            "Sum", "Sum = ChA + ChB", symbol="S", unit=None, label="", description=""
        )
        assert status == 0
        assert data.Data["Sum"].to_list() == [5.0, 7.0, 9.0]

    def test_existing_key_skipped(self):
        data = _make_polars()
        status = data.addData(
            "ChA", "ChA = ChA + 1", symbol="S", unit=None, label="", description=""
        )
        assert status == 1

    def test_undefined_variable_skipped(self):
        data = _make_polars()
        status = data.addData(
            "Sum", "Sum = ChA + Nope", symbol="S", unit=None, label="", description=""
        )
        assert status != 0
        assert "Sum" not in data.getKeys()


class TestComputeData:
    def test_row_wise_apply(self):
        data = _make_polars()
        status = data.computeData(
            lambda a, b: a - b,
            "Diff",
            ["ChA", "ChB"],
            symbol="D",
            unit=None,
            label="",
            description="",
        )
        assert status == 0
        assert data.Data["Diff"].to_list() == [-3.0, -3.0, -3.0]


# ---------------------------------------------------------------------------
# removeData / renameData
# ---------------------------------------------------------------------------


class TestRemoveRenameData:
    def test_remove_data(self):
        data = _make_polars()
        data.removeData(["ChB"])
        assert "ChB" not in data.getKeys()

    def test_rename_data(self):
        data = _make_polars()
        data.renameData({"ChA": "Renamed"})
        assert "Renamed" in data.getKeys()

    def test_rename_conflict_raises(self):
        data = _make_polars()
        with pytest.raises(ValueError):
            data.renameData({"ChA": "ChB"})


# ---------------------------------------------------------------------------
# extract / getData / time utilities
# ---------------------------------------------------------------------------


class TestExtractAndGetData:
    def test_extract_data(self):
        data = _make_polars()
        sub = data.extractData(["ChA"])
        assert sub.columns == ["ChA"]

    def test_get_data_selection(self):
        data = _make_polars()
        sub = data.getData("ChA")
        assert sub.columns == ["ChA"]

    def test_get_data_downsample_not_implemented(self):
        data = _make_polars()
        with pytest.raises(NotImplementedError):
            data.getData(downsample=object())

    def test_shift_time(self):
        data = _make_polars(pl.DataFrame({"t": [0.0, 1.0, 2.0]}))
        data.shiftTime(5.0)
        assert data.Data["t"].to_list() == [5.0, 6.0, 7.0]

    def test_get_duration(self):
        data = PolarsMagnetData.fromtxt(str(SAMPLE_PUPITRE))
        data.addTime()
        assert data.getDuration() > 0


# ---------------------------------------------------------------------------
# Phase B methods: extractTimeData / saveData / plotData / stats
# (exercised by real call sites: viewcsv.py, commands/select.py, MagnetRun.py)
# ---------------------------------------------------------------------------


def _prepared_sample() -> PolarsMagnetData:
    """A real, addTime()'d + Units()'d PolarsMagnetData for Phase B tests."""
    data = PolarsMagnetData.fromtxt(str(SAMPLE_PUPITRE), defs_file="pupitre-defs.json")
    data.addTime()
    data.Units()
    return data


class TestExtractTimeData:
    def test_filters_by_range(self):
        data = _prepared_sample()
        t0 = data.Data["timestamp"][0]
        t1 = data.Data["timestamp"][-1]
        full_range = f"{t0.strftime('%Y-%m-%d %H:%M:%S')};{t1.strftime('%Y-%m-%d %H:%M:%S')}"
        sub = data.extractTimeData(full_range, time_zone="UTC")
        assert sub.height == data.Data.height

    def test_before_addtime_raises(self):
        data = PolarsMagnetData.fromtxt(str(SAMPLE_PUPITRE))
        with pytest.raises(RuntimeError):
            data.extractTimeData("2022-01-01 00:00:00;2022-01-01 00:00:01")


class TestSaveData:
    def test_writes_tsv(self, tmp_path):
        data = _prepared_sample()
        out = tmp_path / "out.txt"
        status = data.saveData(["Field", "t"], str(out))
        assert status == 0
        lines = out.read_text().splitlines()
        assert lines[0] == "Field\tt"
        assert len(lines) == data.Data.height + 1


class TestPlotData:
    def test_plots_t_vs_field(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        data = _prepared_sample()
        fig, ax = plt.subplots()
        data.plotData(x="t", y="Field", ax=ax)
        assert len(ax.lines) == 1
        plt.close(fig)

    def test_plots_timestamp_normalized(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        data = _prepared_sample()
        fig, ax = plt.subplots()
        data.plotData(x="timestamp", y="Field", ax=ax, normalize=True)
        assert len(ax.lines) == 1
        plt.close(fig)

    def test_unknown_y_key_raises(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        data = _prepared_sample()
        fig, ax = plt.subplots()
        with pytest.raises(RuntimeError):
            data.plotData(x="t", y="Nope", ax=ax)
        plt.close(fig)


class TestStats:
    def test_single_key(self, caplog):
        data = _prepared_sample()
        data.stats("Field")  # should not raise

    def test_all_keys(self, capsys):
        data = _prepared_sample()
        data.stats()
        captured = capsys.readouterr()
        assert "statistic" in captured.out

    def test_unknown_key_raises(self):
        data = _prepared_sample()
        with pytest.raises(RuntimeError):
            data.stats("Nope")


# ---------------------------------------------------------------------------
# Full ETL parity against PandasMagnetData (real fixtures)
# ---------------------------------------------------------------------------


def _prepare(data, cfg):
    data.Units()
    available = data.getKeys()
    keys_to_add = {**cfg.pupitre_formula_map, **cfg.get_pupitre_voltage_formulas(available)}
    keys_to_rename = cfg.get_pupitre_rename_map()
    data.addTime()
    data.cleanupData(keys_to_rename=keys_to_rename, keys_to_add=keys_to_add)
    return data


class TestParityWithPandas:
    @pytest.mark.parametrize(
        "path,housing",
        [
            pytest.param(REAL_M10, "M10", id="M10-clean"),
            pytest.param(REAL_M9_MALFORMED, "M9", id="M9-malformed-header"),
        ],
    )
    def test_same_keys_and_values(self, path, housing):
        if not path.is_file():
            pytest.skip(f"fixture not found: {path}")
        cfg = get_housing_config(housing)

        pl_data = _prepare(PolarsMagnetData.fromtxt(str(path)), cfg)
        pd_data = _prepare(PandasMagnetData.fromtxt(str(path)), cfg)

        assert sorted(pl_data.getKeys()) == sorted(pd_data.getKeys())
        assert pl_data.Data.height == len(pd_data.Data)

        for col in pl_data.getKeys():
            a = pl_data.Data[col].to_numpy()
            b = pd_data.Data[col].to_numpy()
            if a.dtype.kind in "fiu" and b.dtype.kind in "fiu":
                assert np.allclose(a, b, equal_nan=True), f"mismatch in column {col!r}"
