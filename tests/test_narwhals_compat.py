"""Unit tests for utils.narwhals_compat.to_pandas."""

from __future__ import annotations

import narwhals as nw
import pandas as pd
import polars as pl

from python_magnetrun.utils.narwhals_compat import to_pandas


def test_converts_narwhals_dataframe():
    nwdf = nw.from_native(pl.DataFrame({"a": [1, 2, 3]}), eager_only=True)
    result = to_pandas(nwdf)
    assert isinstance(result, pd.DataFrame)
    assert result["a"].tolist() == [1, 2, 3]


def test_converts_narwhals_series():
    nwdf = nw.from_native(pl.DataFrame({"a": [1, 2, 3]}), eager_only=True)
    result = to_pandas(nwdf["a"])
    assert isinstance(result, pd.Series)
    assert result.tolist() == [1, 2, 3]


def test_converts_raw_polars_dataframe():
    """extractData()/extractTimeData() return raw Polars, not narwhals-wrapped."""
    pldf = pl.DataFrame({"a": [1, 2, 3]})
    result = to_pandas(pldf)
    assert isinstance(result, pd.DataFrame)
    assert result["a"].tolist() == [1, 2, 3]


def test_converts_raw_polars_series():
    result = to_pandas(pl.Series("a", [1, 2, 3]))
    assert isinstance(result, pd.Series)
    assert result.tolist() == [1, 2, 3]


def test_passes_through_plain_pandas_dataframe():
    pdf = pd.DataFrame({"a": [1, 2, 3]})
    assert to_pandas(pdf) is pdf


def test_passes_through_non_frame():
    assert to_pandas(42) == 42
    assert to_pandas(None) is None
