"""Boundary conversion from narwhals/Polars objects back to pandas.

See ``prompts/narwhals-downstream-consumers.plan.md`` for why this exists:
``PolarsMagnetData.getData()`` returns a narwhals-wrapped frame (the Phase 2
boundary from ``mrun-cache-implementation.plan.md``), while
``extractData()``/``extractTimeData()`` return a *raw* Polars DataFrame
(they don't go through the narwhals boundary). Either way, existing
consumers were written assuming a plain :class:`pandas.DataFrame` and chain
pandas-only idioms (``.iloc``, ``.mode()``, ``.to_csv()``, item assignment,
...) directly on the result. Converting at the boundary — right after the
call — lets that existing code stay unchanged, at the cost of one
conversion per call (not a hot loop; these are CLI/analysis/plotting call
sites, not per-sample operations).

``PandasMagnetData.getData()`` and ``TdmsMagnetData.getData()`` don't wrap
with narwhals yet (Phases 2 / 1+2b-tdms respectively), so most callers see a
plain :class:`pandas.DataFrame` already — :func:`to_pandas` is a no-op for
those and only does real work for a narwhals- or Polars-backed result.
"""

from __future__ import annotations

from typing import Any

import narwhals as nw
import narwhals.dependencies as nw_dep


def to_pandas(data: Any) -> Any:
    """Convert *data* to pandas if it's a narwhals or raw Polars frame/series.

    Parameters
    ----------
    data : Any
        Typically the return value of ``MagnetDataBase.getData()``,
        ``extractData()``, or ``extractTimeData()`` — a
        :class:`pandas.DataFrame`, :class:`narwhals.DataFrame`/`.Series`, or
        a raw :class:`polars.DataFrame`/`.Series`.

    Returns
    -------
    Any
        A :class:`pandas.DataFrame`/:class:`pandas.Series` if *data* was a
        narwhals or Polars object; *data* unchanged otherwise (already
        pandas, or not a frame/series at all).
    """
    if isinstance(data, (nw.DataFrame, nw.Series)):
        return data.to_pandas()
    if nw_dep.is_polars_dataframe(data) or nw_dep.is_polars_series(data):
        return data.to_pandas()
    return data


__all__ = ["to_pandas"]
