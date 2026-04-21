"""Shared label and legend formatting utilities for all plot paths."""
from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pint

    from ..magnetdata_base import FieldMeta


def format_axis_label(symbol: str, unit: pint.Unit | None) -> str:
    """Return 'symbol [unit]' if unit is known, else 'symbol'."""
    if unit is None:
        return symbol
    return f"{symbol} [{unit:~P}]"


def format_legend_label(
    key: str,
    basename: str | None = None,
    unit: pint.Unit | None = None,
    max_val: float | None = None,
) -> str:
    """Build a legend entry following the canonical format.

    basename set  → multi-file mode: "basename: key"
    unit     set  → mixed-unit overlay: "key [unit]" (or "basename: key [unit]")
    max_val  set  → normalized: append "  (max = X.XX [unit])" or "  (max = X.XX)"
    """
    label = f"{basename}: {key}" if basename else key
    if unit is not None:
        label = f"{label} [{unit:~P}]"
    if max_val is not None:
        unit_str = f" [{unit:~P}]" if unit is not None else ""
        label = f"{label}  (max = {max_val:.3g}{unit_str})"
    return label


def resolve_legend_labels(
    fields: list[str],
    field_metas: dict[str, FieldMeta | None],
    aliases: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return {field: legend_label} with symbol-clash resolution.

    Priority per field:
      1. aliases[field]    — explicit caller override, always wins
      2. meta.label        — from JSON "label" field (e.g. "I_{GR1}")
      3. auto-disambiguate — symbol_suffix when symbol is not unique
      4. field name        — always-unique fallback
    """
    aliases = aliases or {}
    result: dict[str, str | None] = {}
    for f in fields:
        if f in aliases:
            result[f] = aliases[f]
        else:
            meta = field_metas.get(f)
            result[f] = meta.label if (meta and meta.label) else None

    # For fields without an assigned label, auto-disambiguate by symbol
    unresolved = [f for f in fields if result[f] is None]
    raw = {
        f: (field_metas[f].symbol if field_metas.get(f) else f)
        for f in unresolved
    }
    clashing = {sym for sym, n in Counter(raw.values()).items() if n > 1}
    for f, sym in raw.items():
        suffix = _extract_suffix(f)
        result[f] = f"{sym}_{suffix}" if (sym in clashing and suffix) else (sym or f)

    return result  # type: ignore[return-value]


def _extract_suffix(field_name: str) -> str:
    """Extract a distinguishing suffix from a field name.

    Examples:
        'Group/Courant_GR1'              → 'GR1'
        'kHz/FEPC-AUX-LNCMI/ALIM1_J1'   → 'J1'
        'Field'                           → ''
    """
    base = field_name.split("/")[-1]
    parts = base.rsplit("_", 1)
    return parts[-1] if len(parts) == 2 else ""
