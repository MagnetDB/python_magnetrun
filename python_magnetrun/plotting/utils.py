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

    # Step 1: aliases always win
    for f in fields:
        result[f] = aliases.get(f, None)

    # Step 2: auto-disambiguate non-aliased fields by symbol
    unresolved = [f for f in fields if result[f] is None]
    raw = {f: (field_metas[f].symbol if field_metas.get(f) else f) for f in unresolved}
    clashing = {sym for sym, n in Counter(raw.values()).items() if n > 1}

    tentative: dict[str, str] = {}  # no-suffix clash group, needs second pass
    for f, sym in raw.items():
        if sym not in clashing:
            # No clash: meta.label if set, else symbol, else field name
            meta = field_metas.get(f)
            lbl = meta.label if (meta and meta.label) else ""
            result[f] = lbl or sym or f
        else:
            suffix = _extract_suffix(f)
            if suffix:
                result[f] = f"{sym}_{suffix}"
            else:
                # No extractable suffix: defer to second pass
                meta = field_metas.get(f)
                tentative[f] = meta.label if (meta and meta.label) else ""

    # Step 3: for no-suffix clash, use label only when it is unique in the group;
    # if labels collide or are absent, fall back to the field name (always unique).
    if tentative:
        label_counts = Counter(tentative.values())
        for f, lbl in tentative.items():
            result[f] = lbl if (lbl and label_counts[lbl] == 1) else f

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
