# Label & Legend Uniformization Plan

Date: 2026-04-20

Effort key: **S** = ~1 h, **M** = half-day.

---

## Motivation

Labels and legends are currently set in at least six locations using two different
APIs (`ax.set_*` vs `plt.*`), with inconsistent unit inclusion and legend content.
The result is plots where some axes show `"t [s]"` and others show `"t"`, and
legends that identify only filenames instead of the quantity being plotted.

This plan is a **sub-plan of `plotting-refactoring.plan.md`**.  It fits between
the existing Steps 3b and 7, adding a new **Step 3c** (shared label utility) that
the Steps 7 and 8 completions will consume.

---

## Canonical label format

The standard for all plot labels in this codebase:

| Axis / element | Format | Example |
|---|---|---|
| Axis with known unit | `"symbol [unit]"` | `"B [T]"`, `"t [s]"`, `"I [A]"` |
| Axis without unit metadata | raw column name | `"Timestamp"` |
| Time axis (universal fallback) | `"t [s]"` | always in seconds |
| Legend — single file | `"key"` | `"Champ_magn"` |
| Legend — multiple files | `"basename: key"` | `"M9_260331: Champ_magn"` |
| Legend — normalized overlay | `"key  (max = X.XX [unit])"` | `"I_GR1  (max = 1234.5 [A])"` |
| Legend — mixed-unit overlay | `"key [unit]"` | `"Field_B [T]"`, `"I_GR1 [A]"` |

Unit format is always **pint compact** (`{unit:~P}`): `T`, `A`, `kA`, `°C`.

---

## Inventory of problems

### P1 — `commands/plot.py` `plot_vs_time()` (matplotlib path)

| Line | Problem |
|---|---|
| 645–648 | `normalize` branch is a second `if`, not `elif` → silently overwrites the unit ylabel set on line 646 |
| 655 | `plt.xlabel("t [s]")` — hardcoded; should use `getUnitKey("t")` like lines 652–653 |
| 645–655 | Uses `plt.ylabel` / `plt.xlabel` instead of `my_ax.set_*` |
| 650 | `my_ax.legend(labels=legends)` — legends already formatted well, but no `loc` from style |

### P2 — `commands/plot.py` `plot_key_vs_key()`

| Line | Problem |
|---|---|
| 698–705 | `legends` = filenames only (e.g. `"M9_260331"`); should be `"basename: key1 vs key2"` |
| 734 | `plt.legend(labels=legends)` — uses `plt.*`; no `loc` from style |
| 735 | `plt.title(...)` — missing xlabel/ylabel entirely; `plotData()` sets them internally but axes method vs pyplot consistency is not guaranteed |

### P3 — `analysis/plotting.py`

| Line | Problem |
|---|---|
| 417 | `ax.set_xlabel(tkey)` — `tkey` is the raw column name, no unit; should be `"t [s]"` or resolved via `getUnitKey` |
| 504, 663 | `ax.set_xlabel(x_col)` — raw name, no unit |
| 664 | `ax.set_ylabel("Normalized" if normalize else "Value")` — generic; loses physical meaning |

### P4 — `utils/plots.py`

| Line | Problem |
|---|---|
| 117–118 | `plt.ylabel(key2)` / `plt.xlabel(key1)` — raw names, no units, wrong API |
| 200–201 | same pattern repeated in `plot_files()` |
| 199 | `plt.legend(loc="best")` — uses `plt.*` |

---

## Step 3c — Add `format_label()` utility *(new step, ~S)*

**File:** `python_magnetrun/plotting/utils.py`  *(new)*

```python
from __future__ import annotations
import pint


def format_axis_label(symbol: str, unit: "pint.Unit | None") -> str:
    """Return 'symbol [unit]' if unit is known, else 'symbol'."""
    if unit is None:
        return symbol
    return f"{symbol} [{unit:~P}]"


def format_legend_label(
    key: str,
    basename: str | None = None,
    unit: "pint.Unit | None" = None,
    max_val: float | None = None,
) -> str:
    """Build a legend entry following the canonical format.

    basename  set  → multi-file mode: "basename: key"
    unit      set  → mixed-unit overlay: "key [unit]" (or "basename: key [unit]")
    max_val   set  → normalized: append "  (max = X.XX [unit])" or "  (max = X.XX)"
    """
    label = f"{basename}: {key}" if basename else key
    if unit is not None:
        label = f"{label} [{unit:~P}]"
    if max_val is not None:
        unit_str = f" [{unit:~P}]" if unit is not None else ""
        label = f"{label}  (max = {max_val:.3g}{unit_str})"
    return label
```

`format_axis_label` is already implicitly used in `timeseries.py`'s `_resolve_units()`
helper; this step makes it an explicit, importable utility so the legacy paths (Steps
7 and 8) can share the same logic without copying the f-string.

---

## Step 3c-fix — Fix immediate bugs in `commands/plot.py` *(~S, independent)*

These are bugs in the **already-done** Step 6 and should be fixed regardless of
whether Steps 7–11 are complete.

### Fix 1 — `normalize` must not overwrite unit ylabel

[commands/plot.py:645–648](../python_magnetrun/commands/plot.py#L645)

```python
# Before (broken — second `if` overwrites the first)
if symbol is not None and unit is not None:
    plt.ylabel(f"{symbol} [{unit:~P}]")
if args.normalize:
    plt.ylabel("normalized")

# After
if args.normalize:
    my_ax.set_ylabel("normalized")
elif symbol is not None and unit is not None:
    my_ax.set_ylabel(f"{symbol} [{unit:~P}]")
```

### Fix 2 — Time axis: use `getUnitKey` consistently, remove hardcoded fallback

[commands/plot.py:651–655](../python_magnetrun/commands/plot.py#L651)

```python
# Before
if t0:
    (t_symbol, t_unit) = inputs[input_files[0]]["data"].getMData().getUnitKey("t")
    plt.xlabel(f"{t_symbol} [{t_unit:~P}]")
else:
    plt.xlabel("t [s]")

# After — always try getUnitKey; fall back to "t [s]" only on exception
try:
    t_symbol, t_unit = inputs[input_files[0]]["data"].getMData().getUnitKey("t")
    my_ax.set_xlabel(f"{t_symbol} [{t_unit:~P}]")
except Exception:
    my_ax.set_xlabel("t [s]")
```

### Fix 3 — `plot_key_vs_key`: legend should include key pair

[commands/plot.py:698–705](../python_magnetrun/commands/plot.py#L698)

```python
# Before — legends = ["M9_260331", ...]
legends.append(os.path.basename(file).replace(f_extension, ""))

# After — legends = ["M9_260331: key1 vs key2", ...]
for pair in plot_args:
    items = pair.split("-")
    key1, key2 = items[0], items[1]
    legends.append(f"{os.path.basename(file).replace(f_extension, '')}: {key1} vs {key2}")
```

### Fix 4 — Switch `plt.*` calls to `ax.set_*` in both functions

Replace `plt.legend(labels=legends)` → `my_ax.legend(labels=legends, loc=cfg.style.legend_loc)`
in both `plot_vs_time` and `plot_key_vs_key`.

---

## Step 7 additions — `analysis/plotting.py`

> Extends the existing Step 7 in `plotting-refactoring.plan.md`.

In addition to importing `PlotStyle`/`PlotColors` from `plotting.style` and using
`AnnotationManager`:

- **Line 417** `ax.set_xlabel(tkey)` → `ax.set_xlabel(format_axis_label("t", ureg.second))`
  or simply `ax.set_xlabel("t [s]")` until pint is wired here.
- **Lines 504, 663** `ax.set_xlabel(x_col)` → resolve unit via `getUnitKey(x_col)` if
  available; fall back to `x_col`.
- **Line 664** `ax.set_ylabel("Normalized" if normalize else "Value")` → for normalized
  plots keep `"normalized"`; for raw, use `format_axis_label(symbol, unit)`.
- Apply `fontsize=style.label_fontsize` **consistently** to all `ax.set_xlabel/ylabel`
  calls (currently only line 417 does this).

---

## Step 8 additions — `utils/plots.py`

> Extends the existing Step 8 in `plotting-refactoring.plan.md`.

- Lines 117–118, 200–201: replace `plt.ylabel(key)` / `plt.xlabel(key)` with
  `ax.set_ylabel(key)` / `ax.set_xlabel(key)`.  The `ax` object is already in scope
  (assigned at line 96 and 147).
- Line 199: replace `plt.legend(loc="best")` with `ax.legend(loc="best")`.
- Add optional `xlabel_label`/`ylabel_label` parameters to `plot_scatter()` and
  `plot_files()` so callers can pass pre-formatted `"symbol [unit]"` strings when they
  have unit metadata.

---

## Step 3c-meta — extend `field_defs.py` + `FieldMeta` in `magnetdata_base.py`

> Prerequisite to Step 3c.  No new files.  `field_defs.py` is the single source of
> truth for the JSON schema; `FieldMeta` is a runtime object scoped to the data layer.

### Why not a separate `field_meta.py`

`field_defs.py` already owns the JSON schema (`load_defs`, `save_defs`, `add_field_def`,
`update_field_def`, alias management, CLI).  Adding `label` there keeps schema
knowledge in one place and avoids a parallel module that would duplicate its role.

### Changes to `field_defs.py`

**`add_field_def`** gains a `label` kwarg:

```python
def add_field_def(
    json_file, key, symbol, unit,
    description: str = "",
    label: str | None = None,      # new
    overwrite: bool = False,
) -> None:
    ...
    defs[key] = {"description": description, "symbol": symbol,
                 "unit": unit, "label": label}
```

**`update_field_def`** gains `label` (uses the `_UNSET` sentinel already in place):

```python
def update_field_def(json_file, key, symbol=None, unit=_UNSET,
                     description=None, label=_UNSET) -> None:
    ...
    if label is not _UNSET:
        entry["label"] = label
```

**`list_field_defs`** adds a `label` column to the printed table and returned dicts.

**CLI** — `add` and `update` subcommands gain `--label`.

### `FieldMeta` — defined in `magnetdata_base.py`, not a new file

`FieldMeta` is a runtime object used only by the data layer.  It lives alongside
the class that owns `self.units` rather than in a separate module:

```python
# magnetdata_base.py  (top of file, after imports)
from typing import NamedTuple
import pint

class FieldMeta(NamedTuple):
    symbol: str              # physics symbol for axis: "I", "B", "U"
    unit: "pint.Unit | None"
    label: "str | None" = None       # from JSON "label"; None → auto-disambiguate
    description: "str | None" = None # from JSON "description"
```

`NamedTuple` is unpacked by position exactly like the current 2-tuple, so any
existing `symbol, unit = self.units[key]` continues to work unchanged.

`self.units` stays `dict[str, tuple[str, pint.Unit]]` (backward compat).
A parallel dict carries the extended data:

```python
# MagnetDataBase.__init__
self.field_meta: dict[str, FieldMeta] = {}
```

`load_units_from_json` populates both dicts:

```python
self.units[key] = (symbol, pint_unit)                # unchanged
self.field_meta[key] = FieldMeta(
    symbol=symbol,
    unit=pint_unit,
    label=defn.get("label"),
    description=defn.get("description") or None,
)
```

New accessor — no API break:

```python
def getFieldMeta(self, key: str) -> "FieldMeta | None":
    return self.field_meta.get(key)
```

### JSON schema addition — `"label"` field (optional)

Both `pigbrother-defs.json` and `hybrid-defs.json` gain an optional `"label"` field.
Existing entries without it continue to work via the auto-subscript fallback in
`resolve_legend_labels()`.  Add `"label"` only where the auto-subscript is wrong or
ugly.

```json
"Courants_Alimentations/Courant_GR1": {
    "description": "Total supply current, group GR1",
    "symbol": "I",
    "label": "I_{GR1}",
    "unit": "ampere"
}
```

### `addData` / `computeData` signature fix

**Base class** (`magnetdata_base.py:228`) — fix type annotation; add `label=` and
`description=` kwargs:

```python
def addData(
    self, key: str, formula: str,
    unit: "tuple[str, pint.Unit] | None" = None,  # was: str | None
    label: str | None = None,
    description: str | None = None,
    debug: bool = False,
) -> int: ...

def computeData(
    self, method, key: str, kparams: list,
    unit: "tuple[str, pint.Unit] | None" = None,  # was: tuple | None
    label: str | None = None,
    description: str | None = None,
    debug: bool = False,
) -> None: ...
```

**`magnetdata_pandas.py`** — after storing data:

```python
if unit:
    self.units[key] = unit
symbol = unit[0] if unit else key
pint_unit = unit[1] if unit else None
self.field_meta[key] = FieldMeta(symbol, pint_unit, label, description)
```

**`magnetdata_tdms.py`** — fix Bug 1 (never updates `self.units`) AND populate
`field_meta`:

```python
if unit:
    self.units[key] = unit          # BUG FIX — was missing entirely
channel = key.split("/")[-1]
symbol = unit[0] if unit else channel
pint_unit = unit[1] if unit else None
self.field_meta[key] = FieldMeta(symbol, pint_unit, label, description)
```

### `resolve_legend_labels()` utility

```python
# python_magnetrun/plotting/utils.py
from collections import Counter
from ..magnetdata_base import FieldMeta

def resolve_legend_labels(
    fields: list[str],
    field_metas: "dict[str, FieldMeta | None]",
    aliases: "dict[str, str] | None" = None,
) -> dict[str, str]:
    """Return {field: legend_label} with symbol-clash resolution.

    Priority per field:
      1. aliases[field]      — explicit caller override, always wins
      2. meta.label          — from JSON "label" field (e.g. "I_{GR1}")
      3. auto-disambiguate   — symbol_suffix when symbol is not unique
      4. field name          — always-unique fallback
    """
    aliases = aliases or {}
    result: dict[str, str | None] = {}
    for f in fields:
        if f in aliases:
            result[f] = aliases[f]
        else:
            meta = field_metas.get(f)
            result[f] = meta.label if meta and meta.label else None

    raw = {
        f: (field_metas[f].symbol if field_metas.get(f) else f)
        for f in fields if result[f] is None
    }
    clashing = {sym for sym, n in Counter(raw.values()).items() if n > 1}
    for f, sym in raw.items():
        suffix = _extract_suffix(f)
        result[f] = f"{sym}_{suffix}" if sym in clashing and suffix else sym or f

    return result


def _extract_suffix(field_name: str) -> str:
    """'Group/Courant_GR1' → 'GR1',  'kHz/FEPC-AUX-LNCMI/ALIM1_J1' → 'J1'."""
    base = field_name.split("/")[-1]
    parts = base.rsplit("_", 1)
    return parts[-1] if len(parts) == 2 else ""
```

---

## Hybrid data — additional issues

`HybridData` is a **standalone class** (not a subclass of `MagnetDataBase`) that
reuses `MagnetDataBase.load_units_from_json()` via duck-typing.  The same `FieldMeta`
approach applies, but three additional problems must be fixed first.

### Bug 2 — Key prefix mismatch makes `self.units` always empty

`self.Keys` stores prefixed keys: `"kHz/FEPC-AUX-LNCMI/ALIM1_J1"`.
`hybrid-defs.json` stores unprefixed keys: `"FEPC-AUX-LNCMI/ALIM1_J1"`.
`load_units_from_json()` at `magnetdata_base.py:175` skips any JSON key not in
`self.Keys` → every entry is skipped → **`self.units` is always empty**.

Fix: `HybridData` overrides `load_units_from_json()` with prefix-aware matching.
One JSON entry registers the same `FieldMeta` for all three prefixed variants
(`kHz/`, `rms/`, `trigger/`):

```python
def load_units_from_json(self, json_file: str, debug: bool = False) -> None:
    from ..field_defs import load_defs
    from ..magnetdata_base import FieldMeta, _make_ureg
    ureg = _make_ureg()
    for key, defn in load_defs(json_file).items():
        if key.startswith("_"):
            continue
        matched = [k for k in self.Keys if k.endswith(f"/{key}") or k == key]
        if not matched:
            continue
        symbol = defn["symbol"]
        unit_str = defn.get("unit")
        pint_unit = ureg.parse_expression(unit_str).units if unit_str else None
        meta = FieldMeta(symbol, pint_unit,
                         defn.get("label"), defn.get("description") or None)
        for full_key in matched:
            self.units[full_key] = (symbol, pint_unit)
            self.field_meta[full_key] = meta
```

### Naming inconsistency — `HybridRun.getUnit()` vs `getUnitKey()`

Add `getUnitKey()` as a proper alias on `HybridRun`:

```python
def getUnitKey(self, key: str) -> tuple:
    """Alias for getUnit() — matches MagnetData interface."""
    return self.getUnit(key)
```

### `field_meta` on `HybridData`

`HybridData.__init__` gains `self.field_meta: dict[str, FieldMeta] = {}` and
`getFieldMeta()` identical to the `MagnetDataBase` version.

### `addData` / `computeData` on `HybridData` — lazy evaluation

`HybridData` is not read-only: `hybrid_formula_map` entries (`ALIM1 = ALIM1_J1 +
ALIM1_J2`) must register as deferred recipes evaluated when `getData(key)` is called.
See *Step 3c-hybrid* for implementation details.  `addData` stores into
`self.field_meta` identically to the pandas/TDMS path.

### Editorial work — `hybrid-defs.json`

All `"description"` fields are currently `""`.  Fill in incrementally.  Adding
`"label"` for frequently-plotted channels (`"ALIM1_J1"`, `"ALIM2_J1"`, etc.) is
higher value and not a prerequisite to the code changes.

---

## Integration with `plotting-refactoring.plan.md`

```
Step 3  — timeseries.py (done)             ← defines label-rules table
Step 3b — df.attrs["units"] (done)         ← feeds _resolve_units()
Step 3c-meta — field_defs.py + FieldMeta   ← NEW (this plan)
  ├── field_defs.py: add/update/list gain label=; CLI --label
  ├── magnetdata_base.py: FieldMeta NamedTuple, field_meta dict, getFieldMeta(),
  │     addData/computeData signature fix, load_units_from_json stores FieldMeta
  ├── magnetdata_pandas.py: addData/computeData populate field_meta
  ├── magnetdata_tdms.py: addData Bug 1 fix + populate field_meta
  └── hybrid/hybrid_data.py: field_meta dict, prefix-aware load_units_from_json
        (Bug 2 fix), getFieldMeta(), addData lazy + field_meta
Step 3c — plotting/utils.py (format_axis_label, resolve_legend_labels)  ← NEW
             ↓ consumed by
Step 7  — analysis/plotting.py cleanup     ← extended by this plan
Step 8  — utils/plots.py + hybrid/         ← extended by this plan
Step 11 — examples migration               ← no change needed
```

---

## File change summary

| File | Step | Status | Change |
|---|---|---|---|
| `python_magnetrun/field_defs.py` | 3c-meta | ⏳ todo | `add_field_def`/`update_field_def`/`list_field_defs` gain `label`; CLI `--label` |
| `python_magnetrun/magnetdata_base.py` | 3c-meta | ⏳ todo | `FieldMeta` NamedTuple; `field_meta` dict; `getFieldMeta()`; `load_units_from_json` stores `FieldMeta`; `addData`/`computeData` type fix + `label=`/`description=` |
| `python_magnetrun/magnetdata_pandas.py` | 3c-meta | ⏳ todo | `addData`/`computeData` populate `field_meta` |
| `python_magnetrun/magnetdata_tdms.py` | 3c-meta | ⏳ todo | `addData` Bug 1 fix (`self.units` never updated) + populate `field_meta` |
| `python_magnetrun/hybrid/hybrid_data.py` | 3c-meta | ⏳ todo | `field_meta` dict; prefix-aware `load_units_from_json` (Bug 2 fix); `getFieldMeta()`; `addData` lazy |
| `python_magnetrun/hybrid/hybrid_run.py` | 3c-meta | ⏳ todo | Add `getUnitKey()` alias for `getUnit()` |
| `python_magnetrun/plotting/utils.py` | 3c | ⏳ todo | new — `format_axis_label`, `format_legend_label`, `resolve_legend_labels`, `_extract_suffix` |
| `python_magnetrun/commands/plot.py` | 3c-fix | ⏳ todo | Fix 1–4: normalize/elif, getUnitKey fallback, legend content, `ax.set_*` |
| `python_magnetrun/analysis/plotting.py` | 7 | ⏳ todo | Time axis unit, y-axis unit, consistent fontsize |
| `python_magnetrun/utils/plots.py` | 8 | ⏳ todo | `ax.set_*` API, optional label params |
| `python_magnetrun/pigbrother-defs.json` | 3c-meta | ⏳ editorial | Add `"label"` for frequently-plotted fields |
| `python_magnetrun/hybrid-defs.json` | 3c-meta | ⏳ editorial | Fill `"description"`; add `"label"` for key fields |

---

## Execution order

1. **Step 3c-fix** — fix bugs in `commands/plot.py` now; zero dependencies, high impact.
2. **Step 3c-meta** — `field_defs.py` + `FieldMeta` + `addData` fixes + hybrid prefix fix; independent of Steps 7–8.
3. **Step 3c** — add `plotting/utils.py`; depends on `FieldMeta`; unblocks Steps 7–8.
4. **Step 7** — complete `analysis/plotting.py` migration (label fixes included).
5. **Step 8** — complete `utils/plots.py` + `hybrid/plotting.py` migration.

Steps 2–3 are independent of Steps 4–5 and can land in any order.
