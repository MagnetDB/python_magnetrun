# Plan: HybridRun.getData — hybrid_formula_map key resolution

## Context

`HybridRun.getData` accepts keys in the form `type/system[/variable]` (e.g.
`kHz/FEPC-AUX-LNCMI/ALIM1_J1`).  The M8 housing config defines computed
channels in `hybrid_formula_map`:

```json
"hybrid_formula_map": {
  "FEPC-AUX-LNCMI/ALIM1": {
    "formula": "FEPC-AUX-LNCMI/ALIM1 = FEPC-AUX-LNCMI/ALIM1_J1 + FEPC-AUX-LNCMI/ALIM1_J2",
    "symbol": "I_ALIM1",
    "unit": "ampere",
    ...
  }
}
```

These keys (`FEPC-AUX-LNCMI/ALIM1`) are also stored in
`reference_gr1_hybrid` / `reference_gr2_hybrid` and passed to `getData` by
`_get_hybrid_group` in `processing.py`.

When `getData` receives `FEPC-AUX-LNCMI/ALIM1` it parses it as:

- `data_type = "FEPC-AUX-LNCMI"` — not `"kHz"`, `"rms"`, or `"trigger"`
- Falls to `else: raise ValueError("Unknown data type: FEPC-AUX-LNCMI")`

In the pandas (`MagnetRun`) flow these keys work because `cleanupData` adds
the computed channel to the DataFrame.  The `HybridRun` flow has no such step.

---

## Failure mode

| Call site | Key passed | Current result |
|---|---|---|
| `processing.py:623` | `"FEPC-AUX-LNCMI/ALIM1"` | `ValueError: Unknown data type: FEPC-AUX-LNCMI` |
| `processing.py:623` | `"FEPC-AUX-LNCMI/ALIM2"` | same |

---

## Design

Add formula-key resolution inside `HybridRun.getData` **before** the
`type/system[/variable]` parse block.  The check:

1. Load the housing config for `self.Housing`.
2. If `key` is in `config.hybrid_formula_map`, call the new helper
   `_resolve_hybrid_formula(key, opts)`.
3. Return its result directly (bypassing the normal parse path).

The cache check already runs *before* the parse block, so formula results are
cached transparently on subsequent calls.

### Formula parsing

Supported grammar (covers all current M8 formulas):

```
<lhs> = <operand1> + <operand2> [+ <operand3> ...]
```

Parser steps:
1. Split on `"="` → take the RHS.
2. Split RHS on `"+"` → strip whitespace from each token → list of bare channel
   names (e.g. `["FEPC-AUX-LNCMI/ALIM1_J1", "FEPC-AUX-LNCMI/ALIM1_J2"]`).
3. Each bare name `SYSTEM/VARIABLE` maps to `kHz/SYSTEM/VARIABLE`.

Only `+` (sum) is implemented; any other operator raises `NotImplementedError`.

### Helper method

```python
def _resolve_hybrid_formula(
    self,
    key: str,
    formula_str: str,
    opts: LoadOptions,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a hybrid_formula_map formula and return (data, time)."""
```

- Parses the formula string.
- Calls `self.getData(f"kHz/{system}/{variable}", options=opts)` for each
  operand (recursive — also benefits from cache).
- Verifies all time arrays are shape-compatible (same length); raises
  `ValueError` with a clear message if not.
- Returns `(sum_of_data_arrays, time_of_first_operand)`.

### Insertion point in `getData`

```python
# After cache check, before parts = key.split("/")
try:
    from ..housing_config import get_housing_config
    hcfg = get_housing_config(self.Housing)
    formula_entry = hcfg.hybrid_formula_map.get(key)
    if formula_entry:
        data, time = self._resolve_hybrid_formula(
            key, formula_entry["formula"], opts
        )
        if opts.cache:
            self._add_to_cache(cache_key, data, time, opts)
        return data, time
except (ValueError, KeyError):
    pass  # housing unknown or key not in formula map — fall through
```

---

## Files affected

| File | Change |
|---|---|
| `python_magnetrun/hybrid/hybrid_run.py` | Add `_resolve_hybrid_formula`; insert formula-key guard in `getData` |

No other files need to change.

---

## Tests to add

In `tests/hybrid/test_hybrid_run_formula.py` (new file):

- `test_resolve_formula_two_operands`: mock `read_khz_variable` for J1 and J2;
  assert `getData("FEPC-AUX-LNCMI/ALIM1")` returns their element-wise sum and
  the correct time array.
- `test_resolve_formula_cache_hit`: call `getData` twice with the same formula
  key; assert `read_khz_variable` is called only once (cache hit on second call).
- `test_resolve_formula_shape_mismatch`: operands have different-length arrays;
  assert `ValueError` is raised with a clear message.
- `test_getData_normal_key_unaffected`: a normal `kHz/...` key still works as
  before (regression guard).
- `test_getData_unknown_key_raises`: a key that is neither a formula key nor a
  valid `type/system/variable` still raises `ValueError`.

---

## Non-goals

- Subtraction, multiplication, or division operators — not used in current
  configs; raise `NotImplementedError` if encountered.
- Modifying the pandas / `MagnetRun` flow — it already handles formula keys via
  `cleanupData`.
- Changing `_get_hybrid_group` — keys stay as-is; the resolution happens inside
  `getData`.
