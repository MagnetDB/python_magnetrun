# Remove the `MagnetData` deprecation shim

## Prerequisite

This prompt assumes `load_magnetdata()` already exists in `magnetdata.py` and
`MagnetData` carries a `.. deprecated::` docstring notice.  See
`prompts/replace-magnetdata-shim.prompt.md` for that prior step.

---

## Goal

Delete the `MagnetData` class from `magnetdata.py` entirely and fix every
remaining reference in the package and tests.

---

## Inventory of usages

### A — Direct constructor `MagnetData(filename, Groups, Keys, Type, Data)`

All occurrences use `Type=0` (PUPITRE), so they all map to `PandasMagnetData`.

| File | Line | Current | Replace with |
|---|---|---|---|
| `python_magnetrun/MagnetRun.py` | ~105 | `MagnetData(filename="", Groups={}, Keys=[])` | `PandasMagnetData(filename="", Groups={}, Keys=[])` |
| `python_magnetrun/utils/txt2csv.py` | ~70 | `MagnetData(filename=..., Groups={}, Keys=..., Type=0, Data=raw_df)` | `PandasMagnetData(filename=..., Groups={}, Keys=..., Data=raw_df)` |
| `tests/test_magnetdata.py` | ~54 | `MagnetData("test.txt", {}, keys, 0, simple_df.copy())` | `PandasMagnetData("test.txt", {}, keys, simple_df.copy())` |
| `tests/test_magnetdata.py` | ~560 | `MagnetData("x.txt", {}, ["A", "B"], 0, pd.DataFrame(...))` | `PandasMagnetData("x.txt", {}, ["A", "B"], pd.DataFrame(...))` |
| `tests/test_magnetdata.py` | ~594 | `MagnetData("x.txt", {}, ["Field"], 0, pd.DataFrame(...))` | `PandasMagnetData("x.txt", {}, ["Field"], pd.DataFrame(...))` |

### B — `MagnetData.from*` classmethods (internal package code)

| File | Current | Replace with |
|---|---|---|
| `python_magnetrun/MagnetRun.py` | `MagnetData.fromtdms(filename)` | `load_magnetdata(filename)` |
| `python_magnetrun/MagnetRun.py` | `MagnetData.fromtxt(filename)` | `load_magnetdata(filename)` |
| `python_magnetrun/MagnetRun.py` | `MagnetData.fromcsv(filename)` | `load_magnetdata(filename)` |
| `python_magnetrun/MagnetRun.py` | `MagnetData.fromStringIO(name)` | `PandasMagnetData.fromStringIO(name)` — move the classmethod to `PandasMagnetData` (see Step 3) |
| `python_magnetrun/processing/plateaux.py` | `MagnetData.fromtxt(name)` | `load_magnetdata(name)` |

### C — `MagnetData.from*` classmethods (tests)

All `from*` calls in tests exercise file-loading behaviour, not the `MagnetData`
class itself.  Switch to `load_magnetdata()` or the concrete classmethods.

| File | Current | Replace with |
|---|---|---|
| `tests/test_magnetdata.py` (many lines) | `MagnetData.fromtxt(...)` | `load_magnetdata(...)` |
| `tests/test_magnetdata.py` (many lines) | `MagnetData.fromtdms(...)` | `load_magnetdata(...)` |
| `tests/test_magnetdata.py` (many lines) | `MagnetData.fromcsv(...)` | `load_magnetdata(...)` |
| `tests/test_magnetdata.py` (many lines) | `MagnetData.fromensight(...)` | `EnsightMagnetData.fromensight(...)` — move classmethod to `EnsightMagnetData` (see Step 3) |
| `tests/test_magnetdata.py` (many lines) | `MagnetData.fromStringIO(...)` | `PandasMagnetData.fromStringIO(...)` |
| `tests/test_file_validation.py` | `MagnetData.fromtxt(...)` | `load_magnetdata(...)` |
| `tests/test_file_validation.py` | `MagnetData.fromtdms(...)` | `load_magnetdata(...)` |
| `tests/test_file_validation.py` | `MagnetData.fromcsv(...)` | `load_magnetdata(...)` |
| `tests/test_processing.py` | `MagnetData.fromtxt(...)` | `load_magnetdata(...)` |

### D — `MagnetData` as a type annotation

All of these should become `MagnetDataBase` — the correct abstract type for
polymorphic use.

| File | Lines |
|---|---|
| `python_magnetrun/processing/correlations.py` | ~221, ~283, ~320, ~362 |
| `python_magnetrun/processing/stats.py` | ~18 |
| `python_magnetrun/processing/cli.py` | ~29 |
| `python_magnetrun/processing/trends.py` | ~139 |
| `python_magnetrun/processing/plateaux.py` | ~23, ~188 |
| `python_magnetrun/runetl.py` | function signature for `prepareData` |
| `python_magnetrun/signature.py` | check and update |
| `python_magnetrun/viewcsv.py` | check and update |
| `tests/test_magnetdata.py` | fixture return types and parameter types — change to `PandasMagnetData` for pandas fixtures, `MagnetDataBase` for polymorphic fixtures |
| `tests/test_processing.py` | fixture return type — change to `PandasMagnetData` |

### E — External files (leave unchanged)

These are outside the package boundary. Do not modify them — they document
that external consumers exist and will need a separate migration.

| File | Usage |
|---|---|
| `examples/proposal.py` | `from ..magnetdata import MagnetData` |
| `magnetcooling/examples/flow_params.py` | `MagnetData.fromtxt(file)` |
| `tests/test-fft.py` | `MagnetData` type annotation |
| `tests/test-simu.py` | `MagnetData` type annotation |

---

## Implementation steps

### Step 1 — Move orphaned classmethods to their natural home

`fromStringIO` and `fromensight` are currently only on `MagnetData`. Before
deleting the class, move them:

- Move `fromStringIO` to `PandasMagnetData` (it returns a `PandasMagnetData`)
- Move `fromensight` to `EnsightMagnetData` (it returns an `EnsightMagnetData`)

Add both to `__all__` in their respective modules, or re-export from `magnetdata.py`
so that `load_magnetdata` can also dispatch on `.ensight` files if desired.

### Step 2 — Replace direct-constructor usages (Group A)

Three test cases and two package files listed in Group A. Change `MagnetData(...)`
to `PandasMagnetData(...)` and drop the `Type` positional argument (not a parameter
of `PandasMagnetData`).

### Step 3 — Replace `from*` classmethod calls (Groups B and C)

- `MagnetData.fromtxt(f)` → `load_magnetdata(f)`
- `MagnetData.fromtdms(f)` → `load_magnetdata(f)`
- `MagnetData.fromcsv(f)` → `load_magnetdata(f)`
- `MagnetData.fromStringIO(s)` → `PandasMagnetData.fromStringIO(s)` (after Step 1)
- `MagnetData.fromensight(f)` → `EnsightMagnetData.fromensight(f)` (after Step 1)

### Step 4 — Replace type annotations (Group D)

Change `MagnetData` annotations to `MagnetDataBase`.  Update imports in each file:
remove `from .magnetdata import MagnetData` and add
`from .magnetdata_base import MagnetDataBase` (or use the re-export from
`magnetdata.py` while it still exists).

For test fixtures that specifically exercise pupitre behaviour (e.g.
`simple_magnetdata`, `txt_magnetdata`), use `PandasMagnetData` as the type;
for fixtures that are intentionally polymorphic, use `MagnetDataBase`.

### Step 5 — Delete `MagnetData` from `magnetdata.py`

Remove the class definition.  Update `__all__` to remove `"MagnetData"`.
Keep `load_magnetdata` and all the re-exports of concrete classes.

### Step 6 — Verify

```bash
source magnetrun-env/bin/activate
pytest tests/ -x -q
# Expected: all tests pass; test-fft.py and test-simu.py may be skipped or
# fail on import — those are in the external-files category and out of scope.
```

Also verify no remaining references:
```bash
grep -r "MagnetData[^B]" python_magnetrun/ tests/test_magnetdata.py \
     tests/test_file_validation.py tests/test_processing.py
# Expected: zero matches (only MagnetDataBase references remain)
```

---

## Files to modify

| File | Changes |
|---|---|
| `python_magnetrun/magnetdata.py` | Delete `MagnetData` class; update `__all__` |
| `python_magnetrun/magnetdata_pandas.py` | Add `fromStringIO` classmethod |
| `python_magnetrun/magnetdata_pandas.py` (EnsightMagnetData) | Add `fromensight` classmethod |
| `python_magnetrun/MagnetRun.py` | Groups A + B: constructor stub → `PandasMagnetData`; `from*` → `load_magnetdata` |
| `python_magnetrun/utils/txt2csv.py` | Group A: constructor → `PandasMagnetData` |
| `python_magnetrun/runetl.py` | Group D: annotation → `MagnetDataBase` |
| `python_magnetrun/processing/correlations.py` | Group D |
| `python_magnetrun/processing/stats.py` | Group D |
| `python_magnetrun/processing/cli.py` | Group D |
| `python_magnetrun/processing/trends.py` | Group D |
| `python_magnetrun/processing/plateaux.py` | Groups B + D |
| `python_magnetrun/signature.py` | Group D (check) |
| `python_magnetrun/viewcsv.py` | Group D (check) |
| `tests/test_magnetdata.py` | Groups A + C + D |
| `tests/test_file_validation.py` | Group C |
| `tests/test_processing.py` | Groups C + D |

## Files NOT to modify

- `examples/proposal.py`
- `magnetcooling/examples/flow_params.py`
- `tests/test-fft.py`
- `tests/test-simu.py`
