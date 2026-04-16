# Replace `MagnetData` shim with a standalone factory function

## Context

`python_magnetrun/magnetdata.py` contains a class `MagnetData(PandasMagnetData)` that
acts as both a backward-compatible constructor shim AND a factory. This is architecturally
wrong: its classmethods (`fromtdms`, `fromtxt`, …) return `TdmsMagnetData` or
`PandasMagnetData` instances — never `MagnetData` itself — and the direct constructor
bypasses `PandasMagnetData.__init__` by calling `MagnetDataBase.__init__` directly.
This makes `isinstance` checks unreliable and the class hierarchy misleading.

The concrete implementation classes already exist and are complete:
- `PandasMagnetData` in `python_magnetrun/magnetdata_pandas.py`
- `TdmsMagnetData` in `python_magnetrun/magnetdata_tdms.py`
- `EnsightMagnetData`, `BProfileMagnetData`, `FeelppMagnetData` in `magnetdata_pandas.py`

The abstract base is `MagnetDataBase` in `python_magnetrun/magnetdata_base.py`.

## Goal

Replace the `MagnetData` class factory pattern with a standalone
`load_magnetdata(filename, ...)` function in `magnetdata.py`, while keeping
`MagnetData` as a deprecated backward-compatible re-export so external callers
do not break immediately.

## What to do

### Step 1 — Add `load_magnetdata()` to `magnetdata.py`

Add a module-level factory function that dispatches on file extension:

```python
def load_magnetdata(
    filename: str,
    defs_file: str | None = None,
    sep: str = r"\s+",
    skiprows: int = 1,
) -> MagnetDataBase:
    """Load a magnet data file and return the appropriate MagnetDataBase subclass.

    Dispatches on file extension:
    - ``.tdms``  → :class:`TdmsMagnetData`
    - ``.txt``   → :class:`PandasMagnetData`
    - ``.csv``   → :class:`PandasMagnetData`

    :param filename: path to the data file
    :param defs_file: optional path to a field definitions JSON file
    :return: the loaded data object
    :raises ValueError: if the file extension is not recognised
    """
```

Inside, call the appropriate existing classmethod:
- `.tdms` → `MagnetData.fromtdms(filename, defs_file=defs_file)`
- `.txt`  → `MagnetData.fromtxt(filename, defs_file=defs_file or "pupitre-defs.json")`
- `.csv`  → `MagnetData.fromcsv(filename, defs_file=defs_file)`
- unknown → raise `ValueError(f"Unsupported file extension: {ext!r}")`

### Step 2 — Update internal callers to use concrete classes directly

There are two places inside the package that use the `MagnetData(...)` direct constructor
instead of a `from*` classmethod:

**`python_magnetrun/MagnetRun.py` line ~105** (`fromStringIO`):
```python
data = MagnetData(filename="", Groups={}, Keys=[])   # empty stub
```
Replace with:
```python
data = PandasMagnetData(filename="", Groups={}, Keys=[])
```
Also update the import at the top of `MagnetRun.py`:
- Add `from .magnetdata_pandas import PandasMagnetData` (or import from `magnetdata`
  which already re-exports it).

**`python_magnetrun/utils/txt2csv.py` line ~70**:
```python
data = MagnetData(filename=input_files[0], Groups={}, Keys=raw_df.columns.tolist(),
                  Type=0, Data=raw_df)
```
Replace with:
```python
data = PandasMagnetData(filename=input_files[0], Groups={}, Keys=raw_df.columns.tolist(),
                        Data=raw_df)
```
The stale comment above that line (`# Wrap in MagnetData and apply prepareData_legacy`)
should be updated to reflect the current reality (no legacy path).

### Step 3 — Update internal callers that use `MagnetData.from*` classmethods

The following internal files call `MagnetData.from*` and should be switched to call
the concrete classmethods or `load_magnetdata` directly:

| File | Current call | Replace with |
|---|---|---|
| `python_magnetrun/MagnetRun.py` | `MagnetData.fromtdms(...)` | `MagnetData.fromtdms(...)` — leave (re-exported) OR switch to `load_magnetdata(filename)` |
| `python_magnetrun/MagnetRun.py` | `MagnetData.fromtxt(...)` | same |
| `python_magnetrun/MagnetRun.py` | `MagnetData.fromcsv(...)` | same |
| `python_magnetrun/MagnetRun.py` | `MagnetData.fromStringIO(...)` | keep (no file extension to dispatch on) |
| `python_magnetrun/processing/plateaux.py` | `MagnetData.fromtxt(name)` | `load_magnetdata(name)` |
| `python_magnetrun/processing/cli.py` | `MagnetData` import | update if needed |

For files in `processing/`, `signature.py`, `viewcsv.py` — check whether they use
`MagnetData.from*` (safe to leave via re-export) or the direct constructor
`MagnetData(...)` (must be changed to `PandasMagnetData(...)`).

### Step 4 — Keep `MagnetData` as a deprecated shim

Do **not** delete the `MagnetData` class. Keep it in `magnetdata.py` with a deprecation
notice in the docstring:

```python
class MagnetData(PandasMagnetData):
    """Backward-compatible shim — use :func:`load_magnetdata` for new code.

    .. deprecated::
        Direct construction ``MagnetData(filename, Groups, Keys, Type, Data)``
        and the ``from*`` classmethods are retained for backward compatibility.
        New code should use :func:`load_magnetdata` or the concrete subclasses
        (:class:`PandasMagnetData`, :class:`TdmsMagnetData`) directly.
    """
```

### Step 5 — Update `__all__` in `magnetdata.py`

Add `load_magnetdata` to `__all__`:

```python
__all__ = [
    "MagnetData",
    "MagnetDataBase",
    "DataType",
    "PandasMagnetData",
    "EnsightMagnetData",
    "BProfileMagnetData",
    "FeelppMagnetData",
    "TdmsMagnetData",
    "FileFormatError",
    "load_magnetdata",   # ← add
]
```

## What NOT to change

- **Tests** (`tests/test_magnetdata.py`, `tests/test_file_validation.py`, etc.) use
  `MagnetData.fromtxt(...)`, `MagnetData.fromtdms(...)` etc. extensively. Leave them
  as-is; they exercise the shim and verify backward compatibility.
- **`magnetcooling/examples/flow_params.py`** — external consumer, leave it.
- **`examples/proposal.py`** — example file, leave it.

## Verification

After the changes:

1. Run the existing test suite — all tests must pass without modification:
   ```
   source magnetrun-env/bin/activate
   pytest tests/ -x -q
   ```

2. Confirm `isinstance(load_magnetdata("file.txt"), PandasMagnetData)` is `True`.
3. Confirm `isinstance(load_magnetdata("file.tdms"), TdmsMagnetData)` is `True`.
4. Confirm `MagnetData.fromtxt(...)` still works (backward compat).
5. Confirm no remaining uses of `MagnetData(filename, ..., Type=N, ...)` direct
   constructor exist inside `python_magnetrun/` (grep for `MagnetData(`).

## Files to modify

- `python_magnetrun/magnetdata.py` — add `load_magnetdata()`, update docstring, update `__all__`
- `python_magnetrun/MagnetRun.py` — replace `MagnetData(filename="", ...)` stub with `PandasMagnetData`
- `python_magnetrun/utils/txt2csv.py` — replace `MagnetData(filename=..., Type=0, ...)` with `PandasMagnetData`

## Files to check but likely leave unchanged

- `python_magnetrun/runetl.py` — imports `MagnetData` for type annotation only; confirm
  and switch annotation to `MagnetDataBase` if so
- `python_magnetrun/processing/plateaux.py` — uses `MagnetData.fromtxt`; optionally
  switch to `load_magnetdata`
- `python_magnetrun/signature.py`, `viewcsv.py`, `processing/cli.py`,
  `processing/stats.py`, `processing/trends.py`, `processing/correlations.py` —
  check for direct constructor use; if only `from*` classmethods, leave unchanged
