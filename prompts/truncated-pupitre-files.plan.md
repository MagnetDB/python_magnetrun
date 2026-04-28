# Plan: Resilient handling of broken pupitre files

## Context

Pupitre `.txt` files are written by the acquisition system and can be broken in
several ways: write interrupted mid-line, encoding issues (Latin-1 vs UTF-8),
or a file that contains only a header with no data rows.  The current code does
not handle these cases uniformly — some callers crash, others silently skip the
file, and a `UnicodeDecodeError` is never caught anywhere in the loading chain.

This plan is **independent** of the lazy-loading refactor.  The lazy-loading
plan (`lazy-loading-pupitre.plan.md`) must apply the same fixes to
`_ensure_data_loaded()` once that method exists.

---

## Failure modes covered

| Mode | Symptom | Current behaviour |
|---|---|---|
| Write interrupted mid-line | Last line has wrong field count | `ParserError` (ValueError) — caught in `select_files`, NOT in `load_files_data` |
| Write interrupted between lines | Last complete line valid, partial line after | `ParserError` in some pandas versions, or silently loads truncated data |
| Non-UTF-8 characters (Latin-1/CP1252) | French accents, unit strings | `UnicodeDecodeError` — caught nowhere |
| Header only, no data rows | Empty DataFrame after load | Loads silently; `getDuration()` returns 0; downstream crashes vary |
| Empty file | 0 bytes | Caught by `validate_txt_format` ✓ |

---

## Changes

### Step 1 — Add truncation check to `validation.py`

Add `check_pupitre_truncation(path, keys)` as a **warning-only** function (does
not raise).  Raising would prevent loading any data from a partially-written
file; logging a warning keeps the file usable.

```python
def check_pupitre_truncation(path: str, keys: list[str]) -> bool:
    """Return True and log a WARNING if the file appears truncated.

    Checks two conditions:
    - the file does not end with '\\n' (last line incomplete), or
    - the last non-empty line has fewer fields than the header.
    """
```

Call site: inside `PandasMagnetData.fromtxt` (and later `_ensure_data_loaded`)
**after** `validate_txt_format` and **before** `pd.read_csv`.

### Step 2 — Encoding fallback in `fromtxt`

`open(name)` uses the system locale (usually UTF-8).  Older pupitre files may
be Latin-1.  Replace the bare `open` with a two-attempt helper:

```python
def _open_text_with_fallback(path: str):
    """Try UTF-8 first, fall back to Latin-1."""
    try:
        f = open(path, encoding="utf-8")
        f.read(1)          # force decode of first byte
        f.seek(0)
        return f
    except UnicodeDecodeError:
        return open(path, encoding="latin-1", errors="replace")
```

Apply to `fromtxt` and to the `nrows=0` header read in `_pupitre_end_from_last_line`
(the binary seek already passes `errors="replace"` so that path is fine).

### Step 3 — `on_bad_lines="warn"` in `fromtxt`

Change the `pd.read_csv` call in `PandasMagnetData.fromtxt`:

```python
# before
Data = pd.read_csv(f, sep=r"\s+", engine="python", skiprows=1)

# after
Data = pd.read_csv(
    f, sep=r"\s+", engine="python", skiprows=1, on_bad_lines="warn"
)
```

This skips malformed rows (e.g. a truncated last line) and emits a pandas
warning instead of raising `ParserError`.  The resulting DataFrame contains all
rows up to the first bad line, which is the best possible outcome for a
truncated file.

Apply the same change to `fromcsv` for consistency.

### Step 4 — Add `UnicodeDecodeError` to caller catch blocks

Two locations in `loaders.py` catch file-load exceptions but miss
`UnicodeDecodeError`:

| Location | Line | Current catch | Fix |
|---|---|---|---|
| `select_files` inner loop | ~714 | `(OSError, ValueError, RuntimeError)` | add `UnicodeDecodeError` |
| `load_files_data` inner loop | ~966 | `(OSError, ValueError, RuntimeError, KeyError)` | add `UnicodeDecodeError` |

### Step 5 — Guard against header-only files in `PandasMagnetData`

After `pd.read_csv` in `fromtxt`, check for an empty DataFrame and raise a
clear `FileFormatError` rather than propagating a cryptic downstream crash:

```python
if Data.empty:
    raise FileFormatError(f"{name}: no data rows found (header-only file)")
```

Callers that want to tolerate empty files can catch `FileFormatError`.

---

## Files affected

| File | Change |
|---|---|
| `python_magnetrun/utils/validation.py` | add `check_pupitre_truncation()` |
| `python_magnetrun/magnetdata_pandas.py` | `fromtxt`, `fromcsv`: encoding fallback, `on_bad_lines`, empty-data guard |
| `python_magnetrun/analysis/loaders.py` | `select_files`, `load_files_data`: add `UnicodeDecodeError` to catch |

---

## Tests to add

All in `tests/test_magnetdata_pandas.py` (or a new `tests/test_truncated_pupitre.py`):

- `test_fromtxt_truncated_midline`: file with a complete header + N valid rows +
  one row missing the last 3 fields → loads N rows, emits warning, no exception.
- `test_fromtxt_truncated_between_lines`: file ending immediately after a valid
  row's newline → loads normally (not actually truncated at the data level).
- `test_fromtxt_latin1_encoding`: file with a Latin-1 `é` character in a column
  value → loads without `UnicodeDecodeError`.
- `test_fromtxt_header_only`: file with comment line + column header + no data
  rows → raises `FileFormatError`.
- `test_pupitre_end_from_last_line_truncated`: `_pupitre_end_from_last_line` on a
  file whose last line has too few fields → returns `""`, no exception.
- `test_select_files_skips_unicode_error`: `select_files` with a Latin-1 file
  in the list → skips cleanly, returns the other valid files.

---

## Interaction with the lazy-loading plan

When `PandasMagnetData._ensure_data_loaded()` is implemented, it must apply
steps 2 and 3 (encoding fallback + `on_bad_lines="warn"`) identically to the
full `pd.read_csv` call inside that method.  Step 5 (empty-data guard) applies
there too.  Steps 1 and 4 are caller-level and do not change.
