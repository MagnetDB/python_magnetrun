# Plan: Add File-Format Validation Before Parsing

## Summary

A new `validation.py` module provides a `FileFormatError` (subclass of `ValueError`) and
per-format validators. These validators are called at the top of each parser's entry point,
before any file I/O beyond reading a few bytes.

---

## Current State

| Format | Existence check | Extension check | Magic/structural check |
|---|---|---|---|
| `.txt` (pupitre) | **missing** | yes | none |
| `.tdms` | yes | yes | none |
| `.csv` (generic/ensight/bprofile/feelpp) | **missing** | **missing** | none |
| RMS binary | **missing** | **missing** | none |
| VProcess binary | **missing** | **missing** | none |
| FEPC kHz binary | partial (alignment, but bare `raise`) | **missing** | — |

---

## Steps

### Step 1 — Create `python_magnetrun/utils/validation.py`

Define `FileFormatError(ValueError)` and these validators:

- `validate_file_exists(path)` — raises `FileNotFoundError` if missing
- `validate_txt_format(path)` — checks non-empty + `Date`/`Time` in header line
- `validate_tdms_format(path)` — checks magic bytes `b"TDSm"` at offset 0
- `validate_csv_format(path, required_columns=None)` — non-empty, parseable first line, optional column names
- `validate_rms_format(path)` — first byte is `b"#"`
- `validate_vprocess_format(path)` — first byte is `b"#"`
- `validate_fepc_binary_format(path, card_type)` — file size is a multiple of block size

### Step 2 — Export from `python_magnetrun/utils/__init__.py`

Add `from .validation import FileFormatError`.

### Step 3 — Patch `python_magnetrun/magnetdata.py` (highest leverage)

At the top of each `from*` classmethod, call the matching validator before any `open()` call:

| Method | Validator |
|---|---|
| `fromtxt` | `validate_txt_format` |
| `fromtdms` | `validate_tdms_format` (after existing existence/extension checks) |
| `fromcsv` | `validate_file_exists` + `validate_csv_format` |
| `fromensight` | `validate_file_exists` |
| `frombprofile` | `validate_file_exists` + `validate_csv_format` |
| `fromfeelpp` | `validate_file_exists` + `validate_csv_format` |

### Step 4 — Write `tests/test_file_validation.py`

Tests for each validator (`pytest`, `tmp_path`, `pytest.raises`) + integration tests on
`MagnetData.fromtxt` / `fromtdms` with bad files.

### Step 5 — Patch `rms_reader.py` and `vprocess_reader.py`

Call `validate_rms_format` / `validate_vprocess_format` at the start of `parse_header`.

### Step 6 — Fix `fepc_reader.py` and `trigger_reader.py`

- Replace bare `raise` at line 614 of `fepc_reader.py` with
  `raise FileFormatError(f"{filepath}: file size {file_size} is not a multiple of block size {block_size}")`.
- Upgrade the silent `logger.warning` in `trigger_reader.py` (`read_trigger_file`, line 231-234)
  to also raise a `FileFormatError` so corrupted reads are caught early.

---

## Error Message Format

```
FileFormatError: /path/to/bad.tdms: expected TDMS magic b'TDSm' at offset 0, got b'\x89PNG'
FileFormatError: /path/to/bad.txt: missing required header columns ['Date', 'Time'] in second line
FileFormatError: /path/to/bad.rms: expected ASCII header marker '#' at byte 0, got 0x54
```

All validators raise `FileFormatError` (a `ValueError` subclass) so existing
`except (OSError, ValueError, RuntimeError)` catch blocks in callers continue to work
without change.

---

## Critical Files

- `python_magnetrun/magnetdata.py`
- `python_magnetrun/utils/validation.py` *(to create)*
- `python_magnetrun/utils/__init__.py`
- `python_magnetrun/hybrid/kHz/fepc_reader.py`
- `python_magnetrun/hybrid/trigger/trigger_reader.py`
- `python_magnetrun/hybrid/rms/rms_reader.py`
- `python_magnetrun/hybrid/vprocess/vprocess_reader.py`
- `tests/test_file_validation.py` *(to create)*
