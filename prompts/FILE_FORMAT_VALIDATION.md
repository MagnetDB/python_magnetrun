# Plan: Add File-Format Validation Before Parsing

**Status: COMPLETE** — all steps implemented and tested.

## Summary

A new `validation.py` module provides a `FileFormatError` (subclass of `ValueError`) and
per-format validators. These validators are called at the top of each parser's entry point,
before any file I/O beyond reading a few bytes.

---

## Final State

| Format | Existence check | Extension check | Magic/structural check |
|---|---|---|---|
| `.txt` (pupitre) | `validate_txt_format` in `magnetdata_pandas.py:657` | yes | `Date`/`Time` header check |
| `.tdms` | `validate_tdms_format` in `magnetdata.py:97` | yes | `TDSm` magic bytes |
| `.csv` (generic/ensight/bprofile/feelpp) | `validate_csv_format` / `validate_file_exists` in `magnetdata_pandas.py` | — | non-empty + readable |
| RMS binary | `validate_rms_format` in `rms_reader.py:71` | — | `#` first byte |
| VProcess binary | `validate_vprocess_format` in `vprocess_reader.py:153` | — | `#` first byte |
| FEPC kHz binary | `FileFormatError` raised in `fepc_reader.py:652` | — | block-size alignment |

---

## Steps

### Step 1 — Create `python_magnetrun/utils/validation.py` ✓

Defined `FileFormatError(ValueError)` and these validators:

- `validate_file_exists(path)`
- `validate_txt_format(path)` — checks non-empty + `Date`/`Time` in header line
- `validate_tdms_format(path)` — checks magic bytes `b"TDSm"` at offset 0
- `validate_csv_format(path, required_columns=None)` — non-empty, parseable first line, optional column names
- `validate_rms_format(path)` — first byte is `b"#"`
- `validate_vprocess_format(path)` — first byte is `b"#"`
- `validate_fepc_binary_format(path, card_type)` — file size is a multiple of block size (defined; `fepc_reader.py` duplicates inline)

### Step 2 — Export from `python_magnetrun/utils/__init__.py` ✓

`FileFormatError` and all validators exported.

### Step 3 — Patch `magnetdata_pandas.py` and `magnetdata.py` ✓

| Method | Validator | Location |
|---|---|---|
| `fromtxt` | `validate_txt_format` | `magnetdata_pandas.py:657` |
| `fromtdms` | `validate_tdms_format` | `magnetdata.py:97` |
| `fromcsv` | `validate_csv_format` | `magnetdata_pandas.py:670` |
| `fromensight` | `validate_file_exists` | `magnetdata_pandas.py:721` |
| `frombprofile` | `validate_csv_format` | `magnetdata_pandas.py:740` |
| `fromfeelpp` | `validate_csv_format` | `magnetdata_pandas.py:759` |

### Step 4 — Write `tests/test_file_validation.py` ✓

Unit tests per validator (exists/missing/bad content) + integration tests for
`MagnetData` factory methods (`fromtxt`, `fromtdms`, `fromcsv`).
Integration tests depend on `tests/data/sample_pupitre.txt`.

### Step 5 — Patch `rms_reader.py` and `vprocess_reader.py` ✓

- `rms_reader.py:71` — calls `validate_rms_format`
- `vprocess_reader.py:153` — calls `validate_vprocess_format`

### Step 6 — Fix `fepc_reader.py` and `trigger_reader.py` ✓

- `fepc_reader.py:652` — bare `raise` replaced with
  `raise FileFormatError(f"{filepath}: file size {file_size} is not a multiple of block size {block_size}")`
- `trigger_reader.py:238` — silent `logger.warning` upgraded to also
  `raise FileFormatError(...)` so corrupted reads are caught early

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

## Files Changed / Created

- `python_magnetrun/utils/validation.py` *(created)*
- `python_magnetrun/utils/__init__.py`
- `python_magnetrun/magnetdata.py`
- `python_magnetrun/magnetdata_pandas.py`
- `python_magnetrun/hybrid/kHz/fepc_reader.py`
- `python_magnetrun/hybrid/trigger/trigger_reader.py`
- `python_magnetrun/hybrid/rms/rms_reader.py`
- `python_magnetrun/hybrid/vprocess/vprocess_reader.py`
- `tests/test_file_validation.py` *(created)*
