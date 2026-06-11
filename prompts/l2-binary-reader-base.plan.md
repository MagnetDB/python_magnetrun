# L2 — Extract `_BinaryFileReaderBase`

*Created: 2026-06-11*

Extracts the shared binary-file-reading logic from `RMSFileReader` and
`VProcessFileReader` into an abstract base class, and merges the nearly-identical
`RMSVariable` / `VProcessVariable` dataclasses into a single `ChannelVariable`.

**Prerequisite for:** Stream 4.6 (clean plotting integration) and eventual `TriggerFileReader` unification.

---

## Motivation

`hybrid/rms/rms_reader.py` (425 lines) and `hybrid/vprocess/vprocess_reader.py`
(524 lines) share ~85% of their code. Every bug fix or feature (e.g. endian
handling, timestamp precision) must be applied twice. The remaining 15% consists
of well-understood, isolated differences that map cleanly to abstract hooks.

---

## Concrete differences between the two readers

| Aspect | RMS | VProcess |
|--------|-----|---------|
| Header encoding | `"US-ASCII"` | `"utf-8"` + `errors="ignore"` |
| `parse_header()` reset | no reset | resets `self.variables = []`, `self.metadata = {}` |
| `parse_header()` extra dispatch | `"# processed on"` → `_parse_processed_info()` | `"# vprocess data file"` → `_parse_format()` |
| `_parse_format()` regex | `r"# format\s*=\s*(.+)"` | `r"# vprocess data file - (.+)"` |
| `_parse_variables()` `is_analog` | `var_type == "float32"` | `"float" in var_type.lower()` |
| `_parse_variables()` missing min/max | `float(v) if "min" in props else None` | `float(v) if v != "_" else None` |
| `read()` guard | `if not self.variables: parse_header()` | always calls `parse_header()` |
| Timestamp conversion | `pd.to_datetime(ts_list, unit="s", utc=True)` | `[datetime.fromtimestamp(t, tz=UTC) for t]` → `pd.DatetimeIndex` |
| `get_variable_info()` columns | includes `byte_size` | no `byte_size` |
| `print_summary()` output | `print()` calls | `logger.info()` calls |
| `_parse_processed_info()` | ✅ RMS-only extra method | ✗ absent |

---

## Proposed design

### `ChannelVariable` (replaces both variable classes)

```python
# hybrid/_binary_reader_base.py

@dataclass
class ChannelVariable:
    name: str
    var_type: str
    unit: str | None = None
    min_val: float | None = None
    max_val: float | None = None
    display_format: str | None = None

    def __post_init__(self) -> None:
        self.is_analog: bool = "float" in self.var_type.lower()
        self.byte_size: int = FLOAT32_SIZE if self.is_analog else DIGITAL_SIZE

    def __repr__(self) -> str:
        return f"ChannelVariable({self.name}, {self.var_type}, {self.unit})"
```

Using `"float" in var_type.lower()` is a strict superset of the RMS form
(`var_type == "float32"`) and correct for all currently observed type strings
(`"float32"`, `"bit"`, `"dig"`).

### `_BinaryFileReaderBase`

```python
class _BinaryFileReaderBase(ABC):
    _encoding: str = "US-ASCII"   # override in subclass

    def __init__(self, filepath: str, endian: str = "big") -> None:
        self.filepath = Path(filepath)
        self.header_lines: list[str] = []
        self.variables: list[ChannelVariable] = []
        self.metadata: dict[str, Any] = {}
        self.data: pd.DataFrame | None = None
        self.endian = ">" if endian == "big" else "<"

    # ── shared header parsing ──────────────────────────────────────────────
    def parse_header(self) -> None:
        self.variables = []
        self.metadata = {}
        self._validate_format()
        with open(self.filepath, "rb") as f:
            lines = []
            while True:
                raw = f.readline()
                if not raw.startswith(b"#"):
                    break
                lines.append(raw.decode(self._encoding, errors="ignore").strip())
        self.header_lines = lines
        for line in self.header_lines:
            if line.startswith("# variables"):
                self._parse_variables(line)
            elif line.startswith("# windows"):
                self._parse_windows(line)
            elif line.startswith("# frequency"):
                self._parse_frequency(line)
            elif line.startswith("# data-helper"):
                self._parse_data_helper(line)
            else:
                self._parse_extra_header(line)   # subclass hook

    @abstractmethod
    def _validate_format(self) -> None:
        """Call the format-specific validator (validate_rms_format, etc.)."""

    @abstractmethod
    def _parse_variables(self, line: str) -> None:
        """Parse `# variables = …` — differs in is_analog check and missing-value sentinel."""

    def _parse_extra_header(self, line: str) -> None:
        """Hook for format-specific header lines. Default: no-op."""

    # _parse_windows, _parse_frequency, _parse_data_helper — shared, go in base

    # ── shared binary reading ──────────────────────────────────────────────
    def read_binary_data(self) -> pd.DataFrame:
        """Shared binary loop — only timestamp conversion is delegated."""
        ...   # identical loop; calls self._make_timestamps(raw_ts_list)

    @abstractmethod
    def _make_timestamps(self, raw: list[float]) -> pd.DatetimeIndex:
        """Convert raw Unix-epoch floats to a DatetimeIndex."""

    # ── shared query methods ───────────────────────────────────────────────
    def read(self) -> pd.DataFrame: ...
    def get_variable_info(self) -> pd.DataFrame: ...  # base version; subclass can extend
    def get_metadata(self) -> dict[str, Any]: ...
    def print_summary(self) -> None: ...
```

### Subclass implementations

**`RMSFileReader`** inherits `_BinaryFileReaderBase`:
- `_encoding = "US-ASCII"`
- `_validate_format()` → calls `validate_rms_format()`
- `_parse_variables()` → no `"_"` sentinel, `"float32"` exact check acceptable (handled by `ChannelVariable.__post_init__`)
- `_parse_extra_header()` → handles `"# format"` and `"# processed on"` lines
- `_make_timestamps()` → `pd.to_datetime(raw, unit="s", utc=True)`
- `get_variable_info()` → adds `byte_size` column

**`VProcessFileReader`** inherits `_BinaryFileReaderBase`:
- `_encoding = "utf-8"`
- `_validate_format()` → calls `validate_vprocess_format()`
- `_parse_variables()` → `"_"` sentinel for missing min/max
- `_parse_extra_header()` → handles `"# vprocess data file"` line
- `_make_timestamps()` → `pd.DatetimeIndex([datetime.fromtimestamp(t, tz=UTC) for t in raw])`

---

## Implementation plan

```
Step 1 — Create `hybrid/_binary_reader_base.py`
  · Define FLOAT32_SIZE, DIGITAL_SIZE, TIMESTAMP_SIZE, DEFAULT_SAMPLE_WIDTH,
    DEFAULT_DATA_OFFSET constants (move from rms_reader.py)
  · Define `ChannelVariable` dataclass
  · Define `_BinaryFileReaderBase` with all shared methods implemented
  · Abstract: _validate_format, _parse_variables, _make_timestamps
  · Hook: _parse_extra_header (default no-op)
  → verify: module imports cleanly

Step 2 — Update `hybrid/rms/rms_reader.py`
  · Remove `RMSVariable`; import `ChannelVariable` from `._binary_reader_base`
  · Keep `RMSVariable = ChannelVariable` re-export alias for backward compat
  · `RMSFileReader` inherits `_BinaryFileReaderBase`
  · Move only the RMS-specific overrides into the class body (5 methods)
  · `get_variable_info()` extends base result with `byte_size` column
  · Remove duplicated constants (now in base module)
  · Convenience functions (`read_rms_file`, `get_rms_info`) unchanged
  → verify: `tests/test_rms_reader.py` passes (or existing rms tests)

Step 3 — Update `hybrid/vprocess/vprocess_reader.py`
  · Remove `VProcessVariable`; import `ChannelVariable`
  · Keep `VProcessVariable = ChannelVariable` alias
  · `VProcessFileReader` inherits `_BinaryFileReaderBase`
  · Override: `_encoding`, `_validate_format`, `_parse_variables`,
    `_parse_extra_header`, `_make_timestamps`
  · Convenience functions unchanged; `__all__` updated if needed
  → verify: existing vprocess tests pass

Step 4 — Verify `HybridData` is unaffected
  · `hybrid_data.py` imports `RMSFileReader`/`VProcessFileReader` by name —
    no changes needed since class names are preserved
  → verify: full test suite passes (1093+)

Step 5 — Add tests
  · `tests/readers/test_binary_reader_base.py`
  · Test `ChannelVariable` post-init for float32, bit, dig types
  · Parametrize `_make_timestamps` round-trip for both subclasses
  · Smoke test: construct each reader, call `parse_header()` on a minimal
    header fixture (no real binary data needed for unit tests)
  → verify: new tests green
```

---

## Files affected

| File | Action |
|------|--------|
| `python_magnetrun/hybrid/_binary_reader_base.py` | **Create** |
| `python_magnetrun/hybrid/rms/rms_reader.py` | **Edit** |
| `python_magnetrun/hybrid/vprocess/vprocess_reader.py` | **Edit** |
| `tests/readers/test_binary_reader_base.py` | **Create** |

`hybrid_data.py`, `hybrid/plotting.py`, and all callers of `RMSFileReader` /
`VProcessFileReader` are **unaffected** — class names are preserved.

---

## Risk assessment

| Risk | Mitigation |
|------|-----------|
| `ChannelVariable` `is_analog` change (broader check) | Both `"float32"` and `"dig"` / `"bit"` covered; add a parametrized test |
| `_parse_windows` missing-millisecond fallback (VProcess only) | Move entire VProcess version to base as the default (it's strictly safer) |
| `get_variable_info()` column difference | RMS subclass adds `byte_size`; VProcess leaves it absent — no existing callers inspect `byte_size` on VProcess output |
| `print_summary()` output style difference | Normalize to `logger.info()` in base; `RMSFileReader.print_summary()` currently uses `print()` — acceptable breaking change (it's a debug helper) |

**Overall risk: Medium.** No public API changes; `HybridData` callers are insulated.

---

## Effort

~1 day. Sequencing: do after M3, before Stream 4.6 if you want the cleanest
integration — but 4.6 plotting can proceed without it.
