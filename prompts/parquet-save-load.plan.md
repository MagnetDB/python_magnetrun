# Parquet save/load with rich metadata — design plan

*Created: 2026-04-25 — captures the design discussion before implementation begins*

## Context

`python_magnetrun` data containers (`PandasMagnetData`, `TdmsMagnetData`, future
`HybridData`) need a persistent format that survives the round-trip from raw
acquisition file → ETL → S3 storage (RustFS) → downstream analysis. CSV, the
current `saveData` target, drops type information, unit metadata, group
structure, and timing anchors. Parquet is the right target: columnar, compressed,
typed, and supports both file-level and per-column metadata via PyArrow.

The `rustfs/magnetfs` package was a demonstrator. The intended workflow is:

```
raw file (.txt / .tdms / hybrid)
        ↓
load_magnetdata() / HybridRun.fromdir()
        ↓
ETL via prepareData(), addData(), Units(), analysis pipelines
        ↓
saveParquet() — bytes
        ↓
S3 upload (RustFS or other) — orthogonal layer
```

The goal is for the Parquet artifact to be *self-describing enough* that it can
be loaded back into a working `MagnetDataBase` instance without the original
raw file or defs JSON.

---

## Design decisions

The following decisions were made during discussion. Each is paired with the
rationale so future readers don't need to rediscover the reasoning.

### D1 — `pint.Unit` is stored as a string in Parquet metadata

`pint.Unit` cannot be embedded in Parquet metadata (which is `bytes`-only) and
is not self-contained: a `Unit` references its `UnitRegistry`, and registries
don't round-trip through serialization.

**Convention:** store `str(pint_unit)` (pint's long form, e.g. `"tesla"`,
`"meter ** 3 / second"`) as UTF-8. Empty string represents `None`. Never store
the pretty-printed `f"{unit:~P}"` form — it's for display, not parsing.

**Load:** call `_make_ureg()` fresh inside `loadParquet`, then
`ureg.parse_expression(s)`, taking `.units` if the result is a `Quantity`.
Calling `_make_ureg()` per-load guarantees project-specific units (`percent`,
`ppm`, `var`) are registered regardless of what the rest of the process did.

**Known limitation:** `Unit` identity is per-registry. Cross-file pint
arithmetic between two separately-loaded Parquet files will fail unless callers
adopt `pint.set_application_registry(ureg)`. Flag, don't fix speculatively.

### D2 — Add `category` to `FieldMeta`, do **not** add `Groups` to `PandasMagnetData`

The asymmetry between TDMS (`Groups[group][channel]` carrying `wf_start_time`,
`wf_increment`) and pandas (no Groups) is deliberate — TDMS Groups carries
*timing structure*, not just categorization. Adding a degenerate Groups dict
to `PandasMagnetData` would unify the name without unifying the contract.

The thing actually wanted for the Parquet saver is a per-channel category tag.
That's a per-field property, not a structural one. Extend `FieldMeta`:

```python
class FieldMeta(NamedTuple):
    symbol: str
    unit: Any                  # pint.Unit | None
    label: str
    description: str
    category: str = ""         # NEW — TDMS group name, or pupitre bucket
```

For TDMS the category is `key.split("/")[0]`. For pupitre it comes from JSON
defs (`"Currents"`, `"Voltages"`, `"Flows"`, `"Temperatures"`, `"Pressures"`,
`"Field"`). For HybridData it's the prefix (`"kHz"`, `"rms"`, `"trigger"`).

Round-trip on load: bucket fields by category to rebuild `Groups` for TDMS;
ignore for pandas.

**Prerequisite work:** `pupitre-defs.json`, `pigbrother-defs.json`, and
`hybrid-defs.json` need a `"category"` field added. Default to `""` for forward
compatibility on older Parquet files.

### D3 — Drop the `timestamp` column from saved data; keep only `t`

`timestamp` is fully reconstructible from `start_timestamp + t * 1s`, and for
high-rate data (pigbrother kHz, hybrid kHz) the 8 bytes/row of `datetime64[ns]`
is non-trivial.

**Implication:** add a lazy `getTimestamp()` (or `materializeTimestamp()`)
method that computes the column on demand from `t` plus file-level
`start_timestamp`. `extractTimeData` and cross-source alignment call it lazily;
everything else operates on `t`.

### D4 — Sub-second precision in `start_timestamp` requires care

ISO-8601 strings lose nanosecond precision unless explicitly formatted. TDMS
`wf_start_time` is `numpy.datetime64[ns]`. For pupitre at 1 Hz this doesn't
matter; for pigbrother and hybrid kHz streams it does.

**Convention:** store two keys to be unambiguous:

```
magnetrun.start_timestamp_seconds : "1715271243"        # epoch seconds
magnetrun.start_timestamp_nanos   : "123456789"         # remainder
```

Same for `end_timestamp`. Document that the timestamp is naive UTC by storing:

```
magnetrun.timestamp_tz : "UTC-naive"
```

### D5 — Properties are caller-populated, not computed at save time

`saveParquet` should be cheap and have no surprise side effects. Plateau
detection needs thresholds; signature extraction needs a key and a threshold;
energy needs an integration window. The saver doesn't know these.

**API:** add `self.properties: dict` on `MagnetDataBase`. Analysis code attaches
results via `setProperty(name, value)`. `saveParquet` serializes whatever is
there, no computation.

```python
mdata.setProperty("plateaux", nplateaus(mdata, ...))
mdata.setProperty("signature", asdict(Signature.from_mdata(mdata, ...)))
mdata.saveParquet("run.parquet")
```

This matches how `field_meta` already works (populated by `Units()` / `addData`,
consumed at use time).

### D6 — TDMS / HybridRun multi-rate: save one group/stream per file initially

A single Parquet file assumes one time grid. Pupitre, single TDMS group, and a
single HybridRun stream all fit. Multi-group TDMS and full HybridRun do not.

**Initial implementation:** `saveParquet(path, group=None)`. For pupitre, `group`
must be `None`. For TDMS, `group` must be specified. For HybridRun, `group` is
the stream identifier.

**Future:** directory layout — `run.parquet/` containing one Parquet per
group plus a top-level `_manifest.json`. Defer until the single-group case is
proven.

### D7 — S3 layer is orthogonal to `python_magnetrun`

`saveParquet` accepts a path *or* a writable binary file-like. No boto3 import
in `python_magnetrun`. The RustFS / S3 wrapper composes:

```python
buf = BytesIO()
mdata.saveParquet(buf, group=group)
buf.seek(0)
s3.upload_fileobj(buf, bucket, key)
```

Keep `pyarrow.fs.S3FileSystem` off the table for now — it complicates RustFS
endpoint/credential handling and ties pyarrow to the storage backend.

### D8 — `pyarrow` for the saver, not `polars`

`rustfs/magnetfs/conversion.py` uses polars, which has weaker control over
Parquet metadata. The core saver in `python_magnetrun` uses pyarrow
(`pa.Table.from_pandas(df, preserve_index=False)` then
`table.replace_schema_metadata(...)`). Polars stays fine for the RustFS upload
path if it's already there, but new code uses pyarrow.

---

## Metadata schema — version 1

### File-level (`schema.metadata`)

```
magnetrun.schema_version           : "1"
magnetrun.source_file              : "M9_2024.05.09---16_34_03.txt"
magnetrun.source_type              : "pupitre" | "pigbrother" | "ensight" | "hybrid"
magnetrun.housing                  : "M9"
magnetrun.site                     : "M9"
magnetrun.group                    : ""                          # empty for pupitre, group name for TDMS
magnetrun.start_timestamp_seconds  : "1715271243"
magnetrun.start_timestamp_nanos    : "123456789"
magnetrun.end_timestamp_seconds    : "..."
magnetrun.end_timestamp_nanos      : "..."
magnetrun.timestamp_tz             : "UTC-naive"
magnetrun.defs_file                : "pupitre-defs.json"         # if any, for traceability only
magnetrun.properties               : { ...JSON... }              # see below
```

### Field-level (per `pa.field` in the schema)

```
magnetrun.symbol      : "B"
magnetrun.unit        : "tesla"            # pint-parseable; "" → None
magnetrun.label       : "B"                # may be ""
magnetrun.description : "..."              # may be ""
magnetrun.category    : "Field"            # TDMS group, pupitre bucket, hybrid prefix
```

### Properties blob (file-level, JSON-encoded)

```json
{
  "B": {"min": 0.0, "max": 24.1, "unit": "tesla"},
  "energy": {"value": 3.42e8, "unit": "joule"},
  "plateaux": [
    {"key": "Field", "start": 1234.5, "end": 1300.0,
     "value": 24.0, "duration": 65.5, "unit": "tesla"}
  ],
  "signature": {
    "key": "Référence_GR1",
    "regimes": ["U", "P", "D"],
    "times":   [0.0, 12.3, 78.0, 90.5],
    "values":  [0.0, 31200.0, 31200.0, 0.0],
    "threshold": 1e-2
  }
}
```

These map directly onto `nplateaus()` output and `Signature.from_mdata()` —
serialization is `dataclasses.asdict()` / `dict(...)`.

### Forward compatibility

- Required keys: `schema_version`, `source_type`, `start_timestamp_*`,
  `timestamp_tz`. Loaders raise on missing.
- Optional keys: everything else. Loaders use `.get(..., b"")` and tolerate
  absence.
- Version bump: increment `schema_version` whenever a new top-level key has
  non-default semantics. Loaders warn (not fail) on unknown versions.

---

## API

### `MagnetDataBase` additions

```python
# magnetdata_base.py
from typing import IO
import os

@abstractmethod
def saveParquet(
    self,
    target: str | os.PathLike | IO[bytes],
    group: str | None = None,
) -> None:
    """Write data + metadata to Parquet.

    For TDMS data, *group* must be specified (one group per file). For
    pandas data, *group* must be None. *target* may be a filesystem path
    or an open binary stream.
    """

@classmethod
@abstractmethod
def loadParquet(
    cls,
    source: str | os.PathLike | IO[bytes],
) -> "MagnetDataBase":
    """Reconstruct a data container from a Parquet file written by saveParquet."""

def setProperty(self, name: str, value: Any) -> None:
    """Attach a JSON-serializable property for later inclusion in saveParquet."""
    self.properties[name] = value

def getProperty(self, name: str, default: Any = None) -> Any:
    return self.properties.get(name, default)
```

`MagnetDataBase.__init__` gains `self.properties: dict = {}`.

### Top-level factory

```python
# python_magnetrun/io/parquet.py  (new module)
def load_magnetrun_parquet(source) -> MagnetDataBase:
    """Read magnetrun.source_type from metadata and dispatch to the right subclass."""

# MagnetRun.py
@classmethod
def fromparquet(cls, source) -> "MagnetRun":
    """Restore Housing/Site from file-level metadata, wrap MagnetDataBase."""
```

### Unit serialization helpers

```python
# python_magnetrun/io/parquet.py

def _serialize_unit(unit) -> bytes:
    return (str(unit) if unit is not None else "").encode("utf-8")

def _deserialize_unit(unit_bytes: bytes, ureg) -> "pint.Unit | None":
    s = unit_bytes.decode("utf-8")
    if not s:
        return None
    parsed = ureg.parse_expression(s)
    return parsed.units if hasattr(parsed, "units") else parsed

def _serialize_field_meta(meta: FieldMeta) -> dict[bytes, bytes]:
    return {
        b"magnetrun.symbol":      meta.symbol.encode("utf-8"),
        b"magnetrun.unit":        _serialize_unit(meta.unit),
        b"magnetrun.label":       meta.label.encode("utf-8"),
        b"magnetrun.description": meta.description.encode("utf-8"),
        b"magnetrun.category":    meta.category.encode("utf-8"),
    }

def _deserialize_field_meta(field_metadata: dict[bytes, bytes], ureg) -> FieldMeta:
    return FieldMeta(
        symbol      = field_metadata[b"magnetrun.symbol"].decode("utf-8"),
        unit        = _deserialize_unit(field_metadata[b"magnetrun.unit"], ureg),
        label       = field_metadata.get(b"magnetrun.label",       b"").decode("utf-8"),
        description = field_metadata.get(b"magnetrun.description", b"").decode("utf-8"),
        category    = field_metadata.get(b"magnetrun.category",    b"").decode("utf-8"),
    )
```

### Timestamp helpers

```python
def _serialize_timestamp(ts: datetime) -> tuple[bytes, bytes]:
    """Split naive UTC datetime into (seconds, nanos) byte-strings."""
    pts = pd.Timestamp(ts)
    seconds = pts.value // 1_000_000_000
    nanos   = pts.value % 1_000_000_000
    return str(seconds).encode("utf-8"), str(nanos).encode("utf-8")

def _deserialize_timestamp(seconds_bytes: bytes, nanos_bytes: bytes) -> datetime:
    seconds = int(seconds_bytes.decode("utf-8"))
    nanos   = int(nanos_bytes.decode("utf-8"))
    return pd.Timestamp(seconds * 1_000_000_000 + nanos, unit="ns").to_pydatetime()
```

### S3 wrapper (separate package)

```python
# rustfs/magnetfs/parquet_io.py  or  python_magnetrun/io/s3.py
def save_to_s3(
    mdata: MagnetDataBase,
    bucket: str,
    key: str,
    group: str | None = None,
    s3_client = None,
) -> None:
    buf = BytesIO()
    mdata.saveParquet(buf, group=group)
    buf.seek(0)
    s3 = s3_client or get_s3_client()
    s3.upload_fileobj(buf, bucket, key)

def load_from_s3(bucket: str, key: str, s3_client = None) -> MagnetDataBase:
    s3 = s3_client or get_s3_client()
    buf = BytesIO()
    s3.download_fileobj(bucket, key, buf)
    buf.seek(0)
    return load_magnetrun_parquet(buf)
```

---

## Implementation phases

### Phase 1 — `category` on `FieldMeta` *(prerequisite)*

Independently useful even if Parquet work is deferred. Status: **⏳ todo**.

- Add `category: str = ""` to `FieldMeta` in `magnetdata_base.py`.
- `load_units_from_json` reads `defn.get("category", "")` and stores it.
- `addData` / `computeData` gain a `category=""` kwarg; default for derived
  fields is the source field's category (or `""`).
- TDMS-specific: `addData` derives category from `key.split("/")[0]` if not
  explicitly given.
- HybridData: category from prefix (`kHz`, `rms`, `trigger`).
- `field_defs.add_field_def`/`update_field_def`/`list_field_defs` gain `category`
  kwarg; CLI gains `--category`.
- Editorial: add `"category"` to entries in `pupitre-defs.json`,
  `pigbrother-defs.json`, `hybrid-defs.json`. Categories: `Currents`,
  `Voltages`, `Flows`, `Temperatures`, `Pressures`, `Field`, `Power`, `Misc`.

**Validate:**
```bash
pytest tests/ -k "field_meta or category"
magnetrun-field-defs python_magnetrun/pupitre-defs.json list  # shows category column
```

### Phase 2 — `properties` dict on `MagnetDataBase`

Status: **⏳ todo**.

- Add `self.properties: dict = {}` in `MagnetDataBase.__init__`.
- Add `setProperty(name, value)` and `getProperty(name, default)` methods.
- No change to existing analysis code yet — opt-in.

**Validate:**
```bash
pytest tests/test_magnetdata_base.py -k "properties"
```

### Phase 3 — `saveParquet` / `loadParquet` for `PandasMagnetData`

Status: **⏳ todo**. The simpler case; lands first.

- New module `python_magnetrun/io/parquet.py` with serialization helpers.
- `PandasMagnetData.saveParquet(target, group=None)`:
  - If `group is not None`: raise `ValueError`.
  - Build `pa.Table` with `preserve_index=False`.
  - Drop `timestamp` column if present (D3); keep `t`.
  - Replace schema metadata with file-level keys (D4 timestamps, source info,
    properties JSON).
  - Replace each field's metadata with `_serialize_field_meta(...)`.
  - Write via `pq.write_table`.
- `PandasMagnetData.loadParquet(source)`:
  - Read with `pq.read_table`.
  - Validate `schema_version`.
  - Reconstruct `start_timestamp` / `end_timestamp` from split-nanos keys.
  - Convert table to pandas DataFrame.
  - Walk schema fields, populate `self.units` and `self.field_meta` via
    `_deserialize_field_meta`.
  - Restore `self.properties` from JSON blob.
  - Do *not* materialize `timestamp` column — leave to lazy
    `getTimestamp()`.

**Validate:**
```bash
pytest tests/io/test_parquet_pandas.py -v
# round-trip test: load .txt → saveParquet → loadParquet → assert field_meta,
# units, properties, start_timestamp, t column all match
```

### Phase 4 — `getTimestamp()` lazy materializer

Status: **⏳ todo**.

- Concrete method on `MagnetDataBase`:
  ```python
  def getTimestamp(self) -> pd.Series:
      if "timestamp" in self.Data.columns:
          return self.Data["timestamp"]
      return pd.Timestamp(self.start_timestamp) + pd.to_timedelta(self.Data["t"], unit="s")
  ```
- Audit `extractTimeData` and any cross-source alignment code to call it
  instead of assuming the column exists.

**Validate:**
```bash
pytest tests/ -k "extract_time or timestamp"
```

### Phase 5 — `saveParquet` / `loadParquet` for `TdmsMagnetData`

Status: **⏳ todo**. Multi-group split.

- `TdmsMagnetData.saveParquet(target, group=None)`:
  - If `group is None`: raise `ValueError("TDMS requires a group")`.
  - Extract DataFrame for that group via `getData(group)`.
  - File-level `magnetrun.group` = group name.
  - File-level `magnetrun.start_timestamp_*` = group's `wf_start_time`.
  - Same field-level treatment as pandas, with category derived from key prefix.
- `TdmsMagnetData.loadParquet`:
  - Read `magnetrun.group` from metadata.
  - Reconstruct `self.Groups[group][channel]` with `wf_start_time`,
    `wf_increment` derived from `t` column spacing.
  - Field meta same as pandas.

**Note:** loaded TDMS-from-Parquet has only one group populated. That's by
design — multi-group save is deferred to Phase 8 (manifest layout).

**Validate:**
```bash
pytest tests/io/test_parquet_tdms.py -v
```

### Phase 6 — Top-level factory + `MagnetRun.fromparquet`

Status: **⏳ todo**.

- `python_magnetrun/io/parquet.py::load_magnetrun_parquet(source)` reads
  `magnetrun.source_type` and dispatches.
- `MagnetRun.fromparquet(source)` reads `magnetrun.housing` / `magnetrun.site`
  and wraps the loaded `MagnetDataBase`.
- Symmetric `MagnetRun.saveParquet(path, group=None)` delegates to the
  contained data object.

**Validate:**
```bash
pytest tests/test_magnetrun.py -k "parquet"
```

### Phase 7 — S3 thin wrapper

Status: **⏳ todo**. Lives in `rustfs/magnetfs` *or* a new
`python_magnetrun/io/s3.py` (decide based on whether boto3 should be a hard
dependency of `python_magnetrun` — recommendation: keep it in `rustfs/magnetfs`
to avoid the dependency).

- `save_to_s3(mdata, bucket, key, group=None)`.
- `load_from_s3(bucket, key) -> MagnetDataBase`.
- `magnetfs` CLI gains `magnetrun-save` / `magnetrun-load` subcommands using
  the new path (in addition to the existing demo-only `convert`).

### Phase 8 — Multi-group manifest layout *(future, deferred)*

Directory of Parquet files + top-level `_manifest.json`. Pursue when a concrete
need surfaces (e.g. shipping a complete TDMS run as one artifact, or HybridRun's
three streams together).

---

## Open questions

### Q1 — S3 key convention

Current demo: `M10_2020.10.23---20_10_41.parquet`. Insufficient when the same
run has pupitre, pigbrother, and hybrid versions. Options:

- `{source_type}/{housing}/{run_id}.parquet`
- `{housing}/{run_id}/{source_type}.parquet`
- `{housing}/{run_id}/{source_type}-{group_or_stream}.parquet` (handles
  multi-group TDMS naturally)

Whatever is chosen should be derivable from the file's metadata so the key
itself can be opaque if needed. Decide before Phase 7.

### Q2 — boto3 as a `python_magnetrun` dependency?

If S3 helpers live in `python_magnetrun/io/s3.py`, boto3 becomes a runtime
dependency. If they live in `rustfs/magnetfs`, `python_magnetrun` stays
S3-agnostic but users need a second package for the S3 path.

**Recommendation:** keep S3 in `rustfs/magnetfs` (or a new sibling `magnetfs-s3`
package). `python_magnetrun.saveParquet` accepts `IO[bytes]`, so any storage
backend composes externally.

### Q3 — Compression

Pyarrow defaults to snappy. zstd gives 2-3× better compression at modest CPU
cost and is well-supported. For pigbrother kHz data the saving is significant.

**Recommendation:** default to `compression="zstd"` with `compression_level=3`,
expose `compression=` kwarg on `saveParquet` for callers who want to override.

### Q4 — Properties shape — flat dict vs typed object?

Currently `dict[str, Any]`. Could become a `RunProperties` dataclass with
known fields (`b_min`, `b_max`, `energy`, `plateaux`, `signature`).

**Recommendation:** stay with `dict` for v1. The flexibility matches the
"caller populates whatever they computed" model. Promote to a typed object
only if a clear consumption pattern emerges.

### Q5 — HybridRun integration

`HybridRun` has its own `saveData` (CSV-only, single key). The Parquet
equivalent needs to pick: one stream per file (matches D6) or one Parquet
per (system, signal) pair. Address in a follow-up plan once Phases 1-6 land.

---

## File change summary

| File | Phase | Status | Change |
|---|---|---|---|
| `python_magnetrun/magnetdata_base.py` | 1, 2 | ⏳ todo | `category` on `FieldMeta`; `properties` dict; `setProperty`/`getProperty`; abstract `saveParquet`/`loadParquet`; concrete `getTimestamp` |
| `python_magnetrun/magnetdata_pandas.py` | 3 | ⏳ todo | implement `saveParquet`/`loadParquet`; `addData` accepts `category=` |
| `python_magnetrun/magnetdata_tdms.py` | 5 | ⏳ todo | implement `saveParquet`/`loadParquet` (group-required); category from prefix |
| `python_magnetrun/hybrid/hybrid_data.py` | 1 | ⏳ todo | `category` from prefix; load_units override populates it |
| `python_magnetrun/field_defs.py` | 1 | ⏳ todo | `add/update/list_field_def` gain `category=` kwarg; CLI `--category` |
| `python_magnetrun/io/parquet.py` | 3 | ⏳ todo | new — serialization helpers, `load_magnetrun_parquet` factory |
| `python_magnetrun/MagnetRun.py` | 6 | ⏳ todo | `fromparquet` classmethod; `saveParquet` delegating method |
| `python_magnetrun/pupitre-defs.json` | 1 | ⏳ editorial | add `"category"` to all entries |
| `python_magnetrun/pigbrother-defs.json` | 1 | ⏳ editorial | add `"category"` to all entries |
| `python_magnetrun/hybrid-defs.json` | 1 | ⏳ editorial | add `"category"` to all entries |
| `tests/io/test_parquet_pandas.py` | 3 | ⏳ todo | new — round-trip tests |
| `tests/io/test_parquet_tdms.py` | 5 | ⏳ todo | new — round-trip tests, multi-group |
| `rustfs/magnetfs/parquet_io.py` | 7 | ⏳ todo | new — `save_to_s3`/`load_from_s3` thin wrappers |
| `rustfs/magnetfs/cli.py` | 7 | ⏳ todo | new subcommands using the magnetrun saver |

---

## Execution order

1. **Phase 1** (`category` on `FieldMeta` + JSON-defs editorial) — independently
   useful, unblocks 3 and 5.
2. **Phase 2** (`properties` dict) — tiny, no risk, unblocks 3.
3. **Phase 3** (`PandasMagnetData` Parquet) — first real Parquet code; pupitre
   round-trip working end to end.
4. **Phase 4** (`getTimestamp` lazy) — needed before 5 because TDMS tests will
   exercise extractTimeData.
5. **Phase 5** (`TdmsMagnetData` Parquet).
6. **Phase 6** (factory + `MagnetRun.fromparquet`).
7. **Phase 7** (S3 wrapper) — usable end-to-end pipeline.
8. **Phase 8** (manifest layout) and HybridRun integration — defer.
