# Curated Parquet input + S3-backed file access — discussion notes

*Created: 2026-09-18 — captures a design discussion before implementation begins.
Not yet approved/started; status markers below are all ⏳ todo.*

**Related:** [parquet-save-load.plan.md](parquet-save-load.plan.md) — the existing
(fully designed, ⏳ all-todo) plan for `saveParquet`/`loadParquet` round-tripping
through python_magnetrun's own self-describing Parquet schema. This note treats
that plan as prior art for Part A rather than re-designing it, and layers an S3
transport concern (Part B) on top that the older plan explicitly deferred out of
scope (see its Q2).

## Origin

Two related feature requests, discussed together but intentionally **not
interleaved** since they're separately shippable:

1. Support curated magnet data stored in Parquet format (as an *input* format,
   symmetric with `.txt`/`.csv`/`.tdms`).
2. Read Parquet files — and eventually other file types — from S3.

---

## Part A — Curated Parquet as a first-class read format

**Goal:** `load_magnetdata()` (or an equivalent factory) can load a pre-existing
Parquet file into a working `MagnetDataBase` object.

**Relationship to `parquet-save-load.plan.md`:** that plan already designs the
metadata schema (D1–D6), the `saveParquet`/`loadParquet` API, and a phased
implementation (Phases 1–6), all still `⏳ todo`. Part A here is "resume/implement
that plan's read side" (Phases 1, 2, 3, 4, 5, 6), not a new design.

**Files affected** (read-relevant subset of that plan's file-change table):

| File | Change |
|---|---|
| `python_magnetrun/magnetdata_base.py` | `category` on `FieldMeta`; `properties` dict; abstract `loadParquet` |
| `python_magnetrun/magnetdata_pandas.py` | `loadParquet` classmethod |
| `python_magnetrun/magnetdata_tdms.py` | `loadParquet` classmethod (group-required) |
| `python_magnetrun/io/parquet.py` (new) | deserialization helpers, `load_magnetrun_parquet(source)` dispatcher |
| `python_magnetrun/MagnetRun.py` | `fromparquet` classmethod |
| `python_magnetrun/pupitre-defs.json`, `pigbrother-defs.json` | editorial: add `"category"` |
| `tests/io/test_parquet_pandas.py`, `tests/io/test_parquet_tdms.py` (new) | round-trip tests |

**Verification:** the `pytest` commands already specified per-phase in
`parquet-save-load.plan.md`, plus one integration test loading a real Parquet
fixture end-to-end through `load_magnetdata()`.

**Open question (scoping, not detail):** is "curated" Parquet always
self-produced by `saveParquet` (guaranteed schema, D1–D6 conventions), or could
it also be Parquet assembled by a *different* pipeline (e.g. this repo's own
`to_duckdb`/magnetdb curation layer) with an unknown or looser schema? The
existing plan's `loadParquet` is strict — it fails without the `magnetrun.*`
metadata keys. If foreign Parquet is realistic, Part A needs a second,
best-effort loader path (infer columns/units, tolerate missing metadata)
distinct from the strict one. Settle before implementation starts, since it
changes Phase 3/5's scope.

**Open question (write side):** does resuming this plan also mean resuming
`saveParquet` (the write half), or is curation-and-writing owned by another
tool entirely and python_magnetrun only ever reads? Changes whether Phase 3/5
need their write half at all.

---

## Part B — S3-backed file access

**Goal:** any reader that currently takes a local path can instead be pointed
at an S3 (or S3-compatible) URI — Parquet first, `.tdms`/`.txt`/`.csv` later —
without forcing every python_magnetrun user to install an S3 client.

**Key decision — reuse boto3, don't add fsspec/s3fs.**
`rustfs/magnetfs/client.py` already wraps
`boto3.client(..., endpoint_url=RUSTFS_ENDPOINT)` for this exact RustFS
endpoint (a self-hosted, S3-compatible store — not AWS with IAM/SSO
complexity). boto3 is simpler and already proven against it; fsspec's
abstraction layer would mainly pay off if multiple cloud backends (GCS, Azure)
were in scope, which they aren't. Add boto3 as an **optional extra**
(`s3 = ["boto3>=1.34"]`) on `python_magnetrun`, not a base dependency, so
local-only users pay nothing — this is the same dependency-isolation goal as
`parquet-save-load.plan.md`'s D7/Q2, just resolved by making boto3 optional
inside python_magnetrun rather than pushing it into a sibling package.

**Approach:**

1. New `python_magnetrun/io/storage.py`: `open_source(path_or_uri: str) -> IO[bytes]`,
   detecting `s3://bucket/key` vs. a plain local path. On `s3://`, lazily
   `import boto3` (clear `ImportError` + install hint if the `s3` extra is
   missing), download via `get_object`/`download_fileobj` into a `BytesIO`
   (Parquet) or a `NamedTemporaryFile` (TDMS — see point 4).
2. `readers/registry.py::detect_type` (`python_magnetrun/readers/registry.py:60-73`)
   currently does `Path(path).suffix` / `.is_dir()`, which mishandles `s3://`
   URIs (`Path` collapses the `//`, and `.is_dir()` does a local filesystem
   stat). Fix: sniff the suffix from the raw string before falling through to
   `Path`-based logic for local paths.
3. Wire `open_source()` into `load_magnetdata()` (`magnetdata.py:44`) and Part
   A's `load_magnetrun_parquet` — both already need to accept a byte stream
   per the existing API sketch, so this is "resolve URI → stream, then hand to
   the existing loader," not a rewrite.
4. **TDMS is the hard case:** `nptdms.TdmsFile.open()` needs random seek
   access; a 250 MB TDMS file over range-requested S3 reads would be slow and
   chatty. Recommend: download-to-tempfile for TDMS-over-S3 initially; revisit
   true streaming only if it's a measured bottleneck. Parquet has no such
   problem — `pyarrow`/`polars` both accept file-like objects directly.

**Verification:** unit tests against a mocked S3 endpoint (`moto`, or point
boto3 at a local MinIO/RustFS test instance if one exists in CI); a manual
smoke test against the real RustFS endpoint if reachable.

**Files affected:** `pyproject.toml` (new `s3` extra),
`python_magnetrun/io/storage.py` (new), `python_magnetrun/magnetdata.py`,
`python_magnetrun/readers/registry.py`, `tests/io/test_storage_s3.py` (new).

**Open question:** "eventually other files" — scoping this plan to
Parquet-over-S3 only for now; raw-file-over-S3 deferred until a concrete need
names a format.

---

## How the two parts relate

Independent and separately shippable, in either order. Part A works standalone
on local disk. Part B is generic transport — parquet is its first consumer,
but it doesn't depend on Part A existing (it can equally serve `.tdms`-from-S3
later). They only compose at "load curated Parquet that happens to live in
S3," which is just Part B's `open_source()` feeding Part A's
`load_magnetrun_parquet`.

## Status

Discussion only — no implementation started. Nothing here is approved; treat
this as the record of the design conversation, to be turned into an
approved, phased plan (mirroring `parquet-save-load.plan.md`'s format) once
the open questions above are settled.
