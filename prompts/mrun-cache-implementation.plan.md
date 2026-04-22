# MagnetRun Object Cache — Implementation Plan

## Problem

`MagnetRun` objects are expensive to construct: each call to `MagnetRun.fromtxt()` or
`MagnetRun.fromtdms()` parses a file from disk. In the current flow the same file
is loaded **twice** for every path processed:

1. **`extract_data()`** — called by `select_files()` to obtain the time-range
   (`start_ftimestamp`, `end_ftimestamp`) so that files can be filtered.
2. **`load_df()`** — called by `load_data()` to actually read channel data into a
   `DataFrame`.

When `select_files()` is run before `load_data()` (the normal pipeline), every file
that passes the filter is loaded a second time from scratch.

---

## Proposed Solution: `lru_cache` on a private factory helper

Add a single cached factory function `_load_mrun` at module level in
`python_magnetrun/analysis/loaders.py`.  All internal callers (`extract_data`,
`load_df`) delegate to this helper instead of calling `MagnetRun.from*` directly.

```python
import functools

@functools.lru_cache(maxsize=128)
def _load_mrun(filepath: str, housing: str, site: str):
    """Return a cached MagnetRun object for *filepath*."""
    from python_magnetrun.MagnetRun import MagnetRun

    extension = os.path.splitext(filepath)[-1]
    if extension == ".txt":
        return MagnetRun.fromtxt(housing, site, filepath)
    elif extension == ".tdms":
        return MagnetRun.fromtdms(housing, site, filepath)
    else:
        raise RuntimeError(f"{filepath}: unsupported extension {extension}")
```

**Cache key**: `(filepath, housing, site)` — all strings, so hashable by default.

**Cache size**: 128 entries (configurable). Covers all files in a typical session
without unbounded memory growth.

---

## Implementation Steps

- [x] **Step 1** — Add `import functools` to the imports block in `loaders.py`.
- [x] **Step 2** — Define `_load_mrun` with `@functools.lru_cache(maxsize=128)`
  after the module logger, before the dataclass definitions.
- [x] **Step 3** — Replace `MagnetRun.fromtxt(housing, site, file)` calls in
  `extract_data()` with `_load_mrun(file, housing, site)`.
- [x] **Step 4** — Replace `MagnetRun.fromtdms(housing, site, file)` calls in
  `extract_data()` with `_load_mrun(file, housing, site)`.
- [x] **Step 5** — Replace both `MagnetRun.from*` calls in `load_df()` with
  `_load_mrun(file, housing, site)`.
- [x] **Step 6** — Remove the now-redundant lazy `from python_magnetrun.MagnetRun
  import MagnetRun` imports that were local to `extract_data()` and `load_df()`.

---

## Cache Management

| Operation                | How                                       |
| ------------------------ | ----------------------------------------- |
| Inspect cache statistics | `_load_mrun.cache_info()`                 |
| Invalidate all entries   | `_load_mrun.cache_clear()`                |
| Change cache size        | Edit `maxsize` in `@lru_cache(maxsize=N)` |

Call `_load_mrun.cache_clear()` if a file on disk changes during a session (e.g. in
test fixtures that write and re-read the same path).

---

## Affected Files

| File                                   | Change                                             |
| -------------------------------------- | -------------------------------------------------- |
| `python_magnetrun/analysis/loaders.py` | Add `_load_mrun`; update `extract_data`, `load_df` |

No other modules are affected — `_load_mrun` is a private, module-level detail.

---

## Limitations & Future Work

- The cache lives for the **lifetime of the process**.  Long-running services or
  notebooks that re-use the same filenames with different content must call
  `_load_mrun.cache_clear()` between runs.
- `MagnetRun` objects are assumed **immutable** after construction.  If any caller
  mutates the object the mutation will be visible to all future users of the cache.
  If mutation is needed, callers should work on a copy.
- If memory pressure becomes an issue, reduce `maxsize` or switch to
  `functools.cached_property` on the discovery objects that already hold file paths.
