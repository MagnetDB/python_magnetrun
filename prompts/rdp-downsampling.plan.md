# RDP / Visvalingam-Whyatt Downsampling Plan

Date: 2026-05-28

Effort key: **S** = ~1 h, **M** = half-day.

---

## Motivation

The Ramer-Douglas-Peucker (RDP) algorithm simplifies a polyline by removing points whose
deviation from the simplified curve is below an `epsilon` threshold.  Unlike all current
methods (`stride`, `minmax`, `m4`, `lttb`, …), it is **geometry-based** rather than
count-based: it naturally allocates more points to complex regions (spikes, transitions) and
fewer to flat plateaus.  This makes it well-suited for magnet run data where long
constant-field plateaus alternate with fast ramp transitions.

The `simplification` package (Rust-backed, v0.7.x) provides both RDP and
Visvalingam-Whyatt (VW) via `simplify_coords_idx`, returning sorted index arrays of the same
type as `tsdownsample` — so they slot directly into `_downsample_indices`.

**Tested output** (1 000-point sin wave, `epsilon=0.01`):
```
RDP  → 38 points   (simplify_coords_idx)
VW   → 47 points   (simplify_coords_vw_idx)
```

---

## Key design difference: epsilon vs n_out

| Existing methods | RDP / VW |
|-----------------|----------|
| Accept `n_out` (target count) | Accept `epsilon` (geometry tolerance) |
| Output size is deterministic | Output size depends on data shape |
| `DownsampleConfig(n_out=5000)` | `DownsampleConfig(n_out=5000, epsilon=0.01)` |

`DownsampleConfig` currently has `n_out` as a required field and `bucket_size` as an optional
knob.  RDP/VW need an additional `epsilon: float | None = None` field.

Two usage modes are supported:

1. **Epsilon-first** — caller sets `epsilon`; output size is unknown in advance.
   `n_out` acts as a safety cap (truncate if the simplified curve has more points).
2. **n_out-first** — caller sets only `n_out`; implementation binary-searches for the
   epsilon that yields approximately `n_out` points (±10 %).  Slower but consistent with
   the existing interface.

The binary-search path is optional in Phase 1 (document the epsilon requirement); it can be
added in Phase 2 once the basic path is validated.

---

## Algorithm details (full table)

| Method | Param | NaN handling | Characteristic |
|--------|-------|--------------|----------------|
| `stride` | `n_out` | strip | uniform, fast |
| `minmax` | `n_out` | strip | envelope only |
| `lttb` | `n_out` | strip | perceptual fidelity |
| `minmax_lttb` | `n_out` | strip | best perceptual fidelity |
| `m4` | `n_out` | strip | pixel-perfect line chart |
| `nan_m4` | `n_out` | native | pixel-perfect + gap-aware |
| **`rdp`** (new) | **`epsilon`** | strip | geometry-based, plateau-aware |
| **`vw`** (new) | **`epsilon`** | strip | geometry-based, area-based simplification |

---

## Target design

### Phase 1 — `DownsampleConfig` extension + `rdp` / `vw` dispatch (effort: S)

**`python_magnetrun/utils/downsampling.py`**

1. Add optional dependency guard for `simplification`:

   ```python
   try:
       from simplification.cutil import simplify_coords_idx, simplify_coords_vw_idx
       HAS_SIMPLIFICATION = True
   except ImportError:
       HAS_SIMPLIFICATION = False
       logger.debug("simplification not available — rdp/vw methods will use simple stride")
   ```

2. Add `epsilon` field to `DownsampleConfig`:

   ```python
   @dataclass(frozen=True)
   class DownsampleConfig:
       n_out: int
       method: str = "stride"
       bucket_size: int | None = None   # minmax only
       epsilon: float | None = None     # rdp / vw only
   ```

3. Add `rdp` and `vw` branches to `_downsample_indices`.  Both methods need a 2D coordinate
   array `(time, data)`:

   ```python
   if config.method in ("rdp", "vw"):
       if not HAS_SIMPLIFICATION:
           logger.warning(
               "method=%r requires simplification (pip install python_magnetrun[rdp]); "
               "falling back to 'stride'",
               config.method,
           )
       else:
           if config.epsilon is None:
               raise ValueError(
                   f"DownsampleConfig.epsilon must be set when method='{config.method}'. "
                   "Use DownsampleConfig(n_out=..., method='rdp', epsilon=0.01) or "
                   "call DownsampleConfig.from_n_out_rdp() to auto-search epsilon."
               )
           coords = np.column_stack([time.astype(float), data.astype(float)])
           fn = simplify_coords_idx if config.method == "rdp" else simplify_coords_vw_idx
           indices = fn(coords, config.epsilon).astype(np.intp)
           # Apply n_out cap if the simplified result exceeds it
           if len(indices) > config.n_out:
               indices = indices[: config.n_out]
           return indices
   ```

4. Add a convenience factory `DownsampleConfig.from_n_out_rdp` that binary-searches epsilon:

   ```python
   @classmethod
   def from_n_out_rdp(
       cls,
       data: np.ndarray,
       time: np.ndarray,
       n_out: int,
       method: str = "rdp",
       tol: float = 0.1,
       max_iter: int = 30,
   ) -> "DownsampleConfig":
       """Find epsilon such that RDP/VW returns approximately n_out points (±tol*n_out).

       Performs a binary search over epsilon in [eps_lo, eps_hi].
       Raises RuntimeError if convergence fails within max_iter.
       """
       from simplification.cutil import simplify_coords_idx, simplify_coords_vw_idx
       fn = simplify_coords_idx if method == "rdp" else simplify_coords_vw_idx
       coords = np.column_stack([time.astype(float), data.astype(float)])

       eps_lo, eps_hi = 1e-9, float(np.ptp(data))
       for _ in range(max_iter):
           eps = (eps_lo + eps_hi) / 2
           n = len(fn(coords, eps))
           if abs(n - n_out) <= tol * n_out:
               return cls(n_out=n_out, method=method, epsilon=eps)
           if n > n_out:
               eps_lo = eps
           else:
               eps_hi = eps
       # Return best-effort epsilon
       return cls(n_out=n_out, method=method, epsilon=(eps_lo + eps_hi) / 2)
   ```

5. Update `DownsampleConfig` docstring:

   ```
   method:
       Algorithm: 'minmax_lttb' | 'lttb' | 'minmax' | 'm4' | 'nan_m4' | 'rdp' | 'vw' | 'stride'.
   epsilon:
       Geometry tolerance for 'rdp' and 'vw' methods.  Larger values produce more
       aggressive simplification.  Required when method is 'rdp' or 'vw' unless
       using DownsampleConfig.from_n_out_rdp().
   ```

---

### Phase 2 — `pyproject.toml` extras group (effort: S)

Add `simplification` as a new optional dependency group:

```toml
[project.optional-dependencies]
rdp = ["simplification>=0.7"]
```

Document the soft requirement in `utils/downsampling.py` module docstring:

```
Optional dependency: ``simplification`` (declared in the ``rdp`` extras group).
The ``rdp`` and ``vw`` methods require it at runtime.
Install with: ``pip install python_magnetrun[rdp]``
```

---

### Phase 3 — CLI surface (effort: S)

**`python_magnetrun/cli_args.py`**

1. Add `'rdp'` and `'vw'` to `DOWNSAMPLE_METHODS`:

   ```python
   DOWNSAMPLE_METHODS = (
       "none", "stride", "minmax", "minmax_lttb", "lttb",
       "m4", "nan_m4", "rdp", "vw",
   )
   ```

2. Add `epsilon` key to the `--downsample-params` help text and docstring examples:

   ```
   --downsample-method rdp --downsample-params '{"n_out": 5000, "epsilon": 0.01}'
       RDP geometry simplification: keeps more points in transitions, fewer in plateaus.

   --downsample-method vw --downsample-params '{"n_out": 5000, "epsilon": 0.001}'
       Visvalingam-Whyatt: area-based variant of RDP, smoother on noisy data.
   ```

3. Update the `DownsampleConfig` construction site in any CLI-to-config bridge (e.g.,
   wherever `--downsample-params` JSON is unpacked into `DownsampleConfig(...)`) to pass
   through the `epsilon` key if present.

   Locate the bridge in `cli_args.py` or `analysis/cli.py` (search for
   `json.loads` near `DownsampleConfig`) and add:

   ```python
   params = json.loads(args.downsample_params) if args.downsample_params else {}
   config = DownsampleConfig(
       n_out=params.get("n_out", 10_000),
       method=args.downsample_method,
       bucket_size=params.get("bucket_size"),
       epsilon=params.get("epsilon"),          # new
   )
   ```

---

### Phase 4 — Tests (effort: S)

Extend **`tests/test_downsampling.py`**:

| Test | What it checks |
|------|---------------|
| `test_rdp_with_epsilon_reduces_points` | output < input for synthetic sin wave |
| `test_rdp_indices_sorted` | output time axis is non-decreasing |
| `test_rdp_plateau_awareness` | a flat segment produces fewer output points than a spike segment |
| `test_rdp_n_out_cap` | when simplified result > `n_out`, output is truncated to `n_out` |
| `test_rdp_raises_without_epsilon` | `ValueError` raised when epsilon is None |
| `test_vw_with_epsilon` | VW path returns reduced output |
| `test_from_n_out_rdp_convergence` | `from_n_out_rdp` returns config with approximately `n_out` points |
| `test_rdp_fallback_no_simplification` | with `HAS_SIMPLIFICATION=False` patched, falls back to stride |
| `test_rdp_downsample_dataframe` | DataFrame path produces correct row count |

---

## File change summary

| File | Change | Effort |
|------|--------|--------|
| `python_magnetrun/utils/downsampling.py` | Add `HAS_SIMPLIFICATION` guard; `epsilon` field on `DownsampleConfig`; `'rdp'`/`'vw'` branches; `from_n_out_rdp()` factory | S |
| `python_magnetrun/cli_args.py` | Add `'rdp'`, `'vw'` to `DOWNSAMPLE_METHODS`; pass `epsilon` from JSON params | S |
| `pyproject.toml` | Add `rdp = ["simplification>=0.7"]` extras group | S |
| `tests/test_downsampling.py` | 9 new test cases | S |

**No changes required** in:
- `analysis/processing.py` — `DownsampleConfig.from_percent(method=...)` passes `method` opaquely
- `analysis/args.py` — inherits updated `create_downsampling_parser()` automatically
- `magnetdata_pandas.py`, `magnetdata_tdms.py`, `hybrid/hybrid_run.py` — consume `DownsampleConfig` opaquely

---

## Sequencing relative to m4-downsampling.plan.md

RDP/VW are **independent** of M4/NaN-M4 — the only shared touch-point is
`DownsampleConfig` (adding `epsilon` field) and `DOWNSAMPLE_METHODS`.  Either plan can land
first; when both are implemented the `DownsampleConfig` dataclass change should be done in a
single commit to avoid double-touches.

Recommended order if implementing both:
1. Land `m4` + `nan_m4` (no `DownsampleConfig` field change needed).
2. Land RDP/VW (adds `epsilon` field, `simplification` dependency).
3. Combine CLI surface update for both in one commit.

---

## Trade-offs vs count-based methods

| | Count-based (`stride`, `m4`, `lttb`) | Geometry-based (`rdp`, `vw`) |
|-|--------------------------------------|------------------------------|
| Output size | Deterministic | Data-dependent |
| Parameter | `n_out` | `epsilon` (or binary-search) |
| Plateau regions | Wastes points | Naturally sparse |
| Transition regions | Fixed density | Naturally dense |
| Reproducibility | Same output every run | Same for same epsilon |
| `from_percent` bridge | Works directly | Needs `from_n_out_rdp()` |

For magnet run data (long plateaus + fast ramps) RDP/VW should produce more compact and
visually accurate representations than count-based methods at equivalent file sizes.
