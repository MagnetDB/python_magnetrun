# Plan: `Data` as abstract property on `MagnetDataBase`

## Motivation

`self.Data` is currently a plain public attribute on `MagnetDataBase`.  The lazy
loading introduced for Problem 2 layers two different magic mechanisms on top of
it:

| Subclass | Mechanism | Trigger |
|---|---|---|
| `PandasMagnetData` | `__getattribute__` override | `obj.Data` (attribute read) |
| `TdmsMagnetData` | `_LazyGroupDict.__getitem__` | `obj.Data[group]` (subscript) |

These are asymmetric, invisible to the ABC, and leave `close()` off the public
interface.  The fix (Option B) makes `Data` an abstract `@property` on the ABC
so that lazy-load triggering is part of the declared contract, both subclasses
follow the same pattern, and resource lifecycle (`close()`) is part of the base
class.

A secondary bug is also revealed: `_validate_start_timestamp` accesses
`self.Data` during `__init__`, which currently fires a full file load for every
pupitre `.txt` file — defeating the lazy loading for that path.

---

## Changes

### Step 1 — `magnetdata_base.py`: replace the `Data` attribute with an abstract property

**Remove** from `__init__`:
```python
# before
self.Data: pd.DataFrame | dict = Data if Data is not None else pd.DataFrame()
```

**Remove** `Data: pd.DataFrame | dict | None = None` from the `__init__`
signature (the parameter moves to subclass constructors only).

**Add** abstract property declarations (getter + setter) to the class body:
```python
@property
@abstractmethod
def Data(self) -> pd.DataFrame | dict:
    """The loaded dataset.  Accessing this property triggers lazy loading."""

@Data.setter
@abstractmethod
def Data(self, value: pd.DataFrame | dict) -> None:
    """Set the dataset backing store."""
```

Python requires both getter and setter to be declared abstract so that a
concrete subclass that does not implement both remains abstract.

**Add** `close()` with a no-op default and context-manager support:
```python
def close(self) -> None:
    """Release any open file handles.  No-op by default."""

def __enter__(self) -> "MagnetDataBase":
    return self

def __exit__(self, *exc_info) -> None:
    self.close()
```

This allows `with load_magnetdata(path) as mdata:` and makes TDMS resource
cleanup possible from generic calling code.

---

### Step 2 — `magnetdata_pandas.py`: implement `Data` as a property

**Add** a private `_data: pd.DataFrame` backing attribute.  Initialize it in
`__init__` with the `Data` parameter (or an empty DataFrame):

```python
def __init__(self, filename, Groups, Keys, Data=None, ...):
    self._data: pd.DataFrame = Data if isinstance(Data, pd.DataFrame) else pd.DataFrame()
    self._data_loaded: bool = self._data is not None and len(self._data) > 1
    self._read_kwargs: dict = _read_kwargs or {}
    super().__init__(filename, Groups, Keys, defs_file=defs_file, ...)
    ...
```

Note: `super().__init__` no longer receives `Data` (that parameter is removed
from the ABC).  The ABC `__init__` no longer touches `Data` at all.

**Add** the property implementation:
```python
@property
def Data(self) -> pd.DataFrame:
    self._ensure_data_loaded()
    return self._data

@Data.setter
def Data(self, value: pd.DataFrame) -> None:
    self._data = value
```

**Remove** `__getattribute__`.  The property getter replaces it entirely and is
simpler (no `object.__getattribute__` gymnastics, no recursion risk).

**Keep** `_data_loaded`, `_read_kwargs`, and `_ensure_data_loaded()` unchanged.
The explicit `self._ensure_data_loaded()` calls in `getPandasData`,
`cleanupData`, and `addTime` become redundant (the getter calls it) but can stay
as defence-in-depth with no harm.

**Internal raw access**: methods that only need the stub and must not trigger a
full load (specifically `_validate_start_timestamp` — see Step 4) access
`self._data` directly, bypassing the property.

---

### Step 3 — `magnetdata_tdms.py`: implement `Data` as a property

**Add** a private `_data: _LazyGroupDict` backing attribute.  Initialize it in
`__init__` before calling `super()`:

```python
def __init__(self, filename, Groups, Keys, Data=None, ...):
    lazy = _LazyGroupDict(self)
    if isinstance(Data, dict):
        lazy.update(Data)
    self._data: _LazyGroupDict = lazy
    self._tdms_file = _tdms_file
    self._tdms_groups = _tdms_groups or {}
    super().__init__(filename, Groups, Keys, defs_file=defs_file, ...)
    ...
```

Note: `_LazyGroupDict` must be constructed before `super().__init__` because
`_apply_wf_timestamps` (called from `_validate_start_timestamp` inside
`super().__init__`) may indirectly access `self._data`.  Constructing it first
ensures the attribute exists.

**Add** the property:
```python
@property
def Data(self) -> _LazyGroupDict:
    return self._data

@Data.setter
def Data(self, value: dict) -> None:
    if isinstance(value, _LazyGroupDict):
        self._data = value
    else:
        lazy = _LazyGroupDict(self)
        lazy.update(value)
        self._data = lazy
```

The getter does not call `_ensure_group_loaded` — that still happens inside
`_LazyGroupDict.__getitem__` on subscript access.  This is the correct level:
`obj.Data` returns the lazy container; `obj.Data[group]` triggers per-group
loading.

**Remove** the inline `_LazyGroupDict` setup block that currently appears in
`__init__` after the `super()` call:
```python
# remove this block — now handled by __init__ + property setter
lazy: _LazyGroupDict = _LazyGroupDict(self)
if isinstance(self.Data, dict):
    lazy.update(self.Data)
self.Data = lazy
```

`close()` is already implemented on `TdmsMagnetData` and now satisfies the ABC
default override.

---

### Step 4 — Fix the `_validate_start_timestamp` eager-load bug

**Current bug**: `_validate_start_timestamp` accesses `self.Data` during
`__init__`.  With the property in place, this triggers `_ensure_data_loaded()`
while `_read_kwargs` is not yet set (for TDMS) or causes a full file load for
every pupitre file — defeating the lazy loading.

**Fix**: access `self._data` directly (the raw backing store) in
`_validate_start_timestamp`, since the stub (1-row DataFrame from `nrows=1`)
already contains the `Date`/`Time` values needed for cross-checking.

```python
def _validate_start_timestamp(self) -> None:
    if "Date" not in self.Keys or "Time" not in self.Keys:
        return
    df = self._data          # raw backing store — no lazy trigger
    if not isinstance(df, pd.DataFrame) or df.empty:
        return
    ...
```

This is the one place inside the class where bypassing the property is correct:
only row 0 is needed, the stub has it, and firing a full load here would defeat
the purpose.  All other internal accesses use `self.Data` (via the property) and
correctly trigger loading.

---

### Step 5 — Update `assert isinstance` guards

The guards `assert isinstance(self.Data, pd.DataFrame)` and
`assert isinstance(self.Data, dict)` continue to work because:

- `PandasMagnetData.Data` getter returns `self._data` (a `pd.DataFrame`)
- `TdmsMagnetData.Data` getter returns `self._data` (a `_LazyGroupDict`, which
  is a `dict` subclass)

No changes required.

---

## Files affected

| File | Change |
|---|---|
| `magnetdata_base.py` | Remove `Data` attribute from `__init__` and its parameter; add abstract `Data` property (getter + setter); add `close()`, `__enter__`, `__exit__` |
| `magnetdata_pandas.py` | Add `_data` backing attr; implement `Data` property; remove `__getattribute__`; fix `_validate_start_timestamp` to use `self._data` |
| `magnetdata_tdms.py` | Add `_data` backing attr; implement `Data` property; remove inline `_LazyGroupDict` setup from `__init__` |
| `magnetdata.py` | No changes — `TdmsMagnetData(...)` constructor call is unchanged |

---

## External callers

All external accesses to `.Data` (in `loaders.py`, `test-fft.py`,
`test-simu.py`, `test-paramident.py`, `test-intercept.py`, `test-fieldfactor.py`)
continue to work unchanged:

- `mdata.Data[group]["channel"]` → property getter → `_LazyGroupDict.__getitem__`
  → `_ensure_group_loaded`
- `mdata.Data["column"]` → property getter → `_ensure_data_loaded` → full
  DataFrame column access

No caller changes are required.

---

## Tests to add

All in `tests/test_magnetdata.py`:

- `test_data_property_triggers_load_pandas`: construct via `fromtxt`, check
  `_data_loaded` is False, access `obj.Data`, assert `_data_loaded` is now True
  and `len(obj.Data) > 1`.
- `test_data_property_triggers_load_tdms`: construct via `_fromtdms`, access
  `obj.Data["Courants_Alimentations"]`, assert group is now in `obj._data`.
- `test_validate_start_timestamp_does_not_trigger_full_load`: construct via
  `fromtxt`, assert `_data_loaded` is False immediately after construction (i.e.
  `_validate_start_timestamp` did not trigger the full load).
- `test_context_manager_closes_tdms`: use `with load_magnetdata(tdms_path) as
  mdata:`, assert `mdata._tdms_file` is `None` after the `with` block exits.
- `test_close_noop_for_pandas`: `PandasMagnetData.fromtxt(...).close()` raises
  no exception.

---

## Interaction with the truncated-pupitre plan

The truncated-pupitre plan applies `on_bad_lines="warn"` and encoding fallback
inside `_ensure_data_loaded`.  That method is unchanged by this plan — only how
it is triggered changes (property getter instead of `__getattribute__`).  The
two plans are independent and can be applied in either order.
