# Prompt: Move Server-Bridge Objects into `requests/` Submodule

## Objective

Consolidate the server-data bridge layer by moving `GObject.py`, `HMagnet.py`,
`MRecord.py`, and `deserialize.py` from the top-level `python_magnetrun/` package
into the `python_magnetrun/requests/` submodule, where they exclusively belong.

## Rationale

- `GObject`, `HMagnet`, and `MRecord` are data-model classes whose **only** consumers
  are `requests/cli.py` and `requests/webscrapping.py` — they represent objects
  fetched and parsed from `args.server` (the LNCMI monitoring web service).
- `MRecord` already had a hard dependency on `requests/connect.py` (`download`).
- `deserialize.py` only registers and (de)serialises these three classes — it has no
  other consumers in the package.
- Moving all four files creates a self-contained **fetch → model → serialise** pipeline
  entirely inside `requests/`.

## Changes Made (branch: `separate-cooling`)

### Files moved
```
python_magnetrun/GObject.py     → python_magnetrun/requests/GObject.py
python_magnetrun/HMagnet.py     → python_magnetrun/requests/HMagnet.py
python_magnetrun/MRecord.py     → python_magnetrun/requests/MRecord.py
python_magnetrun/deserialize.py → python_magnetrun/requests/deserialize.py
```

### Import fixes

| File | Old import | New import |
|---|---|---|
| `requests/MRecord.py` | `from .requests.connect import download` | `from .connect import download` |
| `requests/cli.py` | `from .. import HMagnet` / `from .. import MRecord` | `from . import HMagnet` / `from . import MRecord` |
| `requests/webscrapping.py` | `from .. import MRecord` / `from .. import GObject` | `from . import MRecord` / `from . import GObject` |
| `requests/GObject.py`, `HMagnet.py`, `MRecord.py` | `from . import deserialize` (`.` was `python_magnetrun`) | unchanged — `.` now correctly refers to `python_magnetrun.requests` |
| `requests/deserialize.py` | `from . import MRecord / GObject / HMagnet` | unchanged — same package |

### `requests/__init__.py` — added re-exports
```python
from .GObject import GObject
from .HMagnet import HMagnet
from .MRecord import MRecord
from . import deserialize
```

## Future Considerations

- If additional serialisable model classes are ever added outside `requests/`, extract
  a shared `python_magnetrun/serialization.py` registry rather than importing from
  `requests/` in the top-level package.
- The `requests/` directory name reflects its original "HTTP request helpers" purpose.
  If the bridge layer grows substantially, consider renaming it to `srvdata/` or
  `magnetdb_bridge/` for clarity.
