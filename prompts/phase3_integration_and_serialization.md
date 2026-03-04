# Phase 3 – Integration & Serialization (Weeks 9–14)

## Goal

Add production-grade serialization (replacing the hand-written JSON code),
integrate with `python_magnetapi` via a structured client module, centralise
all site/housing configuration in a single YAML-driven source, and improve test
coverage and reporting.

**Prerequisite:** Phase 2 is complete (`fetchers/`, `cli_main.py`, `magnetdata/` package, `protocols.py`, ruff/mypy green).

---

## Scope

### 3.1 Replace hand-written JSON serialization with Pydantic v2

**Problem:** `python_magnetrun/deserialize.py` contains a custom `serialize_instance()` function that is brittle and hard to extend. `MRecord`, `GObject`, and `HMagnet` build and parse JSON by hand.

**Target:** Convert `MRecord`, `GObject`, `HMagnet` to `pydantic.BaseModel`. This provides:
- Automatic validation on construction.
- `.model_dump()` / `.model_dump_json()` for free.
- JSON Schema generation (useful for API compatibility with `python_magnetdb`).

**Steps:**

1. Add `pydantic>=2.0` to core dependencies in `pyproject.toml`.

2. Rewrite `python_magnetrun/MRecord.py`:

   ```python
   from pydantic import BaseModel, Field, model_validator

   class MRecord(BaseModel):
       """A record for a single magnet run."""
       site: str
       starttime: datetime
       stoptime: datetime
       # ... other fields

       @model_validator(mode="before")
       @classmethod
       def _coerce_times(cls, data):
           # handle string → datetime conversion
           ...

       def to_json(self) -> str:
           return self.model_dump_json(indent=2)

       @classmethod
       def from_json(cls, s: str) -> "MRecord":
           return cls.model_validate_json(s)

       def getDataFilename(self) -> str:
           ...
   ```

3. Rewrite `python_magnetrun/GObject.py` and `python_magnetrun/HMagnet.py` similarly.

4. Simplify or remove `python_magnetrun/deserialize.py`:
   - If `serialize_instance()` is only used by the above three classes, delete it.
   - If used elsewhere, document which callers remain and annotate with `# TODO: migrate`.

5. Verify `MRecord.__eq__` still works (pydantic BaseModel provides `__eq__` by default, comparing all fields).

**Acceptance criteria:**
- `from python_magnetrun.MRecord import MRecord` imports without error.
- `record.to_json()` returns valid JSON.
- `MRecord.from_json(record.to_json()) == record` is `True`.
- `pytest tests/test_mrecord.py -v` passes.
- `mypy` still passes.

---

### 3.2 Centralise site/housing configuration in YAML

**Problem:** Housing-specific logic (M8, M9, M10) is duplicated between
`python_magnetrun/MagnetRun.py` (`prepareData()`, lines 40–60) and
`python_magnetrun/analysis/config.py`. Adding a new site (e.g., M11) requires
editing Python code.

**Target:** Load site config from a YAML file; `prepareData()` reads from the
loaded dict.

#### 3.2.1 Create `data/sites.toml`

```toml
# data/sites.toml
# Site-specific channel mappings for prepareData()
# Loaded via tomllib (Python 3.12 stdlib — no extra dependency).

[M8]
IH_channels = ["Idcct3", "Idcct4"]
IB_channels = ["Idcct1", "Idcct2"]

[M8.flow_mapping]
Flow1 = "FlowB"
Flow2 = "FlowH"

[M9]
IH_channels = ["Idcct1", "Idcct2"]
IB_channels = ["Idcct3", "Idcct4"]

[M9.flow_mapping]
Flow1 = "FlowH"
Flow2 = "FlowB"

[M10]
IH_channels = ["Idcct1", "Idcct2"]
IB_channels = ["Idcct3", "Idcct4"]

[M10.flow_mapping]
Flow1 = "FlowH"
Flow2 = "FlowB"
```

Use `tomllib` (Python 3.12 stdlib — no extra dependency) with a `.toml` file instead of YAML.
This avoids a `pyyaml` dependency and prevents conflicts with `python_magnetgeo`, which owns its own YAML configuration.

#### 3.2.2 Create a `SiteConfig` dataclass

**File:** `python_magnetrun/analysis/config.py` (extend existing `SiteConfig`)

```python
from dataclasses import dataclass, field
from pathlib import Path
import yaml

@dataclass
class ChannelMapping:
    IH_channels: list[str]
    IB_channels: list[str]
    flow_mapping: dict[str, str]

@dataclass
class SiteConfig:
    name: str
    channel_mapping: ChannelMapping
    # ... existing fields

_SITES: dict[str, SiteConfig] = {}

def load_sites(toml_path: Path | None = None) -> dict[str, SiteConfig]:
    import tomllib
    if toml_path is None:
        toml_path = Path(__file__).parent.parent.parent / "data" / "sites.toml"
    with toml_path.open("rb") as f:
        raw = tomllib.load(f)
    return {
        name: SiteConfig(
            name=name,
            channel_mapping=ChannelMapping(**cfg),
        )
        for name, cfg in raw.items()
    }

def get_site_config(housing: str) -> SiteConfig:
    global _SITES
    if not _SITES:
        _SITES = load_sites()
    if housing not in _SITES:
        raise KeyError(
            f"Unknown housing {housing!r}. Known: {list(_SITES)}. "
            f"Add it to data/sites.yaml."
        )
    return _SITES[housing]
```

#### 3.2.3 Rewrite `prepareData()` in `MagnetRun.py`

Replace the `if housing == "M9": ... elif housing == "M8": ...` chain:

```python
def prepareData(self) -> None:
    from python_magnetrun.analysis.config import get_site_config
    cfg = get_site_config(self.Housing)
    cm = cfg.channel_mapping
    # Use cm.IH_channels, cm.IB_channels, cm.flow_mapping
    ...
```

#### 3.2.4 Remove hard-coded developer paths

**File:** `python_magnetrun/analysis/config.py` (lines ~116–134)

Replace absolute paths like `/home/LNCMI-G/christophe.trophime/...` with
environment-variable lookups:

```python
import os
DATA_DIR = Path(os.environ.get("MAGNETRUN_DATA_DIR", Path(__file__).parent.parent.parent / "data"))
CONFIG_DIR = Path(os.environ.get("MAGNETRUN_CONFIG_DIR", Path.home() / ".config" / "magnetrun"))
```

**Acceptance criteria:**
- `prepareData()` works for M8, M9, M10 after loading from YAML.
- Adding a new site to `data/sites.yaml` and calling `prepareData()` works without code changes.
- No absolute `/home/...` paths remain in any source file.
- `pytest tests/test_magnetrun.py -v` passes.

---

### 3.3 Create `python_magnetrun/api/` client module

**File structure:**

```
python_magnetrun/api/
├── __init__.py
├── client.py      # MagnetAPIClient class
├── models.py      # Pydantic response models
├── auth.py        # Token/credential management
└── cli.py         # magnetrun-api CLI subcommands
```

#### 3.3.1 `api/auth.py`

```python
"""
Credential management for python_magnetapi.

Reads from environment variables:
    MAGNETAPI_URL  – base URL of the API server
    MAGNETAPI_KEY  – API key or bearer token (optional for read-only ops)
"""
import os
from dataclasses import dataclass

@dataclass
class APICredentials:
    base_url: str
    api_key: str | None = None

    @classmethod
    def from_env(cls) -> "APICredentials":
        url = os.environ.get("MAGNETAPI_URL")
        if not url:
            raise EnvironmentError(
                "MAGNETAPI_URL environment variable is not set."
            )
        return cls(base_url=url.rstrip("/"), api_key=os.environ.get("MAGNETAPI_KEY"))
```

#### 3.3.2 `api/models.py`

Pydantic models for API responses:

```python
from pydantic import BaseModel
from datetime import datetime

class MagnetRunRecord(BaseModel):
    id: int
    site: str
    starttime: datetime
    stoptime: datetime
    insert: str
    status: str

class StatsSummary(BaseModel):
    run_id: int
    field_max: float
    current_max: float
    duration_s: float
```

#### 3.3.3 `api/client.py`

```python
"""
REST client wrapping the python_magnetapi endpoints.

Uses httpx for async-capable transport. Falls back gracefully if
python_magnetapi or httpx is not installed.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class MagnetAPIClient:
    def __init__(self, base_url: str, api_key: str | None = None) -> None:
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx is required for API access. "
                "Install with: pip install python-magnetrun[api]"
            )
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(base_url=self._base_url, headers=self._headers)

    def get_run(self, run_id: int) -> "MagnetRunRecord":
        from python_magnetrun.api.models import MagnetRunRecord
        resp = self._client.get(f"/api/runs/{run_id}")
        resp.raise_for_status()
        return MagnetRunRecord.model_validate(resp.json())

    def list_runs(
        self,
        site: str | None = None,
        limit: int = 100,
    ) -> list["MagnetRunRecord"]:
        from python_magnetrun.api.models import MagnetRunRecord
        params: dict[str, Any] = {"limit": limit}
        if site:
            params["site"] = site
        resp = self._client.get("/api/runs/", params=params)
        resp.raise_for_status()
        return [MagnetRunRecord.model_validate(r) for r in resp.json()]

    def upload_run(self, run: "MagnetRun") -> int:
        """Serialize a MagnetRun and upload it; returns the assigned run_id."""
        payload = {
            "site": run.Housing,
            "insert": run.getInsert(),
            "starttime": run.Start.isoformat(),
            "stoptime": run.Stop.isoformat(),
        }
        resp = self._client.post("/api/runs/", json=payload)
        resp.raise_for_status()
        return resp.json()["id"]

    def get_stats(self, run_id: int) -> dict[str, Any]:
        resp = self._client.get(f"/api/runs/{run_id}/stats")
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "MagnetAPIClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
```

#### 3.3.4 `api/cli.py`

Minimal CLI subcommand for testing the connection:

```python
"""magnetrun-api CLI: interact with the python_magnetapi REST backend."""
import argparse
import json
import sys

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="magnetrun-api")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ls = sub.add_parser("list", help="List runs")
    ls.add_argument("--site", default=None)
    ls.add_argument("--limit", type=int, default=20)

    info = sub.add_parser("info", help="Show run details")
    info.add_argument("run_id", type=int)

    args = parser.parse_args(argv)

    from python_magnetrun.api.auth import APICredentials
    from python_magnetrun.api.client import MagnetAPIClient
    creds = APICredentials.from_env()

    with MagnetAPIClient(creds.base_url, creds.api_key) as client:
        if args.cmd == "list":
            runs = client.list_runs(site=args.site, limit=args.limit)
            for r in runs:
                print(f"{r.id:5d}  {r.site:<4}  {r.starttime}  {r.insert}")
        elif args.cmd == "info":
            run = client.get_run(args.run_id)
            print(json.dumps(run.model_dump(mode="json"), indent=2))
    return 0
```

#### 3.3.5 Register CLI entry point

```toml
# pyproject.toml
[project.scripts]
magnetrun-api = "python_magnetrun.api.cli:main"
```

#### 3.3.6 Add `api` optional dependency group

```toml
[project.optional-dependencies]
api = [
    "httpx>=0.27",
    "pydantic>=2.0",
]
```

**Acceptance criteria:**
- `from python_magnetrun.api import MagnetAPIClient` imports cleanly (no network needed).
- `MagnetAPIClient("http://localhost:8000")` can be instantiated.
- `magnetrun-api --help` works.
- With mock HTTP (using `respx` or `httpx` transport), `get_run()` and `list_runs()` return correct Pydantic models.

---

### 3.4 Add `MRecord.getData()` via API fallback

**File:** `python_magnetrun/MRecord.py`

Current `getData()` calls `python_magnetrun.fetchers.connect` (legacy scraping).
Add an API-first path:

```python
def getData(self) -> MagnetData:
    import os
    api_url = os.environ.get("MAGNETAPI_URL")
    if api_url:
        logger.info("Fetching run data via API for %s", self)
        from python_magnetrun.api.client import MagnetAPIClient
        from python_magnetrun.api.auth import APICredentials
        creds = APICredentials.from_env()
        with MagnetAPIClient(creds.base_url, creds.api_key) as client:
            return client.get_run_data(self.id)
    else:
        logger.debug("MAGNETAPI_URL not set; falling back to legacy scraping")
        return self._getData_legacy()
```

Keep the legacy scraping path in `_getData_legacy()` for backwards compatibility.

**Acceptance criteria:**
- When `MAGNETAPI_URL` is unset, behaviour is identical to current.
- When `MAGNETAPI_URL` is set, the API path is attempted (tested via mock).

---

### 3.5 Expand test coverage

New test files:

#### `tests/test_api_client.py`

```python
"""Tests for MagnetAPIClient using httpx mock transport."""
import pytest
from unittest.mock import patch
import json

# Use httpx.MockTransport or respx for network-free tests
def test_get_run_returns_model():
    ...

def test_list_runs_with_site_filter():
    ...

def test_upload_run_returns_int():
    ...

def test_credentials_from_env(monkeypatch):
    monkeypatch.setenv("MAGNETAPI_URL", "http://test.local")
    from python_magnetrun.api.auth import APICredentials
    creds = APICredentials.from_env()
    assert creds.base_url == "http://test.local"

def test_credentials_missing_raises(monkeypatch):
    monkeypatch.delenv("MAGNETAPI_URL", raising=False)
    from python_magnetrun.api.auth import APICredentials
    with pytest.raises(EnvironmentError, match="MAGNETAPI_URL"):
        APICredentials.from_env()
```

#### `tests/test_site_config.py`

```python
def test_load_sites_returns_all_known():
    from python_magnetrun.analysis.config import load_sites
    sites = load_sites()
    assert "M8" in sites
    assert "M9" in sites
    assert "M10" in sites

def test_get_site_config_m9():
    from python_magnetrun.analysis.config import get_site_config
    cfg = get_site_config("M9")
    assert "Idcct1" in cfg.channel_mapping.IH_channels

def test_unknown_site_raises():
    from python_magnetrun.analysis.config import get_site_config
    with pytest.raises(KeyError, match="M99"):
        get_site_config("M99")
```

#### `tests/test_serialization.py`

```python
def test_mrecord_roundtrip_json():
    from python_magnetrun.MRecord import MRecord
    # build a minimal valid record
    ...
    assert MRecord.from_json(r.to_json()) == r

def test_mrecord_model_dump():
    ...
    d = r.model_dump()
    assert isinstance(d, dict)
    assert "site" in d
```

**Acceptance criteria:**
- `pytest tests/ -x` passes.
- Coverage for `api/` module ≥ 60 % (via mocks).
- Coverage for `analysis/config.py` ≥ 70 %.

---

## Files Modified / Created This Phase

| File | Action |
|------|--------|
| `python_magnetrun/MRecord.py` | Pydantic BaseModel, API fallback in `getData()` |
| `python_magnetrun/GObject.py` | Pydantic BaseModel |
| `python_magnetrun/HMagnet.py` | Pydantic BaseModel |
| `python_magnetrun/deserialize.py` | Simplify / remove |
| `python_magnetrun/MagnetRun.py` | `prepareData()` reads from YAML config |
| `python_magnetrun/analysis/config.py` | `SiteConfig` + `ChannelMapping`, env-var paths |
| `data/sites.toml` | New (TOML, loaded via stdlib `tomllib`) |
| `python_magnetrun/api/__init__.py` | New |
| `python_magnetrun/api/client.py` | New |
| `python_magnetrun/api/models.py` | New |
| `python_magnetrun/api/auth.py` | New |
| `python_magnetrun/api/cli.py` | New |
| `pyproject.toml` | `api` extras, `pydantic`, `pyyaml`, `httpx`, entry point |
| `tests/test_api_client.py` | New |
| `tests/test_site_config.py` | New |
| `tests/test_serialization.py` | New |

---

## Verification Checklist

- [ ] `from python_magnetrun.api import MagnetAPIClient` imports cleanly.
- [ ] `magnetrun-api --help` works.
- [ ] `MRecord.to_json()` produces valid JSON; `MRecord.from_json(...)` round-trips correctly.
- [ ] `prepareData()` for M8, M9, M10 works from YAML.
- [ ] Adding a new site to `data/sites.toml` alone makes it usable — no code changes.
- [ ] No `/home/LNCMI-G/...` absolute paths anywhere in source.
- [ ] `pytest tests/ -x` passes.
- [ ] `ruff check python_magnetrun/` exits 0.
- [ ] `mypy python_magnetrun/ --ignore-missing-imports` exits 0.

---

## Dependencies

- **Requires Phase 2** complete: `fetchers/`, `magnetdata/` package, `protocols.py`.
- Phase 4 will build dashboards on top of the `MagnetData` and `MagnetAPIClient` introduced here.
