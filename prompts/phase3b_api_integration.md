# Prompt: Phase 3b — python_magnetapi Integration

## Context

`python_magnetrun` needs to interact with `python_magnetdb` (the database backend) through
`python_magnetapi` (a REST API). This prompt covers **Phase 3b** of the improvement plan:
creating a typed HTTP client, Pydantic models for API responses, a local cache, and the
ability to push analysis results back to the database.

Phase 3b runs **in parallel** with Phase 3. Both require Phase 2 to be complete.

Reference document: `IMPROVEMENT_PLAN.md` §Phase 3b.

---

## Objective

1. Create a `python_magnetrun/api/` subpackage with a typed `MagnetAPIClient`.
2. Define Pydantic models for all API responses.
3. Add a local Parquet cache so that downloaded runs are not re-fetched.
4. Make `MRecord.getData()` fall back to the API when `MAGNETAPI_URL` is set.
5. Add `MagnetRun.upload()` to push analysis results (stats, anomalies, signatures) back
   to the database.
6. Add a `magnetrun-api` CLI entry point for scripted interactions.

---

## Prerequisites

- `python_magnetapi` must be running at a known URL (or a mock URL for tests).
- Pydantic v2 is available (added in Phase 2 task 2.9).
- `httpx` must be added to dependencies (replaces `requests` for async-capable HTTP).

Add to `pyproject.toml`:

```toml
[project.dependencies]
# add alongside existing deps:
httpx = ">=0.27"

[project.optional-dependencies]
api = [
    "httpx[http2]>=0.27",
    "pydantic>=2.0",
]
```

---

## Task 3b.1 — Create `python_magnetrun/api/` subpackage

```
python_magnetrun/api/
├── __init__.py      # exports MagnetAPIClient
├── client.py        # MagnetAPIClient class
├── models.py        # Pydantic response models
├── auth.py          # credential management
├── cache.py         # local Parquet cache
└── cli.py           # magnetrun-api CLI subcommands
```

### `api/__init__.py`

```python
"""
python_magnetrun API client for python_magnetapi.

Usage::

    from python_magnetrun.api import MagnetAPIClient

    client = MagnetAPIClient()  # reads MAGNETAPI_URL and MAGNETAPI_KEY from env
    runs = client.list_runs(site="M9")
    run = client.get_run_data(runs[0].id)
"""
from python_magnetrun.api.client import MagnetAPIClient

__all__ = ["MagnetAPIClient"]
```

---

## Task 3b.2 — Define Pydantic models in `api/models.py`

These models mirror the `python_magnetapi` REST API response schema.
Adjust field names once the actual API schema is known — use these as the starting point.

```python
"""
Pydantic models for python_magnetapi REST API responses.

Field names follow the API schema. Update as the API evolves.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class MagnetRunRecord(BaseModel):
    """A single experimental run record as returned by /api/runs/{id}."""

    id: int
    site: str                          # e.g. "M9"
    housing: str                       # e.g. "M9_Coil1"
    start_time: datetime
    end_time: datetime | None = None
    duration_s: float | None = None    # seconds
    file_url: str | None = None        # URL to download raw data file
    status: str = "unknown"            # "ok", "incident", "incomplete"

    @field_validator("site")
    @classmethod
    def site_must_be_known(cls, v: str) -> str:
        # Soft validation — warn but don't reject unknown sites
        known = {"M8", "M9", "M10", "M1", "M5", "M7"}
        if v not in known:
            import logging
            logging.getLogger(__name__).warning("Unknown site in API response: %r", v)
        return v


class RunStats(BaseModel):
    """Statistical summary of a run, uploadable via POST /api/runs/{id}/stats."""

    run_id: int
    field_max: float | None = None      # Tesla
    ih_mean: float | None = None        # Amperes
    ib_mean: float | None = None        # Amperes
    plateau_count: int = 0
    anomaly_count: int = 0
    duration_s: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class Anomaly(BaseModel):
    """A detected anomaly event, uploadable via POST /api/runs/{id}/anomalies."""

    run_id: int
    t_start: float                      # seconds from run start
    t_end: float
    channel: str
    severity: str = "warning"           # "info", "warning", "critical"
    description: str = ""


class SignatureRecord(BaseModel):
    """A detected operational signature (U/P/D regimes)."""

    run_id: int
    name: str
    symbol: str                         # "U", "P", "D"
    t_start: float
    t_end: float
    value: float | None = None
    unit: str = ""
```

---

## Task 3b.3 — Implement `MagnetAPIClient` in `api/client.py`

```python
"""
Typed HTTP client for the python_magnetapi REST API.

All API requests are synchronous (httpx in sync mode). Use the async variant
(httpx.AsyncClient) if you need to fetch many runs in parallel.

Configuration via environment variables:
    MAGNETAPI_URL : str   Base URL (default: http://localhost:8000)
    MAGNETAPI_KEY : str   Bearer token (optional)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import httpx

from python_magnetrun.api.models import (
    Anomaly,
    MagnetRunRecord,
    RunStats,
    SignatureRecord,
)

logger = logging.getLogger(__name__)

_DEFAULT_URL = "http://localhost:8000"
_DEFAULT_TIMEOUT = 30.0   # seconds


class MagnetAPIClient:
    """
    Client for the python_magnetapi REST API.

    Parameters
    ----------
    base_url : str, optional
        Base URL of the API server. Defaults to the MAGNETAPI_URL environment
        variable, then http://localhost:8000.
    api_key : str or None, optional
        Bearer token for authentication. Defaults to the MAGNETAPI_KEY env var.
    cache_dir : Path or None, optional
        Directory for caching downloaded run data as Parquet files.
        Defaults to ~/.cache/magnetrun. Pass None to disable caching.
    timeout : float, optional
        HTTP request timeout in seconds (default 30).

    Examples
    --------
    >>> client = MagnetAPIClient()
    >>> runs = client.list_runs(site="M9", limit=10)
    >>> run = client.get_run_data(runs[0].id)
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        cache_dir: Path | None = Path.home() / ".cache" / "magnetrun",
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self.base_url = (base_url or os.environ.get("MAGNETAPI_URL", _DEFAULT_URL)).rstrip("/")
        self._api_key = api_key or os.environ.get("MAGNETAPI_KEY")
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _get(self, path: str, **params: Any) -> Any:
        url = f"{self.base_url}{path}"
        logger.debug("GET %s params=%s", url, params)
        response = httpx.get(url, headers=self._headers(), params=params,
                             timeout=self._timeout)
        response.raise_for_status()
        return response.json()

    def _post(self, path: str, data: Any) -> Any:
        url = f"{self.base_url}{path}"
        logger.debug("POST %s", url)
        response = httpx.post(url, headers=self._headers(),
                              json=data, timeout=self._timeout)
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # Run listing and metadata
    # ------------------------------------------------------------------

    def list_runs(
        self,
        site: str | None = None,
        housing: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MagnetRunRecord]:
        """
        List experimental run records.

        Parameters
        ----------
        site : str, optional
            Filter by site (e.g., "M9").
        housing : str, optional
            Filter by housing identifier.
        limit : int
            Maximum number of records to return.
        offset : int
            Pagination offset.

        Returns
        -------
        list[MagnetRunRecord]
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if site:
            params["site"] = site
        if housing:
            params["housing"] = housing
        raw = self._get("/api/runs/", **params)
        return [MagnetRunRecord.model_validate(r) for r in raw.get("results", raw)]

    def get_run(self, run_id: int) -> MagnetRunRecord:
        """
        Fetch metadata for a single run.

        Parameters
        ----------
        run_id : int
            Run identifier.

        Returns
        -------
        MagnetRunRecord
        """
        raw = self._get(f"/api/runs/{run_id}/")
        return MagnetRunRecord.model_validate(raw)

    # ------------------------------------------------------------------
    # Run data (with local cache)
    # ------------------------------------------------------------------

    def get_run_data(self, run_id: int) -> "MagnetRun":
        """
        Download run data and return a MagnetRun object.

        Downloads the raw data file from the API, caches it locally as
        Parquet, and returns a MagnetRun object loaded from the cache.

        Parameters
        ----------
        run_id : int
            Run identifier.

        Returns
        -------
        MagnetRun
        """
        from python_magnetrun import MagnetRun

        # Check cache first
        if self.cache_dir is not None:
            cached = self.cache_dir / f"run_{run_id}.parquet"
            if cached.exists():
                logger.info("Loading run %d from cache: %s", run_id, cached)
                return MagnetRun.from_parquet(str(cached))

        # Fetch metadata to get file URL
        record = self.get_run(run_id)
        if not record.file_url:
            raise ValueError(f"Run {run_id} has no file_url in API response")

        # Download the raw file
        logger.info("Downloading run %d from %s", run_id, record.file_url)
        response = httpx.get(record.file_url, headers=self._headers(),
                             timeout=self._timeout, follow_redirects=True)
        response.raise_for_status()

        # Save to a temp file with the right extension
        suffix = Path(record.file_url).suffix or ".txt"
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        try:
            run = MagnetRun.from_file(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        # Cache as Parquet for future use
        if self.cache_dir is not None:
            cached = self.cache_dir / f"run_{run_id}.parquet"
            run.saveData(str(cached), fmt="parquet")
            logger.info("Cached run %d to %s", run_id, cached)

        return run

    # ------------------------------------------------------------------
    # Uploading analysis results
    # ------------------------------------------------------------------

    def post_stats(self, run_id: int, stats: dict[str, Any]) -> None:
        """
        Upload statistical summary for a run.

        Parameters
        ----------
        run_id : int
            Run identifier.
        stats : dict
            Statistics dictionary (output of MagnetRun.getStats() or similar).
        """
        payload = RunStats(run_id=run_id, extra=stats)
        self._post(f"/api/runs/{run_id}/stats/", payload.model_dump())
        logger.info("Uploaded stats for run %d", run_id)

    def post_anomalies(
        self,
        run_id: int,
        anomalies: list[dict[str, Any]],
    ) -> None:
        """
        Upload detected anomalies for a run.

        Parameters
        ----------
        run_id : int
            Run identifier.
        anomalies : list[dict]
            List of anomaly dicts. Each must have: t_start, t_end, channel.
        """
        payload = [Anomaly(run_id=run_id, **a).model_dump() for a in anomalies]
        self._post(f"/api/runs/{run_id}/anomalies/", payload)
        logger.info("Uploaded %d anomalies for run %d", len(payload), run_id)

    def post_signatures(
        self,
        run_id: int,
        signatures: list[dict[str, Any]],
    ) -> None:
        """
        Upload operational signatures (U/P/D regimes) for a run.

        Parameters
        ----------
        run_id : int
        signatures : list[dict]
            Each dict must have: name, symbol, t_start, t_end.
        """
        payload = [SignatureRecord(run_id=run_id, **s).model_dump() for s in signatures]
        self._post(f"/api/runs/{run_id}/signatures/", payload)
        logger.info("Uploaded %d signatures for run %d", len(payload), run_id)
```

---

## Task 3b.4 — Credential management in `api/auth.py`

```python
"""
Credential management for the python_magnetapi client.

Credentials are resolved in this order:
1. Arguments passed directly to MagnetAPIClient().
2. Environment variables: MAGNETAPI_URL, MAGNETAPI_KEY.
3. Config file: ~/.config/magnetrun/api.toml (optional).
"""
from __future__ import annotations

import os
from pathlib import Path


def load_credentials() -> dict[str, str | None]:
    """
    Load API credentials from environment and config file.

    Returns
    -------
    dict with keys: 'url', 'key'
    """
    # Environment variables take priority
    url = os.environ.get("MAGNETAPI_URL")
    key = os.environ.get("MAGNETAPI_KEY")

    if url and key:
        return {"url": url, "key": key}

    # Fall back to config file
    config_file = Path.home() / ".config" / "magnetrun" / "api.toml"
    if config_file.exists():
        import tomllib
        with open(config_file, "rb") as f:
            cfg = tomllib.load(f)
        api_cfg = cfg.get("api", {})
        url = url or api_cfg.get("url")
        key = key or api_cfg.get("key")

    return {"url": url, "key": key}
```

Document the config file format in `README.md`:

```toml
# ~/.config/magnetrun/api.toml
[api]
url = "https://magnetdb.lncmi.cnrs.fr"
key = "your-api-token-here"
```

---

## Task 3b.5 — Update `MRecord.getData()` to use API when available

**File:** `python_magnetrun/MRecord.py`

**Current behaviour:** Always scrapes `fetchers/connect.py` (legacy srv-data endpoint).

**New behaviour:** Check for `MAGNETAPI_URL`; if set, use `MagnetAPIClient.get_run_data()`.
Fall back to the legacy scraper for backwards compatibility.

```python
def getData(self) -> "MagnetRun":
    """
    Download run data.

    Uses python_magnetapi if MAGNETAPI_URL is configured,
    otherwise falls back to the legacy srv-data scraper.
    """
    api_url = os.environ.get("MAGNETAPI_URL")
    if api_url and self.id is not None:
        from python_magnetrun.api import MagnetAPIClient
        client = MagnetAPIClient(base_url=api_url)
        logger.info("Fetching run %s via API at %s", self.id, api_url)
        return client.get_run_data(self.id)

    # Legacy path
    logger.info("Fetching run via legacy scraper (MAGNETAPI_URL not set)")
    from python_magnetrun.fetchers.connect import download_run
    return download_run(self.link)
```

**Note:** This requires `MRecord` to store an optional `id` field mapping to the
`python_magnetdb` run ID. Add `id: int | None = None` to the model if not already present.

---

## Task 3b.6 — Add `MagnetRun.upload()` method

**File:** `python_magnetrun/MagnetRun.py`

```python
def upload(self, client: "MagnetAPIClient") -> int:
    """
    Upload this run's analysis results to python_magnetdb.

    Uploads statistics, detected anomalies, and operational signatures.
    The run must already exist in the database (i.e., it was fetched via
    the API or was previously registered). The run_id must be set.

    Parameters
    ----------
    client : MagnetAPIClient
        Authenticated API client.

    Returns
    -------
    int
        The run_id confirmed by the server.

    Raises
    ------
    ValueError
        If this run has no associated database ID.
    """
    if not hasattr(self, "run_id") or self.run_id is None:
        raise ValueError(
            "Cannot upload: MagnetRun has no run_id. "
            "Either fetch the run via the API or set run_id manually."
        )

    # Upload stats
    stats = self.getStats()
    client.post_stats(self.run_id, stats)

    # Upload anomalies if available
    if hasattr(self, "_anomalies") and self._anomalies:
        client.post_anomalies(self.run_id, self._anomalies)

    # Upload signatures if available
    if hasattr(self, "_signatures") and self._signatures:
        sigs = [
            {
                "name": s.name,
                "symbol": s.symbol,
                "t_start": s.t0,
                "t_end": s.t0 + s.times[-1] if s.times else s.t0,
            }
            for s in self._signatures
        ]
        client.post_signatures(self.run_id, sigs)

    logger.info("Uploaded run %d results to magnetdb", self.run_id)
    return self.run_id
```

---

## Task 3b.7 — `magnetrun-api` CLI entry point

**File:** `python_magnetrun/api/cli.py`

```python
"""CLI for interacting with the python_magnetapi REST API."""
from __future__ import annotations

import argparse
import json
import logging
import sys

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="magnetrun-api",
        description="Interact with the python_magnetapi REST API",
    )
    parser.add_argument("--url", help="API base URL (overrides MAGNETAPI_URL)")
    parser.add_argument("--key", help="API key (overrides MAGNETAPI_KEY)")
    parser.add_argument("-v", "--verbose", action="store_true")

    sub = parser.add_subparsers(dest="command", required=True)

    # list-runs
    p_list = sub.add_parser("list-runs", help="List experimental runs")
    p_list.add_argument("--site", help="Filter by site (M8, M9, M10)")
    p_list.add_argument("--limit", type=int, default=20)

    # get-run
    p_get = sub.add_parser("get-run", help="Download a run and print summary")
    p_get.add_argument("run_id", type=int)

    # upload-stats
    p_up = sub.add_parser("upload-stats", help="Upload stats from a local run file")
    p_up.add_argument("run_id", type=int)
    p_up.add_argument("file", help="Path to the run data file")

    return parser


def main(argv: list[str] | None = None) -> int:
    from python_magnetrun.api import MagnetAPIClient

    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)
    client = MagnetAPIClient(
        base_url=args.url or None,
        api_key=args.key or None,
    )

    if args.command == "list-runs":
        runs = client.list_runs(site=args.site, limit=args.limit)
        for r in runs:
            print(f"{r.id:6d}  {r.site:<4}  {r.housing:<20}  {r.start_time}")

    elif args.command == "get-run":
        run = client.get_run_data(args.run_id)
        print(f"Keys: {run.getKeys()}")
        print(f"Stats: {json.dumps(run.getStats(), indent=2)}")

    elif args.command == "upload-stats":
        from python_magnetrun import MagnetRun
        run = MagnetRun.from_file(args.file)
        run.run_id = args.run_id
        run.upload(client)
        print(f"Uploaded stats for run {args.run_id}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Add the entry point to `pyproject.toml`:

```toml
[project.scripts]
magnetrun-api = "python_magnetrun.api.cli:main"
```

---

## Task 3b.8 — Tests for the API client

**File:** `tests/test_api_client.py`

Use `respx` (a httpx mock library) to test without a live server:

```toml
[project.optional-dependencies]
dev = [
    ...,
    "respx>=0.21",
]
```

```python
"""Tests for MagnetAPIClient using httpx mocking."""
import pytest
import respx
import httpx
from datetime import datetime

from python_magnetrun.api.client import MagnetAPIClient
from python_magnetrun.api.models import MagnetRunRecord, RunStats


@pytest.fixture
def client():
    return MagnetAPIClient(base_url="http://testserver", api_key=None, cache_dir=None)


@respx.mock
def test_list_runs(client):
    respx.get("http://testserver/api/runs/").mock(
        return_value=httpx.Response(200, json=[
            {"id": 1, "site": "M9", "housing": "H1",
             "start_time": "2024-01-01T10:00:00", "status": "ok"},
        ])
    )
    runs = client.list_runs(site="M9")
    assert len(runs) == 1
    assert runs[0].id == 1
    assert runs[0].site == "M9"


@respx.mock
def test_get_run(client):
    respx.get("http://testserver/api/runs/42/").mock(
        return_value=httpx.Response(200, json={
            "id": 42, "site": "M9", "housing": "H1",
            "start_time": "2024-01-01T10:00:00", "status": "ok",
        })
    )
    record = client.get_run(42)
    assert record.id == 42


@respx.mock
def test_post_stats(client):
    respx.post("http://testserver/api/runs/42/stats/").mock(
        return_value=httpx.Response(201, json={"status": "created"})
    )
    client.post_stats(42, {"field_max": 35.0, "plateau_count": 3})
    # No exception = success


@respx.mock
def test_post_anomalies(client):
    respx.post("http://testserver/api/runs/42/anomalies/").mock(
        return_value=httpx.Response(201, json=[])
    )
    client.post_anomalies(42, [
        {"t_start": 10.0, "t_end": 12.0, "channel": "IH", "severity": "warning"}
    ])


def test_client_reads_env(monkeypatch):
    monkeypatch.setenv("MAGNETAPI_URL", "http://envserver")
    monkeypatch.setenv("MAGNETAPI_KEY", "mytoken")
    c = MagnetAPIClient()
    assert c.base_url == "http://envserver"
    assert c._api_key == "mytoken"


def test_client_unknown_extension_in_run_raises(client, monkeypatch):
    """get_run_data raises if file_url has unrecognised format."""
    import respx, httpx
    with respx.mock:
        respx.get("http://testserver/api/runs/99/").mock(
            return_value=httpx.Response(200, json={
                "id": 99, "site": "M9", "housing": "H1",
                "start_time": "2024-01-01T10:00:00",
                "file_url": "http://testserver/files/run.xyz",
            })
        )
        with pytest.raises(Exception):
            client.get_run_data(99)
```

---

## Verification Checklist

```bash
# 1. Module imports cleanly without a live server
python -c "from python_magnetrun.api import MagnetAPIClient; print('OK')"

# 2. CLI help works
magnetrun-api --help
magnetrun-api list-runs --help

# 3. Tests pass (with mocked HTTP)
pytest tests/test_api_client.py -v

# 4. Credential precedence works
MAGNETAPI_URL=http://example.com python -c "
from python_magnetrun.api import MagnetAPIClient
c = MagnetAPIClient()
print(c.base_url)  # should print http://example.com
"

# 5. MRecord.getData() falls back to legacy scraper when env not set
python -c "
import os
os.environ.pop('MAGNETAPI_URL', None)
from python_magnetrun.MRecord import MRecord
# Instantiate without actually downloading
print('import OK')
"

# 6. Pydantic models validate correctly
python -c "
from python_magnetrun.api.models import MagnetRunRecord
from datetime import datetime
r = MagnetRunRecord(id=1, site='M9', housing='H1', start_time=datetime.now())
print(r.model_dump_json())
"
```

---

## Commit Strategy

```
feat(api): add python_magnetrun/api/ subpackage skeleton
feat(api/models): Pydantic models for MagnetRunRecord, RunStats, Anomaly, SignatureRecord
feat(api/client): MagnetAPIClient with list_runs, get_run, get_run_data, post_stats
feat(api/auth): credential resolution from env and ~/.config/magnetrun/api.toml
feat(api/cache): local Parquet cache for downloaded runs
feat(MRecord): fall back to API when MAGNETAPI_URL is set
feat(MagnetRun): add upload() method for pushing results to magnetdb
feat(api/cli): magnetrun-api CLI entry point
test(api): mock-based tests with respx
```
