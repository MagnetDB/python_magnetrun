#! /usr/bin/python3

"""
Connect to the Control/Monitoring site
Retreive MagnetID list
For each MagnetID list of attached record
Check record consistency
"""

import logging
from typing import Any

import requests  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


def createSession(
    s: requests.Session,
    url_logging: str,
    payload: dict[str, Any],
    debug: bool | None = False,
) -> requests.Response:
    """create a request session"""

    p = s.post(url=url_logging, data=payload, verify=True)
    # print the html returned or something more intelligent to see if it's a successful login page.
    logger.debug(f"connect: {p.url}, status={p.status_code}")
    # check return status: if not ok stop
    if p.status_code != 200:
        raise RuntimeError(f"login failed: HTTP {p.status_code} at {url_logging}")
    return p


def download(
    session: requests.Session,
    url_data: str,
    param: dict[str, Any],
    link: str | None = None,
    debug: bool | None = False,
) -> str:
    """download"""

    # Build the query string manually to avoid double-encoding already-encoded
    # values like %20 in filenames (requests.get(params=...) would encode % -> %25)
    query = "&".join(f"{k}={v}" for k, v in param.items())
    full_url = f"{url_data}?{query}"
    d = session.get(url=full_url, verify=True)
    logger.debug(f"downloads: {d.url}, status={d.status_code}")
    if d.status_code != 200:
        raise RuntimeError(f"download failed: HTTP {d.status_code} from {url_data}")

    return d.text
