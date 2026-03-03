#! /usr/bin/python3

"""
Connect to the Control/Monitoring site
Retreive MagnetID list
For each MagnetID list of attached record
Check record consistency
"""

import logging

logger = logging.getLogger(__name__)


def createSession(s, url_logging: str, payload: dict, debug: bool | None = False):
    """create a request session"""

    p = s.post(url=url_logging, data=payload, verify=True)
    # print the html returned or something more intelligent to see if it's a successful login page.
    logger.debug(f"connect: {p.url}, status={p.status_code}")
    # check return status: if not ok stop
    if p.status_code != 200:
        raise RuntimeError(f"login failed: HTTP {p.status_code} at {url_logging}")
    return p


def download(
    session, url_data, param, link: str | None = None, debug: bool | None = False
):
    """download"""

    d = session.get(url=url_data, params=param, verify=True)
    logger.debug(f"downloads: {d.url}, status={d.status_code}")
    if d.status_code != 200:
        raise RuntimeError(f"download failed: HTTP {d.status_code} from {url_data}")

    return d.text
