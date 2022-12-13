#! /usr/bin/python3

"""
Connect to the Control/Monitoring site
Retreive MagnetID list
For each MagnetID list of attached record
Check record consistency
"""

import getpass
import sys
import os
import re
import datetime
import requests
import requests.exceptions
import lxml.html as lh

def createSession(s, url_logging, payload, debug=False):
    """create a request session"""

    p = s.post(url=url_logging, data=payload, verify=True)
    # print the html returned or something more intelligent to see if it's a successful login page.
    if debug:
        print( f"connect: {p.url}, status={p.status_code}" )
    # check return status: if not ok stop
    if p.status_code != 200:
        print(f"error {p.status_code} logging to {url_logging}" )
        sys.exit(1)
    p.raise_for_status()
    return p

def download(session, url_data, param, link=None, debug=False):
    """download """

    d = session.get(url=url_data, params=param, verify=True)
    if debug:
        print(f"downloads: {d.url}, status={d.status_code}")
    if d.status_code != 200:
        print(f"error {d.status_code} dowmload {url_data}" )
        sys.exit(1)
    d.raise_for_status()

    return d.text

