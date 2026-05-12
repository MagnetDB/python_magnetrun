#!/usr/bin/env python

"""Smoke tests for the `python_magnetrun` package.

Verifies that the package is installed, exposes correct metadata, and that
core public-API symbols are importable. These are the first tests to fail
if the package installation or top-level imports are broken.
"""

import python_magnetrun


def test_version_is_set():
    assert isinstance(python_magnetrun.__version__, str)
    assert python_magnetrun.__version__ != "0.0.0", (
        "Package is not installed — run 'pip install -e .' first"
    )


def test_author_metadata():
    assert python_magnetrun.__author__
    assert python_magnetrun.__email__


def test_load_magnetdata_importable():
    from python_magnetrun.magnetdata import load_magnetdata

    assert callable(load_magnetdata)


def test_housing_config_importable():
    from python_magnetrun.housing_config import HousingConfig

    assert HousingConfig is not None


def test_magnetrun_importable():
    from python_magnetrun.MagnetRun import MagnetRun

    assert MagnetRun is not None
