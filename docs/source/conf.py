# Configuration file for the Sphinx documentation builder.
#
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from importlib.metadata import PackageNotFoundError, version

# Make the package importable for autodoc without installing it
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------

project = "python_magnetrun"
copyright = "2024, C. Trophime"
author = "C. Trophime"

try:
    release = version("python_magnetrun")
except PackageNotFoundError:
    release = "0.1.0"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.duration",
    "sphinxcontrib.mermaid",
]

# Auto-generate stub pages for autosummary directives
autosummary_generate = True

# Autodoc defaults: document all members including those without docstrings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# Mock heavy/optional imports so docs can build without them installed
autodoc_mock_imports = [
    "clawpack",
    "ht",
    "nlopt",
    "pwlf",
    "sklearn",
    "stumpy",
    "sympy",
    "ruptures",
    "piecewise_regression",
    "pyts",
    "dtaidistance",
    "pyextremes",
    "visidata",
    "xmltodict",
    "nptdms",
    "iapws",
]

# Cross-reference external documentation
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- EPUB output -------------------------------------------------------------

epub_show_urls = "footnote"
