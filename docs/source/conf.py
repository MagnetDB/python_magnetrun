# Configuration file for the Sphinx documentation builder.
#
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import re
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

# Render NumPy "Attributes" sections as a field list: as attribute directives they
# duplicate the dataclass fields that autodoc already documents (undoc-members).
napoleon_use_ivar = True
# Non-standard docstring sections, rendered as generic sections.
napoleon_custom_sections = ["Timestamp convention"]

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

# -- Re-exported members -----------------------------------------------------

# Modules listed in the autosummary directives of api.md.
with open(os.path.join(os.path.dirname(__file__), "api.md"), encoding="utf-8") as _api:
    _DOCUMENTED_MODULES = set(re.findall(r"^\s+(python_magnetrun[\w.]*)\s*$", _api.read(), re.M))


def _skip_reexported(app, what, name, obj, skip, options):
    """Skip a module member re-exported from another documented module.

    Documenting it twice (e.g. ``MagnetDataBase`` in both ``magnetdata`` and
    ``magnetdata_base``) makes cross-references ambiguous. Re-exports whose
    home module is not documented are kept.
    """
    if what != "module":
        return None
    home = getattr(obj, "__module__", None)
    current = app.env.temp_data.get("autodoc:module")
    if home != current and home in _DOCUMENTED_MODULES:
        return True
    return None


def setup(app):
    app.connect("autodoc-skip-member", _skip_reexported)
