"""Top-level package for Python MagnetRun."""

from importlib.metadata import PackageNotFoundError, version

__author__ = """Christophe Trophime"""
__email__ = "christophe.trophime@lncmi.cnrs.fr"

try:
    __version__ = version("python_magnetrun")
except PackageNotFoundError:
    __version__ = "0.0.0"  # fallback when package is not installed
