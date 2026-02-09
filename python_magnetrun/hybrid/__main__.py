"""
Entry point for running hybrid module as a script.

Usage:
    python -m python_magnetrun.hybrid --help
    python -m python_magnetrun.hybrid --base-dir /data/hybrid --date 2025-01-06
"""

from .cli import main

if __name__ == "__main__":
    main()
