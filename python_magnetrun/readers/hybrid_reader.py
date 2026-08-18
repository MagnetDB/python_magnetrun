"""HybridReader — composite reader that discovers kHz/rms/trigger data."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class HybridReader:
    """Composite reader for hybrid FEPC acquisition data.

    Discovers kHz binary files, RMS ``.rms`` files, and trigger directories
    under a base directory for a given recording date.  Delegates the actual
    binary parsing to the sub-readers already present in
    :mod:`python_magnetrun.hybrid`.

    This reader's primary role is validation and discovery — loading is
    performed lazily by :class:`~python_magnetrun.hybrid.hybrid_data.HybridData`.

    Attributes
    ----------
    defs_file : str
        Default field-definition JSON file name.
    """

    defs_file: str = "hybrid-defs.json"

    def read(self, base_dir: Path) -> dict[str, Any]:
        """Discover available sub-directories and return a metadata dict.

        Does **not** load any binary data.  Returns a lightweight discovery
        result that :class:`~python_magnetrun.hybrid.hybrid_data.HybridData`
        can use to populate its :attr:`Groups` and :attr:`Keys`.

        Parameters
        ----------
        base_dir : Path
            Root directory that contains ``kHz/``, ``rms/``, ``trigger/``
            sub-directories.

        Returns
        -------
        dict[str, Any]
            ``{"khz": {...}, "rms": {...}, "trigger": {...}}`` mapping of
            sub-type to lists of discovered paths.
        """
        base_dir = Path(base_dir)
        result: dict[str, Any] = {}

        khz_dir = base_dir / "kHz"
        if khz_dir.exists():
            result["khz"] = sorted(khz_dir.rglob("*.bin"))

        rms_dir = base_dir / "rms"
        if rms_dir.exists():
            result["rms"] = sorted(rms_dir.rglob("*.rms"))

        trigger_dir = base_dir / "trigger"
        if trigger_dir.exists():
            result["trigger"] = sorted(trigger_dir.iterdir())

        return result

    def validate(self, path: Path) -> bool:
        """Validate that *path* is a directory containing at least one sub-type.

        Parameters
        ----------
        path : Path
            Base directory to validate.

        Returns
        -------
        bool
            Always ``True`` on success; raises on failure.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist or is not a directory.
        ValueError
            If neither ``kHz/``, ``rms/``, nor ``trigger/`` sub-directories
            exist under *path*.
        """
        path = Path(path)
        if not path.is_dir():
            raise FileNotFoundError(
                f"HybridReader.validate: {path} is not a directory"
            )
        sub_dirs = {"kHz", "rms", "trigger"}
        if not any((path / s).exists() for s in sub_dirs):
            raise ValueError(
                f"HybridReader.validate: {path} contains none of {sub_dirs}"
            )
        return True
