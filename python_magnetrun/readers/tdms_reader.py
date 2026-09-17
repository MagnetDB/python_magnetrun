"""TDMS reader — validation and format-specific configuration for pigbrother files."""

from __future__ import annotations

from pathlib import Path


class TdmsReader:
    """Reader configuration for pigbrother TDMS files.

    Holds the TDMS-specific constants (required group name, per-file t-offset
    table) that were previously hardcoded in :func:`magnetdata._fromtdms`.
    Lazy group loading (``_LazyGroupDict``) stays in the container
    (:class:`~python_magnetrun.magnetdata_tdms.TdmsMagnetData`) because it is
    data management, not parsing.

    Attributes
    ----------
    required_group : str
        Fallback group that must be present for a TDMS file to be valid
        (``"Courants_Alimentations"``).
    required_groups : tuple[str, ...]
        Groups of which at least one must be present for a TDMS file to be
        valid. Covers ``required_group`` (Overview/Archive/Default) plus the
        markers for each known Stats TDMS schema generation (``"Moy"`` for
        the 2019-2021 layout, ``"Stats_moy"`` for 2022+) — see
        :meth:`has_required_group`.
    t_offsets : dict[str, float]
        Map of filename substring → ``wf_start_offset`` override value [s].
    defs_file : str
        Default field-definition JSON file name.
    """

    required_group: str = "Courants_Alimentations"
    required_groups: tuple[str, ...] = (required_group, "Moy", "Stats_moy")
    t_offsets: dict[str, float] = {
        "Overview": 0.5,
        "Archive": 1 / 240.0,
    }
    defs_file: str = "pigbrother-defs.json"

    def has_required_group(self, groups: dict) -> bool:
        """Return whether *groups* contains at least one of :attr:`required_groups`.

        Checks presence rather than a single fixed name so that Stats TDMS
        files from any schema generation validate, without needing to infer
        the generation from the filename or date.

        Parameters
        ----------
        groups : dict
            Group names present in the loaded TDMS file (keys of ``Groups``
            in :func:`~python_magnetrun.magnetdata._fromtdms`).

        Returns
        -------
        bool
            ``True`` if any of :attr:`required_groups` is present.
        """
        return any(g in groups for g in self.required_groups)

    def t_offset_for(self, filename: str) -> float:
        """Return the ``wf_start_offset`` correction for *filename* [s].

        Parameters
        ----------
        filename : str
            TDMS file path (basename is inspected for known substrings).

        Returns
        -------
        float
            Offset correction in seconds; ``0.0`` when the filename matches
            none of the known patterns.
        """
        for substring, offset in self.t_offsets.items():
            if substring in filename:
                return offset
        return 0.0

    def validate(self, path: Path) -> bool:
        """Validate that *path* is a well-formed TDMS file.

        Delegates to :func:`~python_magnetrun.utils.validation.validate_tdms_format`
        which checks the four-byte magic number ``b"TDSm"`` at offset 0.

        Parameters
        ----------
        path : Path
            TDMS file to validate.

        Returns
        -------
        bool
            Always ``True`` on success; raises on failure.

        Raises
        ------
        python_magnetrun.utils.validation.FileFormatError
            If the magic bytes do not match.
        FileNotFoundError
            If *path* does not exist.
        """
        from ..utils.validation import validate_tdms_format

        validate_tdms_format(str(path))
        return True
