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
        Group that must be present for a TDMS file to be valid
        (``"Courants_Alimentations"``).
    t_offsets : dict[str, float]
        Map of filename substring → ``wf_start_offset`` override value [s].
    defs_file : str
        Default field-definition JSON file name.
    """

    required_group: str = "Courants_Alimentations"
    t_offsets: dict[str, float] = {
        "Overview": 0.5,
        "Archive": 1 / 240.0,
    }
    defs_file: str = "pigbrother-defs.json"

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
