"""Shared helpers for data-command _run() functions."""

from __future__ import annotations

import logging
import os
import traceback

from natsort import natsorted

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = [".txt", ".tdms", ".csv"]


def load_inputs(args):
    """Load MagnetRun objects from args.input_file.

    Returns
    -------
    (input_files, inputs, extensions)
        input_files : list[str] — ordered file paths
        inputs : dict[str, dict] — {path: {"data": MagnetRun}}
        extensions : dict[str, list[int]] — extension -> indices into input_files
    """
    import sys

    from ..cli_args import get_datadir_mapping
    from ..log_utils import format_exception_location
    from ..MagnetRun import MagnetRun
    from ..utils.files import expand_input_files

    datadir = get_datadir_mapping(args)
    expanded_files = expand_input_files(
        args.input_file, datadir, housing=getattr(args, "housing", None)
    )

    ext_groups: dict[str, list[str]] = {}
    for f in expanded_files:
        ext = os.path.splitext(f)[-1]
        ext_groups.setdefault(ext, []).append(f)
    for ext in ext_groups:
        ext_groups[ext] = natsorted(ext_groups[ext])

    input_files: list[str] = [f for files in ext_groups.values() for f in files]

    extensions: dict[str, list[int]] = {}
    for i, f in enumerate(input_files):
        ext = os.path.splitext(f)[-1]
        extensions.setdefault(ext, []).append(i)

    inputs: dict = {}
    for file in input_files:
        f_extension = os.path.splitext(file)[-1]
        if f_extension not in SUPPORTED_FORMATS:
            raise RuntimeError(
                f"unsupported extension '{f_extension}'; supported: {SUPPORTED_FORMATS}"
            )

        filename = os.path.basename(file)
        housing = getattr(args, "housing", None) or "notdefined"
        assembly = getattr(args, "assembly", None) or "notdefined"

        if filename.startswith("M"):
            try:
                idx = filename.index("_")
                if housing == "notdefined":
                    housing = filename[:idx]
            except ValueError:
                logger.warning(f"{file}: no housing detected — use --housing")
                continue

        try:
            match f_extension:
                case ".txt":
                    mrun = MagnetRun.fromtxt(housing=housing, assembly=assembly, filename=file)
                case ".tdms":
                    mrun = MagnetRun.fromtdms(housing=housing, assembly=assembly, filename=file)
                case ".csv":
                    mrun = MagnetRun.fromcsv(housing=housing, assembly=assembly, filename=file)
                case _:
                    raise RuntimeError(f"unhandled extension '{f_extension}'")
        except (OSError, ValueError, RuntimeError) as error:
            tb_str = "".join(traceback.format_exception(*sys.exc_info()))
            logger.error(f"{file}: load error at {format_exception_location()}")
            logger.error(f"Error: {error}")
            logger.debug(f"Traceback:\n{tb_str}")
            continue

        mrun.setHousing(housing)
        inputs[file] = {"data": mrun}

    return input_files, inputs, extensions
