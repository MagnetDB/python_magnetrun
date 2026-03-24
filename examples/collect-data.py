#!/usr/bin/env python3
"""
Collect pbsurv and srv-data-install files for a given housing between two dates.
For M8, also collects CEA/kHz, CEA/rms, CEA/vprocess and CEA/trigger directories.

Usage:
    python collect-data.py --housing M8 --start 2024-01-01 --end 2024-12-31
    python collect-data.py --housing M9 --start 2023-06-01 --end 2023-12-31 --output results.txt
"""

import re
import sys
import shutil
import tarfile
import argparse
from datetime import date, datetime
from pathlib import Path

BASE_DIR = Path("/home/LNCMI-G/christophe.trophime/LNCMIG-Data")
PBSURV_DIR = BASE_DIR / "pbsurv"
SRV_INSTALL_DIR = BASE_DIR / "srv-data-install"
CEA_DIR = BASE_DIR / "CEA"

# Variants observed:
#   *_YYMMDD-hhmm.tdms               (Archive, Overview, FFT, …)
#   *_YYMMDD-HHmmss.tdms             (Default, 6-digit time)
#   *_YYMMDD-HHmmss_SomeSuffix.tdms  (Default with label)
PBSURV_RE = re.compile(r"_(\d{2})(\d{2})(\d{2})-\d{4,6}(?:_[^.]+)?\.tdms$", re.IGNORECASE)
# YYYY.MM.DD - hh:mm:ss.txt  (seconds field may be 1 or 2 digits)
SRV_RE = re.compile(r"^(\d{4})\.(\d{2})\.(\d{2}) - .+\.txt$")
# TRIGGER__YYYY-MM-DD__HH-MM
TRIGGER_RE = re.compile(r"^TRIGGER__(\d{4}-\d{2}-\d{2})__\d{2}-\d{2}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect LNCMIG data files for a housing between two dates."
    )
    parser.add_argument(
        "--housing", required=True, help="Housing name, e.g. M1 … M10"
    )
    parser.add_argument(
        "--start", required=True, metavar="YYYY-MM-DD", help="Start date (inclusive)"
    )
    parser.add_argument(
        "--end", required=True, metavar="YYYY-MM-DD", help="End date (inclusive)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Write file list to this path instead of stdout"
    )
    parser.add_argument(
        "--copy-to", default=None, metavar="DEST_DIR",
        help="Copy all found files/dirs into DEST_DIR (directory will be created)"
    )
    parser.add_argument(
        "--archive", default=None, metavar="ARCHIVE.tar.gz",
        help="Archive all found files/dirs into a tar.gz file with paths relative to LNCMIG-Data"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def in_range(d: date, start: date, end: date) -> bool:
    return start <= d <= end


def parse_ymd(name: str) -> date | None:
    """Parse a YYYY-MM-DD string into a date, or return None."""
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", name)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    return None


# ---------------------------------------------------------------------------
# pbsurv .tdms files
# ---------------------------------------------------------------------------

def _check_tdms(path: Path, start: date, end: date) -> date | None:
    """Return file date if it matches the pbsurv naming pattern and is in range."""
    m = PBSURV_RE.search(path.name)
    if not m:
        return None
    yy, mm, dd = int(m.group(1)), int(m.group(2)), int(m.group(3))
    try:
        d = date(2000 + yy, mm, dd)
    except ValueError:
        return None
    return d if in_range(d, start, end) else None


def get_pbsurv_files(housing: str, start: date, end: date) -> list[Path]:
    """
    Collect pbsurv .tdms files for *housing* within [start, end].

    Searches in:
      - PBSURV_DIR/{housing}/{Fichiers_*,Overview,Model,Models}/   (current main dirs)
      - PBSURV_DIR/Fichiers_Data_ACQ_ENET_{Y}/{housing}/Fichiers_*/  (older yearly dirs)
    """
    collected: list[Path] = []

    TDMS_SUBDIRS = {"Fichiers_", "Overview", "Model", "Models"}

    def scan_housing_dir(hdir: Path) -> None:
        if not hdir.is_dir():
            return
        for subdir in hdir.iterdir():
            if not subdir.is_dir():
                continue
            if subdir.name.startswith("Fichiers_") or subdir.name in TDMS_SUBDIRS:
                for f in subdir.rglob("*.tdms"):
                    if _check_tdms(f, start, end) is not None:
                        collected.append(f)

    # Current main directory
    scan_housing_dir(PBSURV_DIR / housing)

    # Older yearly directories
    for entry in PBSURV_DIR.iterdir():
        if entry.is_dir() and entry.name.startswith("Fichiers_Data_ACQ_ENET_"):
            scan_housing_dir(entry / housing)

    return sorted(collected)


# ---------------------------------------------------------------------------
# srv-data-install files
# ---------------------------------------------------------------------------

def get_srv_install_files(housing: str, start: date, end: date) -> list[Path]:
    """
    Collect srv-data-install .txt files matching 'YYYY.MM.DD - *.txt'
    for *housing* within [start, end].
    """
    collected: list[Path] = []
    hdir = SRV_INSTALL_DIR / housing
    if not hdir.is_dir():
        return collected

    for f in hdir.iterdir():
        if not f.is_file():
            continue
        m = SRV_RE.match(f.name)
        if not m:
            continue
        try:
            d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            continue
        if in_range(d, start, end):
            collected.append(f)

    return sorted(collected)


# ---------------------------------------------------------------------------
# CEA helpers (M8 only)
# ---------------------------------------------------------------------------

def _get_cea_dated_dirs(base: Path, start: date, end: date) -> list[Path]:
    """
    Collect dated sub-directories of *base* that fall within [start, end].

    Two layouts are handled:
      - base/{YYYY}/{YYYY-MM-DD}/   (pre-2025 style)
      - base/{YYYY-MM-DD}/          (2025+ flat style)
    """
    dirs: list[Path] = []
    if not base.is_dir():
        return dirs

    for entry in base.iterdir():
        if not entry.is_dir():
            continue
        d = parse_ymd(entry.name)
        if d is not None:
            # Flat YYYY-MM-DD directly under base
            if in_range(d, start, end):
                dirs.append(entry)
        elif re.fullmatch(r"\d{4}", entry.name):
            # Year directory — look one level deeper
            for sub in entry.iterdir():
                if not sub.is_dir():
                    continue
                d2 = parse_ymd(sub.name)
                if d2 is not None and in_range(d2, start, end):
                    dirs.append(sub)

    return sorted(dirs)


def get_cea_trigger_dirs(start: date, end: date) -> list[Path]:
    """
    Collect CEA/trigger directories whose date falls within [start, end].
    Directory name pattern: TRIGGER__YYYY-MM-DD__HH-MM
    """
    dirs: list[Path] = []
    trigger_base = CEA_DIR / "trigger"
    if not trigger_base.is_dir():
        return dirs

    for entry in trigger_base.iterdir():
        if not entry.is_dir():
            continue
        m = TRIGGER_RE.match(entry.name)
        if not m:
            continue
        d = parse_ymd(m.group(1))
        if d is not None and in_range(d, start, end):
            dirs.append(entry)

    return sorted(dirs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    try:
        start = datetime.strptime(args.start, "%Y-%m-%d").date()
        end = datetime.strptime(args.end, "%Y-%m-%d").date()
    except ValueError as exc:
        sys.exit(f"Invalid date format: {exc}")

    if start > end:
        sys.exit("Error: --start must not be after --end")

    housing = args.housing.upper()

    # ---- collect ----
    results: dict[str, list[Path]] = {}

    results["pbsurv"] = get_pbsurv_files(housing, start, end)
    results["srv-data-install"] = get_srv_install_files(housing, start, end)

    if housing == "M8":
        results["CEA/kHz"] = _get_cea_dated_dirs(CEA_DIR / "kHz", start, end)
        results["CEA/rms"] = _get_cea_dated_dirs(CEA_DIR / "rms", start, end)
        results["CEA/vprocess"] = _get_cea_dated_dirs(CEA_DIR / "vprocess", start, end)
        results["CEA/trigger"] = get_cea_trigger_dirs(start, end)

    # ---- report ----
    lines: list[str] = []
    lines.append(f"Housing : {housing}")
    lines.append(f"Period  : {start} → {end}")
    lines.append("")

    for category, paths in results.items():
        lines.append(f"[{category}]  ({len(paths)} item(s))")
        for p in paths:
            lines.append(f"  {p}")
        lines.append("")

    report = "\n".join(lines)

    if args.output:
        Path(args.output).write_text(report)
        print(f"Results written to {args.output}")
    else:
        print(report)

    # ---- optional copy ----
    if args.copy_to:
        dest = Path(args.copy_to)
        dest.mkdir(parents=True, exist_ok=True)
        total = 0
        for category, paths in results.items():
            cat_dest = dest / category.replace("/", "_")
            cat_dest.mkdir(parents=True, exist_ok=True)
            for p in paths:
                target = cat_dest / p.name
                if p.is_file():
                    shutil.copy2(p, target)
                    total += 1
                elif p.is_dir():
                    shutil.copytree(p, cat_dest / p.name, dirs_exist_ok=True)
                    total += 1
        print(f"Copied {total} item(s) to {dest}")

    # ---- optional archive ----
    if args.archive:
        archive_path = Path(args.archive)
        with tarfile.open(archive_path, "w:gz") as tar:
            for paths in results.values():
                for p in paths:
                    arcname = p.relative_to(BASE_DIR.parent)
                    tar.add(p, arcname=arcname)
        total = sum(len(v) for v in results.values())
        print(f"Archived {total} item(s) to {archive_path}")


if __name__ == "__main__":
    main()
