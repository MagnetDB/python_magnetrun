"""Regression tests: bundled JSON files are found whatever the working directory.

When Python is started from a directory that contains a plain
``python_magnetrun/`` folder (e.g. the root of a repository embedding this
package as a git submodule), that folder is imported as a namespace package
before the editable-install finder runs. Bundled data must still resolve to
the real package directory.
"""

import json
import subprocess
import sys
from pathlib import Path

import python_magnetrun.housing_config as housing_config

PACKAGE_DIR = Path(housing_config.__file__).resolve().parent

_PROBE = """
import json, sys
from pathlib import Path
from python_magnetrun.housing_config import HOUSING_CONFIGS, get_housing_config
from python_magnetrun.field_defs import resolve_defs_file
nouser = Path(sys.argv[1])
print(json.dumps({
    "housings": sorted(HOUSING_CONFIGS),
    "m10_gr1": get_housing_config("M10").reference_gr1_current,
    "pupitre": str(resolve_defs_file("pupitre-defs.json", user_dir=nouser)),
    "magfile": str(resolve_defs_file("magfile-defs.json", user_dir=nouser)),
}))
"""


def _probe_from(cwd: Path) -> dict:
    """Run the probe in a fresh interpreter started from *cwd*."""
    nouser = cwd / "no-user-config"
    nouser.mkdir()
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, str(nouser)],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_bundled_files_found_when_cwd_shadows_package(tmp_path: Path):
    """A ``python_magnetrun/`` folder in the cwd must not hide the bundled JSON files."""
    (tmp_path / "python_magnetrun").mkdir()
    probe = _probe_from(tmp_path)

    assert {"M5", "M7", "M8", "M9", "M10"} <= set(probe["housings"])
    assert probe["m10_gr1"] == "IB"
    assert Path(probe["pupitre"]).parent == PACKAGE_DIR
    assert Path(probe["magfile"]).parent == PACKAGE_DIR
