"""expand_input_files: pigbrother TDMS names resolve to their on-disk subdirectory."""

import os

import pytest

from python_magnetrun.utils.files import expand_input_files


@pytest.fixture
def pbsurv(tmp_path, monkeypatch):
    """A pbsurv-like tree with Overview and Fichiers_stats files for M8."""
    root = tmp_path / "pbsurv"
    files = [
        "M8/Overview/M8_Overview_261113-0813.tdms",
        "M8/Fichiers_stats/M8_Stats_261113-081343.tdms",
        "M8/Fichiers_stats/M8_Stats__190110-150955.tdms",
    ]
    for rel in files:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    return root


@pytest.mark.parametrize(
    "name, subdir",
    [
        ("M8_Overview_261113-0813.tdms", "Overview"),
        ("M8_Stats_261113-081343.tdms", "Fichiers_stats"),
        ("M8_Stats__190110-150955.tdms", "Fichiers_stats"),
    ],
)
def test_tdms_name_resolves_to_mode_subdirectory(pbsurv, name, subdir):
    result = expand_input_files([name], {".tdms": str(pbsurv)}, housing="M8")
    assert result == [os.path.join(str(pbsurv), "M8", subdir, name)]
