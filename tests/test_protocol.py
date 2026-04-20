"""Protocol compliance tests for DataLoader.

Verifies that MagnetRun and HybridRun structurally satisfy the DataLoader
protocol via Python's runtime_checkable isinstance() check.
"""


from python_magnetrun.hybrid.data_protocol import DataLoader
from python_magnetrun.hybrid.hybrid_run import HybridRun
from python_magnetrun.MagnetRun import MagnetRun


def test_magnetrun_satisfies_protocol():
    mrun = MagnetRun()
    assert isinstance(mrun, DataLoader), (
        "MagnetRun does not satisfy the DataLoader protocol — "
        f"missing: {[m for m in ('getData','getKeys','getType','getSite','getHousing','getDomain','get_time_range') if not hasattr(mrun, m)]}"
    )


def test_hybridrun_satisfies_protocol():
    hrun = HybridRun()
    assert isinstance(hrun, DataLoader), (
        "HybridRun does not satisfy the DataLoader protocol — "
        f"missing: {[m for m in ('getData','getKeys','getType','getSite','getHousing','getDomain','get_time_range') if not hasattr(hrun, m)]}"
    )


def test_magnetrun_getdomain():
    assert MagnetRun().getDomain() == "operational"


def test_hybridrun_getdomain():
    assert HybridRun().getDomain() == "operational"


def test_dataloader_required_methods():
    required = {"getData", "getKeys", "getType", "getSite", "getHousing", "getDomain", "get_time_range"}
    protocol_attrs = set(m for m in dir(DataLoader) if not m.startswith("_"))
    missing = required - protocol_attrs
    assert not missing, f"DataLoader protocol is missing: {missing}"
