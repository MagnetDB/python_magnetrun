import logging

import pytest

ON_DEMAND_TESTS = {
    "test-paramident.py",
    "test-breakpoint-analysis.py",
    "test-fft.py",
    "test-fieldfactor.py",
    "test-intercept.py",
    "test-signature.py",
    "test-simu.py",
    "test-tin.py",
}


def pytest_addoption(parser):
    parser.addoption(
        "--on-demand",
        action="store_true",
        default=False,
        help="run on-demand tests that require data files",
    )


def pytest_ignore_collect(collection_path, config):
    if collection_path.name in ON_DEMAND_TESTS and not config.getoption(
        "--on-demand", default=False
    ):
        return True


@pytest.fixture(autouse=True)
def _reset_package_logger():
    """Save and restore python_magnetrun logger state around each test.

    Prevents tests that call setup_logging() (which sets propagate=False and
    adds stdout handlers) from breaking caplog-based tests that run afterwards.
    """
    pkg_logger = logging.getLogger("python_magnetrun")
    saved_propagate = pkg_logger.propagate
    saved_level = pkg_logger.level
    saved_handlers = list(pkg_logger.handlers)
    yield
    pkg_logger.handlers.clear()
    for handler in saved_handlers:
        pkg_logger.addHandler(handler)
    pkg_logger.propagate = saved_propagate
    pkg_logger.setLevel(saved_level)
