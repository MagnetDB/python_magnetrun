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
