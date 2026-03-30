"""
Integration smoke-tests for CLI entry points in python_magnetrun

These tests check that each CLI entry point runs without crashing and prints usage/help text.
"""
import subprocess
import sys

import pytest

# List of CLI entry points as installed scripts or module paths
CLI_COMMANDS = [
    [sys.executable, '-m', 'python_magnetrun.python_magnetrun'],
    [sys.executable, '-m', 'python_magnetrun.requests.cli'],
    [sys.executable, '-m', 'python_magnetrun.analysis.cli'],
    [sys.executable, '-m', 'python_magnetrun.hybrid.cli'],
    [sys.executable, '-m', 'python_magnetrun.configAlims.convertxml'],
    [sys.executable, '-m', 'python_magnetrun.hybrid.__main__'],
    [sys.executable, '-m', 'python_magnetrun.processing.cli'],
    # [sys.executable, '-m', 'python_magnetrun.tdms.log_parser'],  # Requires file argument, not just --help
    # [sys.executable, '-m', 'rustfs.magnetfs.cli'],  # Rust CLI not supported in this test
]

@pytest.mark.parametrize("cmd", CLI_COMMANDS)
def test_cli_entrypoint_runs(cmd):
    """Smoke test: CLI entry point runs and prints help/usage."""
    result = subprocess.run(cmd + ['--help'], capture_output=True, text=True)
    assert result.returncode == 0, f"{cmd} failed: {result.stderr}"
    assert 'usage' in result.stdout.lower() or 'help' in result.stdout.lower(), f"{cmd} did not print usage/help"
