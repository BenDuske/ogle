"""Smoke test — proves the package imports and the CLI runs.

Real coverage lands as the engine is built out in W1–W3.
"""

import ogle
from ogle.cli import main


def test_package_imports():
    assert ogle.__version__ == "0.1.0"


def test_cli_runs_and_exits_zero(capsys):
    # Explicit empty argv — otherwise argparse reads real sys.argv, which includes
    # pytest's own flags (e.g. -q from pyproject addopts) and confuses the CLI.
    rc = main([])
    assert rc == 0
    captured = capsys.readouterr()
    assert "ogle" in captured.out.lower()
