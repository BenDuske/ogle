"""Smoke test — proves the package imports and the CLI runs.

Real coverage lands as the engine is built out in W1–W3.
"""

import ogle
from ogle.cli import main


def test_package_imports():
    assert ogle.__version__ == "0.1.0"


def test_cli_runs_and_exits_zero(capsys):
    rc = main()
    assert rc == 0
    captured = capsys.readouterr()
    assert "ogle" in captured.out.lower()
