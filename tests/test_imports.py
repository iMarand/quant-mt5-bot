"""Every module must compile and import.

This exists because a syntax error shipped in `cli.py`: no test imported it, so
the whole suite passed while `python -m quantbot` was dead on arrival. Cheap
insurance — it catches the class of mistake that unit tests structurally miss.
"""

from __future__ import annotations

import importlib
import pkgutil
import py_compile
from pathlib import Path

import pytest

import quantbot

ROOT = Path(quantbot.__file__).resolve().parent
PROJECT = ROOT.parent


def _python_files() -> list[Path]:
    files = sorted(ROOT.rglob("*.py"))
    files += sorted((PROJECT / "tools").rglob("*.py"))
    return files


def _modules() -> list[str]:
    names = ["quantbot"]
    for info in pkgutil.walk_packages([str(ROOT)], prefix="quantbot."):
        # __main__ executes the CLI on import; compiling it is enough.
        if info.name.endswith("__main__"):
            continue
        names.append(info.name)
    return names


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: p.name)
def test_file_compiles(path: Path):
    py_compile.compile(str(path), doraise=True)


@pytest.mark.parametrize("module", _modules())
def test_module_imports(module: str):
    importlib.import_module(module)


def test_cli_parser_builds_and_every_command_is_wired():
    """Catches a subcommand registered without a handler."""
    from quantbot.cli import build_parser

    parser = build_parser()
    subparsers = [
        action for action in parser._actions if hasattr(action, "choices") and action.choices
    ]
    assert subparsers, "no subcommands registered"
    commands = subparsers[0].choices
    expected = {
        "doctor", "ingest", "import-csv", "import-calendar", "predict",
        "train", "search", "backtest", "run", "report", "gate", "resolve",
    }
    assert expected <= set(commands), f"missing: {expected - set(commands)}"
    for name, sub in commands.items():
        assert sub.get_default("func") is not None, f"{name} has no handler"


@pytest.mark.parametrize("args", [
    ["doctor"], ["report"], ["gate"], ["resolve"], ["predict"], ["train"],
    ["run", "--once"], ["backtest"], ["ingest"],
    ["import-csv", "x.csv", "--symbol", "EURUSD"], ["import-calendar", "x.csv"],
])
def test_command_line_parses(args):
    """Each command's flags actually parse — no typos in the parser wiring."""
    from quantbot.cli import build_parser

    parsed = build_parser().parse_args(args)
    assert callable(parsed.func)
