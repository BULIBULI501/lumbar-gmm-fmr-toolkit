"""Regression tests for the thin CLI dispatcher (lumbarmri.cli)."""

from __future__ import annotations

import importlib

import pytest

from lumbarmri.cli import GROUPS, main


def test_usage_on_no_args(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "groups:" in out and "synth" in out


def test_usage_on_help_flag(capsys):
    assert main(["--help"]) == 0
    assert "usage:" in capsys.readouterr().out


def test_unknown_group_returns_2(capsys):
    assert main(["definitely-not-a-group"]) == 2
    err = capsys.readouterr().err
    assert "unknown group" in err and "usage:" in err


def test_every_group_module_importable_and_has_main():
    for group, module_name in GROUPS.items():
        module = importlib.import_module(f"lumbarmri.{module_name}")
        assert callable(getattr(module, "main", None)), group


def test_dispatch_forwards_to_module(capsys):
    # synth's verify is fast, deterministic and self-contained
    assert main(["synth", "verify"]) == 0
    out = capsys.readouterr().out
    assert "all checks passed" in out


def test_forwarded_unknown_command_is_module_error(capsys):
    # the dispatcher must not swallow module-level argparse errors
    with pytest.raises(SystemExit) as excinfo:
        main(["synth", "not-a-command"])
    assert excinfo.value.code == 2
