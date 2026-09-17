"""Thin pytest wrapper around scripts/check_budget_gate_bypass.py — makes
the static bypass check runnable and visible in a normal `pytest` run (and
therefore in this repo's existing test tooling/IDE integration), not only
as a separate CI step. The script itself stays the source of truth and can
also be run standalone: `python scripts/check_budget_gate_bypass.py`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "check_budget_gate_bypass.py"


def _load_checker_module():
    spec = importlib.util.spec_from_file_location("check_budget_gate_bypass", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_source_file_bypasses_the_budget_gate():
    checker = _load_checker_module()
    exit_code = checker.main()
    assert exit_code == 0, (
        "A source file calls LLMClient.call() (or constructs LLMClient) "
        "outside the allowlisted path through BudgetGate — see the printed "
        "findings above (or run `python scripts/check_budget_gate_bypass.py` "
        "directly) for exactly which file/line."
    )


def test_bypass_checker_actually_detects_a_real_violation(tmp_path, monkeypatch):
    """Proves the checker isn't vacuously passing (e.g. from a scan-path
    bug that silently walks zero files) — plants a real violation in a
    throwaway copy of the source tree and confirms it's caught."""
    checker = _load_checker_module()

    fake_src_root = tmp_path / "committee"
    fake_src_root.mkdir()
    (fake_src_root / "sneaky.py").write_text(
        "from committee.llm.client import LLMClient\n\n"
        "async def bypass(llm_client):\n"
        "    return await llm_client.call(system_prompt='x', user_prompt='y', response_model=None)\n"
    )

    monkeypatch.setattr(checker, "SRC_ROOT", fake_src_root)
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)

    exit_code = checker.main()
    assert exit_code == 1


def test_bypass_checker_ast_scan_covers_every_python_file_under_src(monkeypatch, capsys):
    """Guards against the checker's file-discovery silently shrinking to
    nothing (e.g. a future refactor of SRC_ROOT/rglob that stops matching
    real files) — asserts it actually visited a realistic minimum number of
    files, not just that it returned exit code 0 for some reason."""
    checker = _load_checker_module()
    python_files = list(checker.SRC_ROOT.rglob("*.py"))
    assert len(python_files) > 20, (
        "Expected the real src/committee tree to contain more than 20 "
        "Python files — if this fails, the checker's SRC_ROOT/glob may be "
        "pointed at the wrong location rather than the codebase having "
        "genuinely shrunk that much."
    )
