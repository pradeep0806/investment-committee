"""Static check: fails (non-zero exit) if any code path in src/committee
calls an LLMClient's `.call(...)` from outside the one place that's allowed
to (orchestration/budget_gate.py) — making "the gate can't be bypassed" a
build-time property of the codebase, not just an observation about the code
paths that happen to exist today (interview follow-up, Core item D).

What this scans for, via a plain `ast` walk (no type inference, deliberately
simple per the task's "AST scan or lint rule" framing):

  1. Any `.call(...)` method call where the receiver expression's source
     text contains "llm_client" (case-insensitive) — covering
     `self._llm_client.call(...)`, `llm_client.call(...)`,
     `self.llm_client.call(...)`, etc. This is a name-based heuristic, not
     real type inference; it deliberately errs toward flagging anything
     that *looks* like it could be an LLMClient, on the theory that a false
     positive (a legitimately-allowed call needing an allowlist entry) is
     far cheaper than a false negative (a real bypass slipping through
     silently).
  2. Any `LLMClient(...)` — a direct constructor call — outside
     `llm/client.py` itself, since a second, independently-constructed
     LLMClient instance held anywhere other than inside a BudgetGate is
     exactly the seam a bypass would use. Building one *and immediately
     handing it to BudgetGate* (orchestrator_factory.py's actual, legitimate
     pattern) is still flagged by this narrow AST-only check since it can't
     see what happens to the value afterward — allowlisted explicitly below
     with the reasoning for why that specific call site is safe.

ALLOWED_FILES lists every file where a raw `.call(...)`/`LLMClient(...)` is
legitimate, with an inline reason each — anything not listed there is a
build-time failure.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src" / "committee"

# path (relative to src/committee) -> reason this file may hold a raw
# LLMClient reference and/or call `.call()` on one directly.
ALLOWED_FILES: dict[str, str] = {
    "llm/client.py": (
        "LLMClient's own module — defines .call() and, internally, retries "
        "the primary/fallback raw callers (still not the same as an "
        "external caller reaching LLMClient.call() directly)."
    ),
    "orchestration/budget_gate.py": (
        "BudgetGate is the one sanctioned caller of LLMClient.call() — its "
        "entire purpose is to be the sole path between any other code and "
        "the LLM (see its own module docstring)."
    ),
    "orchestrator_factory.py": (
        "Constructs exactly one LLMClient via build_llm_client_from_settings "
        "and immediately passes it to BudgetGate(llm_client=...) — never "
        "calls .call() on it directly. Allowlisted for LLMClient(...) "
        "construction only; a .call() invocation here would still fail "
        "this check even with the file allowlisted, since the two checks "
        "are independent (see _check_file)."
    ),
}


class _BypassFinding:
    def __init__(self, file: Path, line: int, kind: str, detail: str):
        self.file = file
        self.line = line
        self.kind = kind
        self.detail = detail

    def __str__(self) -> str:
        rel = self.file.relative_to(REPO_ROOT)
        return f"{rel}:{self.line}: {self.kind} — {self.detail}"


def _relative_to_src(path: Path) -> str:
    return str(path.relative_to(SRC_ROOT))


def _receiver_source(node: ast.expr, source: str) -> str:
    try:
        return ast.get_source_segment(source, node) or ""
    except Exception:
        return ""


def _scan_file(path: Path) -> list[_BypassFinding]:
    source = path.read_text()
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []

    rel = _relative_to_src(path)
    allow_call = rel in ALLOWED_FILES
    # orchestrator_factory.py is allowlisted for *construction* only (see
    # ALLOWED_FILES docstring above) — a .call() there would still be a
    # real bypass, so it's deliberately excluded from allow_call despite
    # being present in ALLOWED_FILES for the constructor check.
    if rel == "orchestrator_factory.py":
        allow_call = False

    findings: list[_BypassFinding] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            # Rule 1: `<something with "llm_client" in it>.call(...)`
            if isinstance(func, ast.Attribute) and func.attr == "call":
                receiver_text = _receiver_source(func.value, source).lower()
                if "llm_client" in receiver_text and not allow_call:
                    findings.append(
                        _BypassFinding(
                            path,
                            node.lineno,
                            "raw .call() on an LLMClient-like receiver",
                            f"receiver expression: {_receiver_source(func.value, source)!r}",
                        )
                    )
            # Rule 2: direct `LLMClient(...)` construction.
            constructed_name = None
            if isinstance(func, ast.Name):
                constructed_name = func.id
            elif isinstance(func, ast.Attribute):
                constructed_name = func.attr
            if constructed_name == "LLMClient" and rel not in ALLOWED_FILES:
                findings.append(
                    _BypassFinding(
                        path,
                        node.lineno,
                        "direct LLMClient(...) construction",
                        "LLMClient must only be constructed by "
                        "llm/client.py's own factory or orchestrator_factory.py",
                    )
                )

    return findings


def main() -> int:
    all_findings: list[_BypassFinding] = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        all_findings.extend(_scan_file(path))

    if not all_findings:
        print("check_budget_gate_bypass: OK — no bypass of BudgetGate found.")
        return 0

    print("check_budget_gate_bypass: FAILED — found code path(s) that bypass BudgetGate:\n")
    for finding in all_findings:
        print(f"  {finding}")
    print(
        "\nEvery call to LLMClient.call() must go through "
        "orchestration/budget_gate.py's BudgetGate.call() — see that "
        "module's docstring for why. If this is a legitimate new call site "
        "(e.g. a new construction point mirroring orchestrator_factory.py), "
        "add it to ALLOWED_FILES in scripts/check_budget_gate_bypass.py "
        "with a reason, rather than suppressing this check."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
