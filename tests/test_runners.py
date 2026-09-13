"""Conventions every runners/*.ps1 must follow, checked statically.

These runners are launched detached (runners/launch.ps1 spawns them via WMI),
and a detached process inherits NONE of the interactive shell's state. A runner
that relies on an activated venv works when tried by hand and dies when
launched -- the first 3B-causal launch (2026-09-13) did exactly that:
`python` resolved to the system interpreter, which has torch but no numpy.
"""

import re
from pathlib import Path

import pytest

RUNNERS = sorted((Path(__file__).resolve().parent.parent / "runners").glob("*.ps1"))
PY_CALL = re.compile(r"^\s*(&\s*)?python(\.exe)?\s", re.IGNORECASE)
# The project convention, in two lines:
#     $activate = '.\.venv\Scripts\Activate.ps1'
#     if (Test-Path $activate) { & $activate } else { throw ... }
ACTIVATE_PATH = r"""['"][^'"]*\.venv\\Scripts\\Activate\.ps1['"]"""
ASSIGN = re.compile(r"(\$\w+)\s*=\s*" + ACTIVATE_PATH, re.IGNORECASE)
INVOKE = re.compile(r"&\s*(\$\w+|" + ACTIVATE_PATH + r")", re.IGNORECASE)


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8-sig").splitlines()


def _code_lines(lines: list[str]):
    """(index, line) outside comment-based help and `#` comments."""
    in_help = False
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("<#"):
            in_help = True
        if in_help:
            if "#>" in s:
                in_help = False
            continue
        if s.startswith("#"):
            continue
        yield i, line


def _calls_python(lines: list[str]) -> int | None:
    """Line index of the first python call, or None."""
    return next((i for i, line in _code_lines(lines) if PY_CALL.search(line)), None)


def _activation(lines: list[str]) -> int | None:
    """Line index where the venv is actually ACTIVATED -- the `&` invocation,
    either of the path directly or of the variable the path was assigned to."""
    var = None
    for i, line in _code_lines(lines):
        m = ASSIGN.search(line)
        if m:
            var = m.group(1)
        inv = INVOKE.search(line)
        if inv and (inv.group(1) == var or "Activate.ps1" in inv.group(1)):
            return i
    return None


def test_runners_were_found():
    assert len(RUNNERS) >= 5


def test_detector_catches_a_runner_without_activation():
    """The check must be able to fail (reporting standard 7, applied to tests)."""
    broken = ["$env:PYTHONUTF8 = '1'", "python scripts/x.py"]
    fixed = ["$activate = '.\\.venv\\Scripts\\Activate.ps1'",
             "if (Test-Path $activate) { & $activate } else { throw 'x' }",
             "python scripts/x.py"]
    assert _calls_python(broken) == 1 and _activation(broken) is None
    assert _activation(fixed) == 1 < _calls_python(fixed)


@pytest.mark.parametrize("runner", RUNNERS, ids=lambda p: p.name)
def test_runner_activates_the_venv_before_calling_python(runner):
    lines = _lines(runner)
    first_call = _calls_python(lines)
    if first_call is None:
        pytest.skip("does not call python")
    act = _activation(lines)
    assert act is not None, (
        f"{runner.name} calls python but never activates .venv -- under a "
        "detached launch `python` is the system interpreter")
    assert act < first_call, (
        f"{runner.name} activates .venv on line {act + 1}, after its first "
        f"python call on line {first_call + 1}")


@pytest.mark.parametrize("runner", RUNNERS, ids=lambda p: p.name)
def test_runner_sets_utf8_and_fails_fast(runner):
    if _calls_python(_lines(runner)) is None:
        pytest.skip("does not call python")
    text = runner.read_text(encoding="utf-8-sig")
    assert re.search(r"\$env:PYTHONUTF8\s*=\s*['\"]1['\"]", text), f"{runner.name}: PYTHONUTF8"
    assert "$PSNativeCommandUseErrorActionPreference = $true" in text, f"{runner.name}: fail-fast"
