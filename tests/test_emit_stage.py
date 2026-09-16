"""scripts/emit_stage.py writes stage_<slug>.md ONLY when an output needs
`git add -f`; everything else shows in git status, so a plan listing it is noise
(decided 2026-09-15)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import emit_stage as es  # noqa: E402


@pytest.fixture
def files(tmp_path, monkeypatch):
    kinds = {"ok.json": "new-untracked", "mod.json": "tracked-modified",
             "keep.pt": "ignored-small", "bulk.pt": "ignored-large"}
    for name in kinds:
        (tmp_path / name).write_bytes(b"x")
    monkeypatch.setattr(es, "classify",
                        lambda p, mb: kinds.get(p.name, "missing") if p.exists() else "missing")
    return tmp_path


def run(monkeypatch, out, paths):
    monkeypatch.setattr(sys, "argv", ["emit_stage.py", "--slug=t", f"--out={out}", *map(str, paths)])
    es.main()


def test_no_file_when_nothing_needs_a_force_add(files, monkeypatch, capsys):
    out = files / "stage_t.md"
    out.write_text("stale plan from an earlier run", encoding="utf-8")
    run(monkeypatch, out, [files / "ok.json", files / "mod.json", files / "bulk.pt", files / "gone.json"])
    assert not out.exists()                                   # stale plan removed
    printed = capsys.readouterr().out
    assert "nothing needs a force-add" in printed and "missing: " in printed


def test_force_add_files_are_the_whole_plan(files, monkeypatch):
    out = files / "stage_t.md"
    run(monkeypatch, out, [files / "ok.json", files / "keep.pt", files / "bulk.pt"])
    md = out.read_text(encoding="utf-8")
    assert "git add -f" in md and "keep.pt" in md
    assert "ok.json" not in md and "bulk.pt" not in md
