"""scripts/build_gold_sets.py reuses an existing raw or positions file only if
its provenance matches the recipe: two draws of one protocol share a raw file
NAME (Wave 1c's gold3r2 and Wave 1b's gold3 both write positions-gold3-*_raw.pt)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import build_gold_sets as bg  # noqa: E402

CFG = {"positions_recipe": {"opponents": "gold3", "seed": 3103, "games": 30000}}


def _raw(path, seed):
    torch.save({"provenance": {"opponents": "gold3", "seed": seed, "num_games": 30000}}, path)


def test_matching_raw_is_reused(tmp_path):
    _raw(tmp_path / "raw.pt", 3103)
    bg.check_reuse("generate", str(tmp_path / "raw.pt"), CFG)


def test_another_draw_is_refused(tmp_path):
    _raw(tmp_path / "raw.pt", 3003)
    with pytest.raises(SystemExit, match="drawn with"):
        bg.check_reuse("generate", str(tmp_path / "raw.pt"), CFG)


def test_filtered_positions_are_checked_through_their_source(tmp_path):
    torch.save({"provenance": {"source_provenances": [
        {"file": "raw.pt", "opponents": "gold3", "seed": 3003, "num_games": 30000}]}},
        tmp_path / "pos.pt")
    with pytest.raises(SystemExit):
        bg.check_reuse("fresh", str(tmp_path / "pos.pt"), CFG)
    bg.check_reuse("labels", str(tmp_path / "pos.pt"), CFG)       # label files carry no recipe


def test_wave1c_sets_have_their_own_raw_dirs():
    from scripts.interchange_3b import load_config
    raws = {c: bg.steps(load_config(f"configs/3B-causal/champYb-{c}.yaml"))[0][2]
            for c in ("gold3", "gold3r2", "gold5", "gold5r2")}
    assert len(set(raws.values())) == 4


def test_several_excluded_runs_reach_the_filter():
    from scripts.interchange_3b import load_config
    fresh = next(argv for st, argv, _ in bg.steps(load_config("configs/3B-causal/champYb-gold3r2.yaml"))
                 if st == "fresh")
    assert sum(a.startswith("--pilot-run=") for a in fresh) == 3
