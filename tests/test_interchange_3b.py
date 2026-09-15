"""Script-level pieces of scripts/interchange_3b.py that decide what a run reads
and how an arm is scored: config inheritance, prerequisites for a gold set, the
3B.C2 arm summary, and verdict assignment under both rules (Wave 1b
pre-registration 2026-09-14, S4, S6, S9)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import interchange_3b as i3  # noqa: E402

GOLD3 = "configs/3B-causal/champYb-gold3.yaml"


def test_gold_config_inherits_everything_but_the_position_set():
    base, gold = i3.load_config("configs/3B-causal/champYb.yaml"), i3.load_config(GOLD3)
    for key in ("champion", "hook", "heads", "readout", "dictionaries", "linear_probes",
                "wave1", "activations", "bsp_suffix"):
        assert gold[key] == base[key], key
    assert gold["rule"] == "3B.C2" and base.get("rule", "3B.C1") == "3B.C1"
    assert gold["positions"] != base["positions"] and gold["labels_suffix"] == "YbGold3"
    assert "extends" not in gold


def test_missing_gold_set_is_produced_by_the_build_script():
    cfg = i3.load_config(GOLD3)
    cfg.update(positions="data/quarto/__nope__.pt", orbit_ids="data/quarto/__nope_orbit__.pt",
               labels_suffix="YbNope")
    missing = i3.check_prereqs(cfg, GOLD3)
    items = {m["item"].split(" (")[0]: m["command"] for m in missing}
    build = f"RUN: python scripts/build_gold_sets.py --config={GOLD3}"
    assert items["positions"] == items["orbit ids"] == build
    assert all(cmd == build for item, cmd in items.items() if item.startswith("labels "))
    # without the config path there is no recipe command to print
    assert all(not (m["command"] or "").startswith("RUN: python scripts/build_gold")
               for m in i3.check_prereqs(cfg))


def _arm(kind, dp, db, ds, nd=None, grp=None):
    n = len(dp)
    return {"dp": torch.tensor(dp), "db": torch.tensor(db), "ds": torch.tensor(ds),
            "hp": torch.zeros(n, dtype=torch.bool), "hb": torch.zeros(n, dtype=torch.bool),
            "grp": torch.arange(n) if grp is None else grp, "nd": nd}


class TestSummarizeC2:
    def test_switch_on_is_network_own_and_chance_corrected(self):
        # base agrees with the source on 2 of 4; the patch reproduces D(s) on 3 of 4
        A = _arm("switch_on", dp=[5, 6, 7, 0], db=[5, 6, 1, 1], ds=[5, 6, 7, 8])
        out = i3.summarize_c2("switch_on", A)
        assert out["r0"] == pytest.approx(0.5) and out["iia"] == pytest.approx(0.75)
        assert out["iia_star"] == pytest.approx(0.5)
        assert out["oracle"]["iia_star"] == 0.0            # oracle: no hits either way
        assert "p" not in out                              # no null given

    def test_specificity_is_the_flip_rate_and_its_excess(self):
        dp, db = [1, 2, 3, 4], [1, 2, 0, 0]               # F = 0.5
        nd = torch.tensor([[1, 2, 3, 0], [1, 2, 0, 0], [1, 0, 0, 0]])   # null F: .25, 0, .25
        out = i3.summarize_c2("specificity", _arm("specificity", dp, db, ds=[9, 9, 9, 9], nd=nd))
        assert out["F"] == pytest.approx(0.5) and out["iia_star"] is None
        assert out["null_median_F"] == pytest.approx(0.25)
        assert out["E"] == pytest.approx(0.25)
        assert out["p"] == pytest.approx(1 / 4, rel=0.5) or out["parametric_tail"]

    def test_switch_on_null_flags(self):
        n = 50
        ds = torch.arange(n) % 16
        db = (ds + 1) % 16                                  # base never agrees: r0 = 0
        dp = ds.clone()                                     # the patch always does: IIA_net* = 1
        nd = torch.stack([(ds + 1 + k % 15) % 16 for k in range(40)])   # null: never reaches D(s)
        out = i3.summarize_c2("switch_on", {"dp": dp, "db": db, "ds": ds, "hp": dp == ds,
                                            "hb": db == ds, "grp": torch.arange(n), "nd": nd})
        assert out["iia_star"] == pytest.approx(1.0) and out["p"] < 0.05
        assert not out["below_null_p5"]


def _result(bsp, on, spec, off):
    return {"bsp_id": bsp, "roles": {"R7": {"switch_on": on, "specificity": spec, "switch_off": off}}}


class TestAssignVerdicts:
    @staticmethod
    def _results(e_leak):
        on = lambda: {"n": 500, "iia_star": 0.8, "p": 0.001}          # noqa: E731
        spec = lambda: {"n": 500, "iia_star": None, "F": 0.5, "E": e_leak, "p": 0.001}  # noqa: E731
        off = lambda: {"n": 80, "iia_star": 0.6, "p": 0.001}          # noqa: E731
        return [_result(f"c{i}", on(), spec(), off()) for i in range(4)]

    def test_small_leak_passes_the_gate_under_c2(self):
        res = self._results(e_leak=0.2)                    # rho = 0.25
        gate = i3.assign_verdicts(res, "3B.C2")
        assert all(r["roles"]["R7"]["verdict"] == "concept-consistent" for r in res)
        assert res[0]["roles"]["R7"]["rho"] == pytest.approx(0.25)
        assert gate["passed"] and gate["rule_version"] == "3B.C2"

    def test_large_leak_is_context_blind_under_c2(self):
        res = self._results(e_leak=0.5)                    # rho = 0.625
        gate = i3.assign_verdicts(res, "3B.C2")
        assert all(r["roles"]["R7"]["verdict"] == "context-blind" for r in res)
        assert not gate["passed"]

    def test_c1_ignores_the_size_of_the_leak(self):
        res = self._results(e_leak=0.2)
        i3.assign_verdicts(res, "3B.C1")
        assert all(r["roles"]["R7"]["verdict"] == "context-blind" for r in res)
        assert "rho" not in res[0]["roles"]["R7"]
