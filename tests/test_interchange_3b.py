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


class TestRuleC3Assignment:
    """Wave 1c S9-S11: installs is the verdict, removal a label, and the removal
    hypotheses count pinned concepts with a powered switch-off arm."""

    @staticmethod
    def _concept(bsp, kind, off_n, removes_by_role, e_leak=0.0):
        def roles():
            out = {}
            for role, rem in removes_by_role.items():
                out[role] = {
                    "switch_on": {"n": 500, "iia_star": 0.8, "p": 0.001},
                    "specificity": {"n": 500, "iia_star": None, "F": 0.2, "E": e_leak,
                                    "p": 0.001 if e_leak else 0.9},
                    "switch_off": {"n": off_n, "iia_star": 0.6 if rem else 0.0,
                                   "p": 0.001 if rem else 0.9}}
            return out
        return {"bsp_id": bsp, "kind": kind, "n_pairs": {"switch_off": off_n}, "roles": roles()}

    @staticmethod
    def _roles(off_ok, k2=False, k4=True):
        return {"R7": False, "R7off": off_ok, "R8k2": k2, "R8k4": k4, "R8k8": True, "R8k16": True}

    def test_install_gate_removal_label_and_c2_continuity(self):
        res = [self._concept(f"c{i}", "pinned", 80, {"R7": False}) for i in range(4)]
        gate = i3.assign_verdicts(res, "3B.C3")
        r7 = res[0]["roles"]["R7"]
        assert r7["verdict"] == "installs" and r7["removal"] == "does not remove"
        assert r7["verdict_3B.C2"] == "install-only"
        assert gate["passed"] and gate["C1_das_concept_consistent"] == 4

    def test_removal_hypotheses_count_pinned_concepts_with_powered_switch_off(self):
        res = ([self._concept(f"p{i}", "pinned", 80, self._roles(i < 1)) for i in range(4)]
               + [self._concept("under", "pinned", 10, self._roles(True)),       # switch-off underpowered
                  self._concept("tig", "winnable", 80, self._roles(True))])      # not pinned
        rm = i3.assign_verdicts(res, "3B.C3")["removal"]
        assert rm["concepts"] == 4
        assert rm["per_representation"]["R8k4"]["fraction"] == 1.0
        assert rm["H-C6"]["outcome"] == "prediction holds" and rm["H-C6"]["k_star"] == 4
        assert rm["per_representation"]["R7off"]["fraction"] == 0.25
        assert rm["H-C7"]["outcome"] == "inconclusive"

    def test_context_blind_subspaces_do_not_count_as_removing(self):
        res = [self._concept(f"p{i}", "pinned", 80, self._roles(False, k2=True), e_leak=0.7)
               for i in range(3)]
        rm = i3.assign_verdicts(res, "3B.C3")["removal"]
        assert all(rm["per_representation"][f"R8k{k}"]["fraction"] == 0.0 for k in (2, 4, 8, 16))
        assert rm["H-C6"]["outcome"] == "falsified"


class _FakeChampion:
    """Closed-form champion on toy inputs: z = linear(board, piece)."""

    def __init__(self, d=24, seed=0):
        from lib.sae import interchange as ix
        g = torch.Generator().manual_seed(seed)
        self.Wb, self.Wp = torch.randn(256, d, generator=g) * 0.3, torch.randn(16, d, generator=g)
        self.readout = ix.LinearReluReadout(torch.randn(16, d, generator=g), torch.zeros(16))

    def hook_values(self, boards, pieces):
        return boards.reshape(len(boards), -1) @ self.Wb + pieces @ self.Wp

    def logits(self, zp, inputs):
        ro = self.readout
        return torch.relu(zp) @ ro.W.T + ro.b


class _FakeDictionary:
    def __init__(self, d=24, d_dict=32, seed=1):
        g = torch.Generator().manual_seed(seed)
        self.We = torch.randn(d, d_dict, generator=g)
        self.W_dec = torch.nn.functional.normalize(torch.randn(d_dict, d, generator=g), dim=1)
        self.freq = torch.rand(d_dict, generator=g) * 0.5 + 0.01

    def encode(self, z):
        return torch.relu(z @ self.We)


def test_c3_concept_runs_every_registered_representation(monkeypatch):
    """Wave 1c S7: R1-R7 as Wave 1b, plus R8-k for k in {2,4,8,16} and R7-off
    where switch-off is powered; S9: every role gets a 3B.C3 verdict, a removal
    label and the 3B.C2 verdict. The untrained-twin smoke cannot reach R7-off
    (its switch-off filter keeps no pairs), so it is exercised here."""
    from lib.sae import interchange as ix
    from scripts.games import quarto_counterfactuals as qc
    monkeypatch.setattr(i3, "DAS_STEPS", 25)
    g = torch.Generator().manual_seed(3)
    N = 400
    boards = (torch.rand(N, 16, 4, 4, generator=g) < 0.05).float()
    pieces = torch.nn.functional.one_hot(torch.randint(16, (N,), generator=g), 16).float()
    ch, D = _FakeChampion(), _FakeDictionary()
    Z_all = ch.hook_values(boards, pieces)
    orbit = torch.randint(60, (N,), generator=g)

    def pairset(kind, n):
        base = torch.randint(N, (n,), generator=g)
        legal = torch.rand(n, 16, generator=g) < 0.6
        legal[:, 0] = True
        target = torch.zeros(n, 16, dtype=torch.bool)
        target[torch.arange(n), torch.randint(16, (n,), generator=g)] = True
        return qc.PairSet(kind, base, torch.randint(16, (n,), generator=g), target, legal,
                          target.clone(), torch.zeros(n, 8, dtype=torch.bool), torch.zeros(n, dtype=torch.bool))

    pairs = {"switch_on": pairset("switch_on", 150), "specificity": pairset("specificity", 150),
             "switch_off": pairset("switch_off", 80)}
    spec = qc.ConceptSpec("row_0_completable_tall", "hawk", "pinned", group=0, pole=0)
    sl = {"top": list(range(16)), "abs_mcc": torch.rand(32, generator=g)}
    rec = i3.Records()
    res = i3.run_concept(spec, pairs, ch, Z_all, boards, pieces, orbit, D, sl, 3,
                         torch.randn(24, generator=g), None, None, 20, 0, "cpu", rec, rule=ix.RULE_C3)
    expected = {"R1", "R2", "R3", "R4", "R5", "R7", "R7off", "R8k2", "R8k4", "R8k8", "R8k16"}
    assert expected <= {r for r, v in res["roles"].items() if isinstance(v, dict)}
    gate = i3.assign_verdicts([res], ix.RULE_C3)
    for role in expected:
        rr = res["roles"][role]
        assert rr["verdict"] in ix.VERDICTS and rr["removal"] in ("removes", "does not remove", "underpowered")
        assert "verdict_3B.C2" in rr and rr["switch_off"]["n"] == 80
    assert f"{spec.bsp_id}|R8k16|switch_off" in rec.rows and f"{spec.bsp_id}|R7off|switch_on" in rec.rows
    assert gate["removal"]["concepts"] == 1


def test_prereqs_list_every_excluded_run():
    cfg = i3.load_config("configs/3B-causal/champYb-gold3r2.yaml")
    cfg["pilot_runs"] = cfg["pilot_runs"] + ["saes/quarto/analysis/__nope__.json"]
    missing = [m for m in i3.check_prereqs(cfg, "x") if m["item"].startswith("earlier run")]
    assert [m["path"] for m in missing] == ["saes/quarto/analysis/__nope__.json"]
