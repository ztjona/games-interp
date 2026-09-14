"""Post-run diagnostics for a 3B-causal Wave-1 run whose gate FAILED.

Reads the run's summary and per-pair records; touches ONLY the gate's own
representation (R7, DAS-1) and representation-free quantities. R1-R6 stay
sealed: the pre-registration forbids reading their verdicts when the gate fails.

Computes, per concept and pair kind:
  * the full-activation CEILING (pre-registration S5.1 lists it; the run used
    it only for Tier A): the network's own decision on the source, scored
    against the run's targets;
  * R7's interchange accuracy against the NETWORK'S OWN counterfactual
    decision (the pre-registered secondary target of S5.3, which the run did
    not compute), chance-corrected by the base's agreement with the source;
  * R7's specificity leak split by whether the concept's own threat is on the
    board (pinned concepts);
  * the DAS-vs-probe cosines stored by the run;
  * an EXPLORATORY sensitivity of the gate to a specificity effect-size floor
    the pre-registration did not set. It is NOT a result and must never be
    reported as the gate's outcome.

Usage:
    diagnose_3b_wave1.py --config=<yaml> [--device=<d>] [--run=<json>]
    diagnose_3b_wave1.py (-h | --help)

Options:
    -h --help        Show this help message.
    --config=<yaml>  configs/3B-causal/champ<Tag>.yaml
    --device=<d>     [default: cuda]
    --run=<json>     Run summary. Default: saes/quarto/analysis/3B-causal_<champ>_wave1.json
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from docopt import docopt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from lib.sae import interchange as ix  # noqa: E402
from scripts.games import quarto_counterfactuals as qc  # noqa: E402
from scripts.interchange_3b import Champion, onehot  # noqa: E402

FLOORS = (0.0, 0.05, 0.10, 0.20)


def family(r: dict) -> str:
    return f"{r['basis']} pinned" if r["kind"] == "pinned" else f"tiger {r['kind']}"


def median(v):
    v = sorted(x for x in v if x is not None)
    return v[len(v) // 2] if v else None


def main() -> int:
    args = docopt(__doc__)
    dev = args["--device"] if torch.cuda.is_available() else "cpu"
    cfg = yaml.safe_load((ROOT / args["--config"]).read_text(encoding="utf-8"))
    ch = Champion(cfg, False, dev)
    run_path = ROOT / (args["--run"] or f"saes/quarto/analysis/3B-causal_{ch.name}_wave1.json")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    rec = torch.load(str(run_path).replace(".json", "_pairs.pt"), weights_only=False)
    if run["gate"]["passed"]:
        print("note: this run's gate PASSED; these diagnostics are for a failed gate")

    pos = torch.load(ROOT / cfg["positions"], map_location="cpu", weights_only=False)
    boards, pieces = pos["boards"].float(), pos["pieces"].float()
    T = qc.board_tables(boards, pieces)
    SW = qc.all_source_wins(T)

    per_concept = {}
    for r in run["results"]:
        spec = qc.parse_concept(r["bsp_id"], r["basis"])
        ps_all = qc.build_pairs(T, spec, SW, seed=run["seed"])
        off = ps_all["switch_off"]
        if off.n:
            zb = ch.hook_values(boards[off.base], pieces[off.base]).to(dev)
            dec = ix.legal_argmax(ch.logits(zb, None), off.legal.to(dev)).cpu()
            ps_all["switch_off"] = off.select(off.expect_base[torch.arange(off.n), dec])
        row = {}
        for kind, ps in ps_all.items():
            ps = qc.cap_pairs(ps, run["cap"], spec.bsp_id, run["seed"])
            if ps.n == 0:
                continue
            key = f"{spec.bsp_id}|R7|{kind}"
            x = rec[key]
            got = sorted(zip(x["base"].tolist(), x["src_piece"].tolist()))
            if got != sorted(zip(ps.base.tolist(), ps.src_piece.tolist())):
                raise SystemExit(f"rebuilt pairs differ from the run's for {key}")
            legal, target = ps.legal.to(dev), ps.target.to(dev)
            zb = ch.hook_values(boards[ps.base], pieces[ps.base]).to(dev)
            zs = ch.hook_values(boards[ps.base], onehot(ps.src_piece)).to(dev)
            db = ix.legal_argmax(ch.logits(zb, None), legal)
            ds = ix.legal_argmax(ch.logits(zs, None), legal)
            hb, hs = ix.hits(db, target), ix.hits(ds, target)
            iia, r0 = float(hs.float().mean()), float(hb.float().mean())
            # the run's R7 records are in fold order: align them to these pairs
            order = {pair: i for i, pair in enumerate(zip(ps.base.tolist(), ps.src_piece.tolist()))}
            idx = torch.tensor([order[p] for p in zip(x["base"].tolist(), x["src_piece"].tolist())])
            dp = x["dec_patched"].long()
            a_p = float((dp == ds.cpu()[idx]).float().mean())
            a_b = float((db.cpu()[idx] == ds.cpu()[idx]).float().mean())
            entry = {"n": ps.n,
                     "ceiling": {"iia": iia, "r0": r0, "iia_star": ix.chance_corrected(iia, r0)},
                     "r7_network_own": {"agree_patched": a_p, "agree_base": a_b,
                                        "iia_star": ix.chance_corrected(a_p, a_b)}}
            if kind == "specificity" and spec.kind == "pinned":
                tp = T.threat[x["base"].long(), spec.group, spec.pole]
                gain = x["hit_patched"].float() - x["hit_base"].float()
                entry["r7_leak_by_threat"] = {
                    lab: {"n": int(m.sum()), "leak": float(gain[m].mean()) if int(m.sum()) else None}
                    for m, lab in ((tp, "present"), (~tp, "absent"))}
            row[kind] = entry
        per_concept[r["bsp_id"]] = row
        print(f"  {r['bsp_id']}", flush=True)

    # family rollups
    fams = defaultdict(list)
    for r in run["results"]:
        fams[family(r)].append(r)
    rollup = {}
    for f, rs in sorted(fams.items()):
        rollup[f] = {"n_concepts": len(rs)}
        for kind in qc.PAIR_KINDS:
            def col(get):
                return median([get(r) for r in rs if kind in per_concept[r["bsp_id"]]])
            r7 = lambda r: (r["roles"]["R7"].get(kind) or {}).get("iia_star")  # noqa: E731
            rollup[f][kind] = {
                "r7_iia_star": col(r7),
                "ceiling_iia_star": col(lambda r: per_concept[r["bsp_id"]][kind]["ceiling"]["iia_star"]),
                "r7_network_own_iia_star": col(
                    lambda r: per_concept[r["bsp_id"]][kind]["r7_network_own"]["iia_star"]),
            }
        cos = [abs(c) for r in rs for c in r["roles"]["R7"].get("cos_with_lp_per_fold", [])]
        rollup[f]["das_probe_abs_cos_median"] = median(cos)
        leaks = [per_concept[r["bsp_id"]].get("specificity", {}).get("r7_leak_by_threat") for r in rs]
        leaks = [lk for lk in leaks if lk]
        if leaks:
            rollup[f]["r7_leak_threat_present_median"] = median([lk["present"]["leak"] for lk in leaks])
            rollup[f]["r7_leak_threat_absent_median"] = median([lk["absent"]["leak"] for lk in leaks])

    verdicts = defaultdict(int)
    for r in run["results"]:
        verdicts[r["roles"]["R7"]["verdict"]] += 1
    cb = [r["roles"]["R7"] for r in run["results"] if r["roles"]["R7"]["verdict"] == "context-blind"]
    sensitivity = {}
    for floor in FLOORS:
        n = 0
        for r in run["results"]:
            x = r["roles"]["R7"]

            def arm(k, fl=None):
                a = x.get(k)
                if not isinstance(a, dict):
                    return None
                sig = a.get("bh_significant", False)
                if fl is not None:
                    sig = sig and (a["iia_star"] or 0.0) >= fl
                return ix.ArmResult(n=a["n"], iia_star=a["iia_star"], significant=sig,
                                    below_null_p5=a.get("below_null_p5", False),
                                    flip_significant=a.get("flip_p", 1.0) < 0.05)
            v = ix.classify(arm("switch_on"), arm("specificity", floor), arm("switch_off"))
            n += v in ("concept-consistent", "concept-consistent (on-only)")
        sensitivity[f"{floor:.2f}"] = n

    out = {
        "status": ("POST-HOC DIAGNOSTICS OF A FAILED GATE. R7 and representation-free "
                   "quantities only; R1-R6 sealed. The sensitivity block is exploratory "
                   "and is NOT the gate's outcome."),
        "run": run_path.relative_to(ROOT).as_posix(),
        "gate_as_registered": run["gate"],
        "r7_verdicts": dict(verdicts),
        "context_blind_specificity_iia_star": {
            "min": min(x["specificity"]["iia_star"] for x in cb),
            "median": median([x["specificity"]["iia_star"] for x in cb]),
            "max": max(x["specificity"]["iia_star"] for x in cb),
            "median_ratio_to_switch_on": median(
                [x["specificity"]["iia_star"] / x["switch_on"]["iia_star"] for x in cb])},
        "exploratory_gate_by_specificity_floor": sensitivity,
        "family_rollup": rollup,
        "per_concept": per_concept,
        "glossary": {**ix.GLOSSARY,
                     "ceiling": {"range": "(-inf, 1]", "meaning": "IIA* of the network's own decision on the source (full-activation patch)"},
                     "network_own_iia_star": {"range": "(-inf, 1], ideal 1", "meaning": "agreement of the patched decision with the network's decision on the source, chance-corrected by base-vs-source agreement"},
                     "leak": {"range": "[-1, 1], ideal 0", "meaning": "specificity pairs: rise in P(play the concept's cell) under the patch"}},
    }
    dest = str(run_path).replace(".json", "_diagnostics.json")
    Path(dest).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {Path(dest).relative_to(ROOT)}")
    for f, v in rollup.items():
        print(f, json.dumps(v))
    print("exploratory gate by floor:", sensitivity)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
