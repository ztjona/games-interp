"""Post-run diagnostics for 3B-causal Wave 1b sets whose gate FAILED.

Reads each set's summary and per-pair records and touches ONLY the gate's own
representation (R7, DAS-1) and representation-free quantities (the ceilings).
R1-R6 stay sealed: the Wave 1b pre-registration (S10) forbids reading their
verdicts in a set whose gate fails. No interchange is computed.

Per set, per concept family:
  * R7 verdict counts and the median arm statistics (IIA_net*, E, rho, the
    oracle score, target margins, flip rates, |cos(DAS, probe)|);
  * the decomposition that locates the failure: on pairs where the network
    itself changes its decision (D(s) != D(b), "informative"), how often the
    patch reproduces D(s) and how often it leaves D(b) at all; on the rest, how
    often it keeps D(b);
  * the full-activation ceiling (network-own and oracle).
Across sets: concepts whose R7 verdict is concept-consistent in every set.

Usage:
    diagnose_3b_wave1b.py [--runs=<globs>] [--output=<json>]
    diagnose_3b_wave1b.py (-h | --help)

Options:
    -h --help        Show this help message.
    --runs=<globs>   Comma-separated run summaries
                     [default: saes/quarto/analysis/3B-causal_champYb-gold3_wave1b.json,saes/quarto/analysis/3B-causal_champYb-gold5_wave1b.json]
    --output=<json>  [default: saes/quarto/analysis/3B-causal_champYb_wave1b_diagnostics.json]
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from docopt import docopt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib.sae import interchange as ix  # noqa: E402

KINDS = ("switch_on", "specificity", "switch_off")


def family(r: dict) -> str:
    return f"{r['basis']} pinned" if r["kind"] == "pinned" else f"tiger {r['kind']}"


def median(v):
    v = [x for x in v if x is not None]
    return statistics.median(v) if v else None


def arm(r7: dict, kind: str):
    a = r7.get(kind)
    return a if isinstance(a, dict) else None


def diagnose(run_path: Path) -> dict:
    run = json.loads(run_path.read_text(encoding="utf-8"))
    rec = torch.load(str(run_path).replace(".json", "_pairs.pt"), map_location="cpu",
                     weights_only=False)
    fams = defaultdict(list)
    for r in run["results"]:
        fams[family(r)].append(r)
    out = {}
    for f, rs in sorted(fams.items()):
        r7s = [r["roles"]["R7"] for r in rs]
        row = {"n_concepts": len(rs), "r7_verdicts": dict(Counter(x["verdict"] for x in r7s)),
               "rho_ge_threshold": sum((x.get("rho") or 0) >= ix.RHO_CONTEXT_BLIND for x in r7s),
               "abs_cos_das_probe_median": median(
                   [abs(c) for x in r7s for c in x.get("cos_with_lp_per_fold", [])])}
        for kind in KINDS:
            arms = [a for a in (arm(x, kind) for x in r7s) if a]
            row[kind] = {
                "n_median": median([a["n"] for a in arms]),
                "bh_significant": sum(a.get("bh_significant", False) for a in arms),
                "flip_rate_median": median([a["flip_rate"] for a in arms]),
                "target_margin_median": median([a.get("target_margin_mean") for a in arms]),
                "oracle_iia_star_median": median([a["oracle"]["iia_star"] for a in arms]),
            }
            if kind == "specificity":
                row[kind].update(F_median=median([a["F"] for a in arms]),
                                 E_median=median([a.get("E") for a in arms]),
                                 rho_median=median([x.get("rho") for x in r7s]))
            else:
                row[kind]["iia_net_star_median"] = median([a["iia_star"] for a in arms])
            # the informative-pair decomposition, from R7's per-pair records
            tot = Counter()
            for r in rs:
                x = rec.get(f"{r['bsp_id']}|R7|{kind}")
                if x is None:
                    continue
                dp, db, ds = (x[k].long() for k in ("dec_patched", "dec_base", "dec_source"))
                inf = ds != db
                tot.update(pairs=len(dp), informative=int(inf.sum()),
                           reproduces=int((dp == ds)[inf].sum()), leaves=int((dp != db)[inf].sum()),
                           keeps=int((dp == db)[~inf].sum()))
            if tot["pairs"]:
                i, u = tot["informative"], tot["pairs"] - tot["informative"]
                row[kind]["pairs_pooled"] = {
                    "pairs": tot["pairs"], "informative_share": i / tot["pairs"],
                    "informative_reproduces_D(s)": tot["reproduces"] / i if i else None,
                    "informative_leaves_D(b)": tot["leaves"] / i if i else None,
                    "uninformative_keeps_D(b)": tot["keeps"] / u if u else None}
            ce = [r["ceiling"][kind] for r in rs if kind in r.get("ceiling", {})]
            row[kind]["ceiling"] = {
                "network_own": (median([c["network_own"].get("F") for c in ce]) if kind == "specificity"
                                else median([c["network_own"].get("A0") for c in ce])),
                "network_own_is": "F of the full swap" if kind == "specificity" else "A0 = P(D(b) = D(s))",
                "oracle_iia_star": median([c["oracle"]["iia_star"] for c in ce])}
        out[f] = row
    cc = sorted(r["bsp_id"] for r in run["results"]
                if r["roles"]["R7"]["verdict"] in ("concept-consistent", "concept-consistent (on-only)"))
    return {"run": run_path.relative_to(ROOT).as_posix(), "gate": run["gate"],
            "feasibility": run["feasibility"],
            "r7_verdicts": dict(Counter(r["roles"]["R7"]["verdict"] for r in run["results"])),
            "gate_passing_concepts": cc, "family_rollup": out}


def main() -> int:
    args = docopt(__doc__)
    sets = {Path(p).stem: diagnose(ROOT / p) for p in args["--runs"].split(",")}
    strict = [set(s["gate_passing_concepts"]) for s in sets.values()]
    out = {
        "status": ("POST-HOC DIAGNOSTICS OF FAILED GATES (Wave 1b pre-registration S10). "
                   "R7 and representation-free quantities only; R1-R6 SEALED in every set."),
        "sets": sets,
        "concept_consistent_in_every_set": sorted(set.intersection(*strict)) if strict else [],
        "glossary": {**ix.GLOSSARY_C2,
                     "informative": {"range": "[0, 1]", "meaning": "pairs where the network itself decides differently on source and base, D(s) != D(b)"},
                     "reproduces_D(s)": {"range": "[0, 1], ideal 1", "meaning": "on informative pairs, the patched decision equals D(s)"},
                     "leaves_D(b)": {"range": "[0, 1], ideal 1", "meaning": "on informative pairs, the patched decision differs from D(b)"},
                     "keeps_D(b)": {"range": "[0, 1], ideal 1", "meaning": "on uninformative pairs (D(s) = D(b)), the patched decision stays D(b)"}},
    }
    dest = ROOT / args["--output"]
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {dest.relative_to(ROOT).as_posix()}")
    for name, s in sets.items():
        g = s["gate"]
        print(f"{name}: gate {g['C1_das_concept_consistent']}/{g['C1_powered']} ({g['C1_fraction']:.1%}) "
              f"passed={g['passed']}; R7 {s['r7_verdicts']}")
    print("concept-consistent in every set:", out["concept_consistent_in_every_set"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
