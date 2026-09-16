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
Across sets: concepts whose R7 verdict is concept-consistent in every set, and
a profile of each set's gate-passing concepts -- their verdicts in the other
set, switch-off removal by pole and group type, removal by pieces on board.

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
import re
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


def _pole_and_group(bsp_id: str):
    m = re.match(r"((row|col|diag|square)_\w+?)_completable_(\w+)", bsp_id)
    return (m.group(3), m.group(2)) if m else (None, None)


def passing_profile(paths: list[Path]) -> dict:
    """What the gate-passing concepts of each set have in common (added
    2026-09-15 on request): their verdicts in the other sets, switch-off
    removal by pole and by group type, and removal by pieces on the board."""
    where = {p.stem: p for p in paths}
    runs = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in paths}
    res = {r_: {x["bsp_id"]: x for x in run["results"]} for r_, run in runs.items()}
    out = {}
    for name, run in runs.items():
        R = res[name]
        others = [o for o in res if o != name]
        passing = {v: sorted(b for b, x in R.items() if x["roles"]["R7"]["verdict"] == v)
                   for v in ("concept-consistent (on-only)", "concept-consistent")}
        prof = {}
        for v, ids in passing.items():
            prof[v] = {"n": len(ids),
                       "switch_off_n_median": median([R[b]["n_pairs"]["switch_off"] for b in ids]),
                       "verdict_in_other_sets": {o: dict(Counter(res[o][b]["roles"]["R7"]["verdict"] for b in ids))
                                                 for o in others},
                       "by_pole": dict(Counter(_pole_and_group(b)[0] for b in ids)),
                       "by_group_type": dict(Counter(_pole_and_group(b)[1] for b in ids))}
            if v == "concept-consistent":
                prof[v]["switch_off_here_vs_other"] = {
                    b: {"here": {k: R[b]["roles"]["R7"]["switch_off"][k] for k in ("iia_star", "ci95", "n")},
                        **{o: {k: res[o][b]["roles"]["R7"]["switch_off"][k] for k in ("iia_star", "ci95", "n")}
                           for o in others}} for b in ids}
        by = {"pole": defaultdict(list), "group_type": defaultdict(list)}
        for b, x in R.items():
            a = x["roles"]["R7"].get("switch_off")
            pole, gt = _pole_and_group(b)
            if pole and isinstance(a, dict) and a["n"] >= ix.N_MIN_SWITCH_OFF:
                by["pole"][pole].append(a["iia_star"])
                by["group_type"][gt].append(a["iia_star"])
        prof["switch_off_iia_net_star_median"] = {
            k: {g: {"median": median(v), "concepts": len(v), "undefined": sum(z is None for z in v)}
                for g, v in sorted(d.items())} for k, d in by.items()}
        # removal by pieces on the board, pinned concepts, informative pairs
        rec = torch.load(str(where[name]).replace(".json", "_pairs.pt"),
                         map_location="cpu", weights_only=False)
        cfg_pos = run["positions"]
        npieces = torch.load(ROOT / cfg_pos, map_location="cpu", weights_only=False)["boards"].sum((1, 2, 3)).long()
        curve = {}
        for kind in ("switch_on", "switch_off"):
            acc = defaultdict(lambda: [0, 0, 0])
            for b, x in R.items():
                if x["kind"] != "pinned" or f"{b}|R7|{kind}" not in rec:
                    continue
                y = rec[f"{b}|R7|{kind}"]
                dp, db, ds = (y[k].long() for k in ("dec_patched", "dec_base", "dec_source"))
                n, inf = npieces[y["base"].long()], ds != db
                for p in n.unique().tolist():
                    m = (n == p) & inf
                    a = acc[p]
                    a[0] += int(m.sum()); a[1] += int((dp == ds)[m].sum()); a[2] += int((dp != db)[m].sum())
            curve[kind] = {str(p): {"informative": a[0], "reproduces_D(s)": a[1] / a[0], "leaves_D(b)": a[2] / a[0]}
                           for p, a in sorted(acc.items()) if a[0]}
        prof["pinned_by_pieces_on_board"] = curve
        out[name] = prof
    return out


def main() -> int:
    args = docopt(__doc__)
    paths = args["--runs"].split(",")
    sets = {Path(p).stem: diagnose(ROOT / p) for p in paths}
    strict = [set(s["gate_passing_concepts"]) for s in sets.values()]
    out = {
        "status": ("POST-HOC DIAGNOSTICS OF FAILED GATES (Wave 1b pre-registration S10). "
                   "R7 and representation-free quantities only; R1-R6 SEALED in every set."),
        "sets": sets,
        "concept_consistent_in_every_set": sorted(set.intersection(*strict)) if strict else [],
        "passing_profile": passing_profile([ROOT / p for p in paths]),
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
