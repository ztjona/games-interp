"""SAE-vs-linear-probe efficiency, per concept family -- the hawk-vs-tiger verdict.

The 2026-05-22 reframing audit chose tiger over hawk as the supervision target by
comparing whole-basis averages on prevalence-DEPENDENT metrics across bases whose
base rates differ by ~7x. Two things were conflated and neither was measured:

  1. *Availability* -- is the concept linearly decodable from the activations at
     all? That is the linear probe, and it is a property of the MODEL and the
     CONCEPT FRAMING, not of any SAE.
  2. *Efficiency* -- of the signal that IS available, how much does the
     unsupervised dictionary actually surface? That is SAE / LP.

A basis can win on raw SAE score purely because its concepts are easier to probe
(high LP), which says nothing about whether an SAE finds them. Efficiency
separates the two, and it is the quantity a supervision-target choice should
turn on: prefer the framing whose available signal the dictionary captures.

Everything is reported per CONCEPT FAMILY (line_threat, square_threat, ...) read
from the schema's ``concept_family`` stamp, because whole-basis means mix
unrelated families -- hawk's 173 BSPs are 76 line + 81 square + 2 global, while
tiger's 36 are 10 line + 9 square + 5 global + 12 pool. Comparing those averages
compares different partitions of the concept menu, not different framings.

Metric: ``mcc_at_pref`` (MCC restated at p_ref = 0.025) on BOTH sides. Raw MCC is
not comparable across bases at different base rates, which is precisely the
defect in the original audit. Youden's J is reported beside it as the
prevalence-invariant cross-check.

Inputs (all must already exist -- this script computes no models):
  * LP reports   data/<game>/linear_probe_<basis>_<n>_<act_stem>_results.json
                 (produced by scripts/linear_probe_baseline.py)
  * SAE rows     saes/<game>/eval_registry.json, key ``<run_id>:<basis><Champ>``

Usage:
    sae_lp_efficiency.py --champ=<tag> [options]
    sae_lp_efficiency.py (-h | --help)

Options:
    -h --help          Show this help message.
    --champ=<tag>      Champion suffix, e.g. Yb (selects hawkYb / tigerYb).
    --game=<name>      Game name [default: quarto]
    --hook=<h>         Activation hook [default: s4.fc1]
    --run-id=<id>      SAE run id. Default: auto-detect the registry's best
                       run for this champion+hook by coverage_mcc_at_pref.
    --bases=<list>     Comma-separated bases [default: gorilla,hawk,tiger]
    --act-stem=<s>     Activation file stem used for the LP reports.
                       Default: <hook>_amalgam_<champ lower>_activations
    --output=<path>    Output JSON [default: auto]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from docopt import docopt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.sae.eval import (  # noqa: E402
    aggregate_per_category_by_family,
    derive_triads,
    resolve_schema_path,
)

# Efficiency bands. A ratio near 1 means the dictionary surfaces essentially all
# the linearly available signal; near 0 means the signal is present in the
# activations but the unsupervised SAE does not isolate it.
BANDS = ((0.80, "high"), (0.50, "moderate"), (0.20, "low"))

# Below this the linear probe itself has too little signal for a ratio to mean
# anything: dividing two small numbers produces a loud, meaningless efficiency.
LP_FLOOR = 0.05


def band(ratio: float | None) -> str:
    if ratio is None:
        return "n/a"
    for threshold, name in BANDS:
        if ratio >= threshold:
            return name
    return "negligible"


def load_registry(game: str) -> dict:
    path = Path("saes") / game / "eval_registry.json"
    if not path.exists():
        sys.exit(f"error: registry not found at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def find_lp_report(data_dir: Path, basis: str, act_stem: str) -> Path | None:
    """LP report for a basis, tolerating the BSP-count part of the filename."""
    matches = sorted(data_dir.glob(
        f"linear_probe_{basis}_[0-9]*_{act_stem}_results.json"))
    return matches[0] if matches else None


def main() -> int:
    args = docopt(__doc__)
    game, champ, hook = args["--game"], args["--champ"], args["--hook"]
    bases = [b.strip() for b in args["--bases"].split(",")]
    data_dir = Path("data") / game

    act_stem = args["--act-stem"] or f"{hook}_amalgam_{champ.lower()}_activations"
    registry = load_registry(game)

    run_id = args["--run-id"]
    if not run_id:
        # Best available run for this champion+hook, by the same standardised
        # metric the comparison reports. Selecting on one metric and reporting
        # another is how the original audit went wrong.
        candidates = []
        for key, entry in registry.items():
            rid = key.split(":")[0]
            if f"champ{champ}" not in rid or not rid.endswith(hook):
                continue
            score = (entry.get("metrics") or {}).get("coverage_mcc_at_pref")
            if score is not None:
                candidates.append((score, rid))
        if not candidates:
            sys.exit(
                f"error: no registry row for champ{champ}/{hook} carries "
                f"coverage_mcc_at_pref. Re-evaluate (or backfill) first, then "
                f"re-run; pass --run-id to override.")
        run_id = max(candidates)[1]
        print(f"auto-selected run_id: {run_id}", file=sys.stderr)

    schemas: dict[str, dict] = {}
    per_basis: dict[str, dict] = {}
    problems: list[str] = []

    for basis in bases:
        bsps = f"{basis}{champ}"
        schema_path = resolve_schema_path(data_dir, bsps)
        if schema_path is None:
            problems.append(f"{basis}: no schema on disk")
            continue
        schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
        schemas[basis] = schema

        lp_path = find_lp_report(data_dir, basis, act_stem)
        if lp_path is None:
            problems.append(
                f"{basis}: no LP report (expected "
                f"linear_probe_{basis}_*_{act_stem}_results.json)")
            continue
        lp = json.loads(lp_path.read_text(encoding="utf-8"))
        lp_fams = lp.get("per_family") or aggregate_per_category_by_family(
            lp.get("per_category", {}), schema)
        if not lp_fams:
            problems.append(
                f"{basis}: LP report predates concept families -- re-run "
                f"scripts/linear_probe_baseline.py for it")
            continue
        # An LP report written before mcc_at_pref existed still rolls up into
        # families (the schema supplies those), so the failure would otherwise
        # be silent: every efficiency ratio would come back "n/a" with no reason
        # given. Name it instead.
        if not any("mean_mcc_at_pref" in v for v in lp_fams.values()):
            problems.append(
                f"{basis}: LP report {lp_path.name} predates mcc_at_pref -- "
                f"re-run scripts/linear_probe_baseline.py for it")
            continue

        entry = registry.get(f"{run_id}:{bsps}")
        if entry is None:
            problems.append(f"{basis}: no registry row {run_id}:{bsps}")
            continue
        pc = (entry.get("metrics") or {}).get("per_category")
        if not pc:
            problems.append(f"{basis}: registry row {run_id}:{bsps} has no "
                            f"per_category breakdown")
            continue
        sae_fams = aggregate_per_category_by_family(pc, schema)
        if not any("mean_mcc_at_pref" in v for v in sae_fams.values()):
            problems.append(
                f"{basis}: registry row {run_id}:{bsps} has no "
                f"mean_mcc_at_pref -- re-evaluate with the current sae_eval")
            continue

        rows = {}
        for family in sorted(set(sae_fams) & set(lp_fams)):
            sae, probe = sae_fams[family], lp_fams[family]
            s = sae.get("mean_mcc_at_pref")
            p = probe.get("mean_mcc_at_pref")
            ratio = None
            if s is not None and p is not None and p >= LP_FLOOR:
                ratio = round(s / p, 4)
            rows[family] = {
                "n_bsps": sae.get("count"),
                "sae_mcc_at_pref": s,
                "lp_mcc_at_pref": p,
                "efficiency": ratio,
                "band": band(ratio),
                "sae_youden_j": sae.get("mean_youden_j"),
                "lp_youden_j": probe.get("mean_youden_j"),
                "base_rate": sae.get("mean_base_rate"),
                "lp_below_floor": p is not None and p < LP_FLOOR,
            }
        per_basis[basis] = {"lp_report": str(lp_path),
                            "registry_key": f"{run_id}:{bsps}",
                            "families": rows}

    if problems:
        print("\nMISSING INPUTS -- these bases are excluded:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)

    if len(per_basis) < 2:
        print("\nFewer than two bases have complete inputs; no cross-basis "
              "verdict is possible.", file=sys.stderr)
        return 1

    # --- report ------------------------------------------------------------
    print(f"\nSAE-vs-LP efficiency | champ={champ} hook={hook}")
    print(f"run: {run_id}")
    print("Metric: mcc_at_pref (p_ref=0.025) on both sides. "
          "efficiency = SAE / LP.\n")
    for basis, blk in per_basis.items():
        print(f"  --- {basis} ---")
        print(f"    {'concept_family':22s}{'n':>4}{'SAE':>8}{'LP':>8}"
              f"{'eff':>8}  {'band':<11}{'base':>7}")
        for family, r in sorted(blk["families"].items()):
            eff = "    n/a" if r["efficiency"] is None else f"{r['efficiency']:>8.2f}"
            print(f"    {family:22s}{r['n_bsps']:>4}"
                  f"{(r['sae_mcc_at_pref'] or 0):>8.3f}"
                  f"{(r['lp_mcc_at_pref'] or 0):>8.3f}{eff}  "
                  f"{r['band']:<11}{(r['base_rate'] or 0):>7.3f}")
        print()

    # --- verdict: which framing does the dictionary serve best? -------------
    triads = derive_triads(schemas)
    verdict: dict[str, dict] = {}
    print("VERDICT, family by family (only families present in >1 basis).")
    print("'available' = the LP upper bound; 'captured' = what the SAE gets;")
    print("'efficiency' = captured / available. Prefer the framing that wins on")
    print("EFFICIENCY -- winning on availability alone says the concept is")
    print("decodable, not that the dictionary represents it.\n")
    for family in sorted(triads):
        present = {b: blk["families"][family]
                   for b, blk in per_basis.items()
                   if family in blk["families"]}
        if len(present) < 2:
            continue
        by_avail = max(present, key=lambda b: present[b]["lp_mcc_at_pref"] or 0)
        by_capt = max(present, key=lambda b: present[b]["sae_mcc_at_pref"] or 0)
        scored = {b: r for b, r in present.items() if r["efficiency"] is not None}
        by_eff = max(scored, key=lambda b: scored[b]["efficiency"]) if scored else None
        verdict[family] = {
            "categories": triads[family],
            "best_availability": by_avail,
            "best_captured": by_capt,
            "best_efficiency": by_eff,
            "agrees": by_capt == by_eff,
        }
        flag = "" if by_capt == by_eff else "   <-- captured and efficiency disagree"
        print(f"  {family:22s} available={by_avail:<9} captured={by_capt:<9} "
              f"efficiency={str(by_eff):<9}{flag}")
        for b, r in sorted(present.items()):
            eff = "n/a" if r["efficiency"] is None else f"{r['efficiency']:.2f}"
            note = "  (LP below floor -- ratio suppressed)" if r["lp_below_floor"] else ""
            print(f"      {b:9s} {triads[family].get(b, '?'):30s} "
                  f"SAE={(r['sae_mcc_at_pref'] or 0):.3f} "
                  f"LP={(r['lp_mcc_at_pref'] or 0):.3f} "
                  f"eff={eff}{note}")
        print()

    hawk_tiger = {f: v for f, v in verdict.items()
                  if {"hawk", "tiger"} <= set(per_basis)}
    if hawk_tiger:
        wins = {}
        for v in hawk_tiger.values():
            if v["best_efficiency"] in ("hawk", "tiger"):
                wins[v["best_efficiency"]] = wins.get(v["best_efficiency"], 0) + 1
        print(f"HAWK vs TIGER on efficiency: "
              f"{', '.join(f'{k}={n}' for k, n in sorted(wins.items())) or 'no scorable family'} "
              f"(of {len(hawk_tiger)} shared families)")
        print("This is a per-family tally, NOT a whole-basis average -- the "
              "whole-basis average is what the 2026-05-22 audit used and is "
              "not interpretable.")

    out = args["--output"]
    if out == "auto" or out is None:
        out = Path(f"saes/{game}/analysis/{run_id}_sae-lp-efficiency.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({
        "run_id": run_id, "champ": champ, "hook": hook,
        "metric": "mcc_at_pref", "lp_floor": LP_FLOOR,
        "bands": {name: t for t, name in BANDS},
        "per_basis": per_basis, "triads": triads, "verdict": verdict,
        "excluded": problems,
    }, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
