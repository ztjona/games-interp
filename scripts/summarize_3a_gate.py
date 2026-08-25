"""Combine the per-run 3A reports into 3A_gate_summary.json and print the table.

Extracted from a heredoc inside runners/3A-dilution.ps1 on 2026-08-17, after the
runner completed all 17 cells (2h12m) and then died writing the summary: the
embedded script still read the `n_diluted` / `n_tiled` gate keys that rule 3A.3
had removed. Logic that can break a run belongs in a tested entry point the
runner merely calls -- lessons.md 3.1, "PowerShell should sequence, never
decide".

Usage:
    summarize_3a_gate.py [options]
    summarize_3a_gate.py (-h | --help)

Options:
    -h --help          Show this help message.
    --game=<name>      Game name [default: quarto]
    --expect-rule=<v>  Warn if any report was verdicted by a different rule
                       [default: 3A.3]
    --output=<path>    Output JSON [default: auto]
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path
from statistics import median

from docopt import docopt

# `spread` is the rule-3A.3 geometric verdict; the retired labels are still
# counted so a not-yet-reclassified report summarises rather than crashes.
GEOMETRIC = ("spread", "diluted", "tiled")


def summarize_run(path: str) -> dict:
    with open(path) as f:
        d = json.load(f)
    s, g = d["summary"], d["gate_g3a"]
    cs = [c for c in d["concepts"] if "asymptote_r2" in c]
    med = lambda key, fn=lambda c, k: c[k]: (  # noqa: E731
        round(median([fn(c, key) for c in cs]), 4) if cs else 0.0)
    n_geo = g.get("n_spread")
    if n_geo is None:  # pre-3A.3 report
        n_geo = g.get("n_diluted", 0) + g.get("n_tiled", 0)
    return {
        "run_id": s["run_id"], "bsp_set": s["bsp_set"],
        "rule_version": s["config"].get("rule_version", "3A.1"),
        "random_control": s.get("random_control"),
        "random_control_coverage": g.get("random_control_coverage"),
        "gate_is_provisional": g.get("gate_is_provisional"),
        "n_threat_bsps": g["n_threat_bsps"], "n_spread": n_geo,
        "n_captured": g["n_captured"], "n_absent": g["n_absent"],
        "geometric_frac": g["geometric_frac"],
        "median_solo_frac": (round(median([c.get("solo_frac", 0.0) for c in cs]), 4)
                             if cs else 0.0),
        "median_intrinsic_dim": med("intrinsic_dim"),
        "median_top_phi": (round(median([abs(c["top_phi"]) for c in cs]), 4)
                           if cs else 0.0),
        "mean_asymptote_r2": (round(sum(c["asymptote_r2"] for c in cs) / len(cs), 4)
                              if cs else 0.0),
        "verdict": g["verdict"],
        "deprioritized_because": g.get("deprioritized_because"),
        # Rule 3A.4 band. None on a pre-3A.4 report, which is why every
        # consumer below must tolerate a missing value rather than default it
        # to 0 -- "not banded" and "banded, nothing undecided" are opposite
        # readings of the same run.
        "n_undecided": g.get("n_undecided"),
        "undecided_frac": g.get("undecided_frac"),
        "geometric_frac_lo": g.get("geometric_frac_lo"),
        "geometric_frac_hi": g.get("geometric_frac_hi"),
        "gate_verdict_is_stable": g.get("gate_verdict_is_stable"),
    }


def main() -> int:
    args = docopt(__doc__)
    analysis = Path("saes") / args["--game"] / "analysis"
    out = (analysis / "3A_gate_summary.json" if args["--output"] == "auto"
           else Path(args["--output"]))

    paths = sorted(glob.glob(str(analysis / "*_dilution-*.json")))
    if not paths:
        print(f"error: no dilution reports in {analysis}", file=sys.stderr)
        return 1
    rows = [summarize_run(p) for p in paths]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"runs": rows}, indent=2, sort_keys=True),
                   encoding="utf-8")

    hdr = (f"{'run_id':<50}{'bsps':<11}{'geom':>6}{'[lo,hi]':>13}{'und':>6}"
           f"{'solo':>6}{'idim':>6}{'|phi|':>7}{'R2':>7}{'ctrl':>6}  verdict")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        cov = r.get("random_control_coverage")
        band = ("    not banded" if r.get("geometric_frac_lo") is None
                else f"[{r['geometric_frac_lo']:.2f},{r['geometric_frac_hi']:.2f}]".rjust(13))
        und = ("   -" if r.get("undecided_frac") is None
               else f"{r['undecided_frac']*100:>5.0f}%")
        print(f"{r['run_id'][:49]:<50}{r['bsp_set']:<11}{r['geometric_frac']:>6.2f}"
              f"{band}{und:>6}"
              f"{r['median_solo_frac']:>6.2f}{r['median_intrinsic_dim']:>6.2f}"
              f"{r['median_top_phi']:>7.3f}{r['mean_asymptote_r2']:>7.3f}"
              f"{'  n/a' if cov is None else f'{cov:>5.0%}'}  {r['verdict']}"
              f"{'' if not r.get('deprioritized_because') else ' (' + r['deprioritized_because'] + ')'}")

    rules = {r["rule_version"] for r in rows}
    expect = args["--expect-rule"]
    if rules != {expect}:
        print(f"\nWARNING: mixed or stale verdict rules {sorted(rules)} "
              f"(expected {expect}). Run: python scripts/dilution_diagnostic.py "
              f"reclassify saes/{args['--game']}/analysis/*_dilution-*.json")

    straddle = [r for r in rows if r.get("gate_verdict_is_stable") is False]
    if straddle:
        print(f"\n!! {len(straddle)}/{len(rows)} runs have a stability band that "
              f"STRADDLES the 0.50 gate: geometric_frac is not determined for "
              f"them at +/-band_sds sd, so their verdict must not be quoted as "
              f"a result. (See the 2026-08-21 bimodality retraction for why "
              f"the band is required. NOTE lo/hi is a WORST-CASE union bound: "
              f"it resolves every undecided concept the same way at once, so "
              f"it is 3-10x wider than the measured cross-seed range and is "
              f"vacuous at small n -- tiger has 23 concepts, so each is 0.043 "
              f"of geometric_frac.)")
        for r in straddle:
            print(f"     {r['run_id'][:44]:<46}{r['bsp_set']:<11}"
                  f"[{r['geometric_frac_lo']:.2f}, {r['geometric_frac_hi']:.2f}]")

    prov = [r for r in rows if r.get("gate_is_provisional")]
    if prov:
        print(f"\n!! {len(prov)}/{len(rows)} runs are PROVISIONAL: not every concept "
              f"was tested against a random-model control, so their `absent` counts "
              f"mean 'the learned-signal test did not run', not 'no concept failed'.")
        for r in prov:
            cov = r.get("random_control_coverage")
            print(f"     {r['run_id'][:44]:<46}{r['bsp_set']:<11}"
                  f"coverage {'n/a' if cov is None else f'{cov:.0%}'}")

    print("\nSaved:", out)
    print("geom = spread/threat BSPs; solo = median share of recoverable signal in")
    print("ONE latent; idim = median effective dims; R2 = mean total recoverable")
    print("signal; ctrl = share of concepts tested against the random-model floor.")
    print("Full glossary: the 'glossary' key in each report JSON.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
