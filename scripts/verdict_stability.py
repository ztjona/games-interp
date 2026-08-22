"""Measure how much a 3A verdict moves on SEED alone, and re-derive the band.

Two numbers come out of here, and they answer different questions:

  * **flip rate** -- the share of per-concept verdicts that are not identical
    across every seed of one condition. This is the GROUND TRUTH for verdict
    stability. Where it exists, quote it; the rule-3A.4 band is only a proxy
    for the runs that have no replicates.
  * **`solo_frac_seed_sd`** -- the per-concept cross-seed sd of `solo_frac`,
    which rule 3A.4 needs as a constant because cross-seed variation cannot be
    estimated from a single run. Reported as median AND mean, because the two
    disagree by ~5x and the choice matters: the distribution is heavily
    right-skewed, and a band is a claim about the tail. `DilutionConfig` uses
    the MEAN. Banding at 3 x median called 96.6% of verdicts confident against
    a measured 12-17% flip rate.

Both are pure functions of the stored reports -- no `_h` caches, no rerun.

Conditions are discovered by stripping `-s<digits>` from each report's run_id,
so any condition with two or more seeds on disk is measured automatically.

Usage:
    verdict_stability.py [<report>...] [options]
    verdict_stability.py (-h | --help)

Arguments:
    <report>           Dilution reports to consider [default: all in analysis/].

Options:
    -h --help          Show this help message.
    --game=<name>      Game name [default: quarto]
    --min-seeds=<n>    Minimum seeds for a condition to be reported [default: 2]
    --output=<path>    Write results as JSON [default: auto]
"""

from __future__ import annotations

import glob
import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from docopt import docopt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib.sae.dilution import DilutionConfig  # noqa: E402

SEED_RE = re.compile(r"-s(\d+)(?=-)")


def condition_of(run_id: str) -> str:
    return SEED_RE.sub("-sSEED", run_id)


def seed_of(run_id: str) -> str:
    m = SEED_RE.search(run_id)
    return m.group(1) if m else "?"


def main() -> int:
    args = docopt(__doc__)
    analysis = ROOT / "saes" / args["--game"] / "analysis"
    paths = args["<report>"] or sorted(
        glob.glob(str(analysis / "*_dilution-*.json")))
    min_seeds = int(args["--min-seeds"])

    # (condition, bsp_set) -> seed -> {bsp_id: concept}
    groups: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for p in paths:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        s = d["summary"]
        key = (condition_of(s["run_id"]), s["bsp_set"])
        groups[key][seed_of(s["run_id"])] = {
            c["bsp_id"]: c for c in d["concepts"]
            if not c.get("degenerate") and "asymptote_r2" in c
        }

    results = {}
    all_sds: list[float] = []
    print(f"{'condition':<46}{'bsps':<11}{'seeds':>6}{'n':>5}"
          f"{'flip%':>8}{'geom range':>14}")
    print("-" * 90)
    for (cond, bsps), by_seed in sorted(groups.items()):
        if len(by_seed) < min_seeds:
            continue
        seeds = sorted(by_seed)
        shared = set.intersection(*(set(v) for v in by_seed.values()))
        flips = 0
        sds: list[float] = []
        for bsp in shared:
            verdicts = {by_seed[s][bsp]["verdict"] for s in seeds}
            if len(verdicts) > 1:
                flips += 1
            vals = [by_seed[s][bsp].get("solo_frac") for s in seeds]
            if all(v is not None for v in vals) and len(vals) > 1:
                # SAMPLE sd (ddof=1). The band asks "how far would a NEW seed
                # land?", so the seeds on disk are a sample and the unbiased
                # estimator is the right one. The population sd (ddof=0) is
                # smaller by sqrt((n-1)/n) -- at n=3 that is 0.82x, which is
                # part of why the originally-specified 0.0110 was too narrow.
                sds.append(st.stdev(vals))
        all_sds += sds
        # geometric_frac per seed, to show the aggregate is steadier than the
        # per-concept verdicts it is built from.
        geo = []
        for s in seeds:
            cs = by_seed[s]
            n_geo = sum(1 for c in cs.values()
                        if c["verdict"] in ("spread", "diluted", "tiled"))
            geo.append(n_geo / len(cs) if cs else 0.0)
        n = len(shared)
        results[f"{cond}|{bsps}"] = {
            "seeds": seeds,
            "n_concepts": n,
            "n_flips": flips,
            "flip_rate": round(flips / n, 4) if n else 0.0,
            "geometric_frac_per_seed": {s: round(g, 4) for s, g in zip(seeds, geo)},
            "geometric_frac_range": round(max(geo) - min(geo), 4) if geo else 0.0,
            "solo_frac_seed_sd": {
                "median": round(st.median(sds), 4) if sds else None,
                "mean": round(st.fmean(sds), 4) if sds else None,
                "p90": round(sorted(sds)[int(0.9 * len(sds))], 4) if sds else None,
                "max": round(max(sds), 4) if sds else None,
            },
        }
        print(f"{cond[:45]:<46}{bsps:<11}{len(seeds):>6}{n:>5}"
              f"{100*flips/n:>7.1f}%"
              f"{f'[{min(geo):.2f},{max(geo):.2f}]':>14}")

    if not results:
        print(f"No condition has >= {min_seeds} seeds on disk.", file=sys.stderr)
        return 1

    cfg = DilutionConfig()
    pooled = {
        "n_concepts": len(all_sds),
        "median": round(st.median(all_sds), 4),
        "mean": round(st.fmean(all_sds), 4),
        "p90": round(sorted(all_sds)[int(0.9 * len(all_sds))], 4),
        "max": round(max(all_sds), 4),
        "in_use": cfg.solo_frac_seed_sd,
    }
    results["_pooled_solo_frac_seed_sd"] = pooled

    print("-" * 90)
    print(f"pooled per-concept solo_frac seed sd over {len(all_sds)} concepts: "
          f"median {pooled['median']}  mean {pooled['mean']}  "
          f"p90 {pooled['p90']}  max {pooled['max']}")
    print(f"DilutionConfig.solo_frac_seed_sd = {cfg.solo_frac_seed_sd} "
          f"({'MEAN -- matches' if abs(pooled['mean'] - cfg.solo_frac_seed_sd) < 5e-4 else 'DOES NOT match the mean; update lib/sae/dilution.py'})")
    print("The MEAN is used deliberately: the sd distribution is right-skewed "
          "(see p90/max), and a band is a claim about the tail.")
    print("Ground truth is the flip% column above -- where seeds exist, quote "
          "that, not the rule-3A.4 band.")

    out = (analysis / "3A_verdict_stability.json" if args["--output"] == "auto"
           else Path(args["--output"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
