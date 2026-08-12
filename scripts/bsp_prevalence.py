"""Prevalence (base rate) of every BSP, by category.

Prevalence is not a nuisance parameter -- it decides how hard a concept is to
score and which metrics are comparable (see docs/methods-reference.md S1.1). This
gives the whole picture in one table: which categories are rare, which are
saturated, and which are so degenerate that they should not be in a headline.

Flags raised per category:
  DEGENERATE    prevalence < 0.001 or > 0.999 -- essentially constant; no
                classifier can score above chance in any useful sense.
  VERY RARE     prevalence < 0.01 -- MCC/R2 are compressed toward 0 by the
                prevalence factor alone; never compare across populations
                without standardising.
  TRIVIAL-F1    prevalence > 0.4 -- the "always true" classifier already scores
                F1 = 2p/(1+p) > 0.57, so F1 is nearly meaningless here (this is
                the offered_piece F1 = 0.667 artefact).

Usage:
    bsp_prevalence.py --bsps=<name>... [options]
    bsp_prevalence.py (-h | --help)

Options:
    -h --help        Show this help message.
    --bsps=<name>    BSP set(s), repeatable, e.g. --bsps=gorillaYb --bsps=tigerYb
    --game=<name>    Game name [default: quarto]
    --per-bsp        Also list every individual BSP, not just category summaries.
    --output=<path>  Output JSON [default: auto]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docopt import docopt

from lib.sae.eval import resolve_schema_path


def flags_for(p: float) -> list[str]:
    out = []
    if p < 0.001 or p > 0.999:
        out.append("DEGENERATE")
    elif p < 0.01:
        out.append("VERY RARE")
    if p > 0.4:
        out.append("TRIVIAL-F1")
    return out


def analyse(game: str, bsps: str, per_bsp: bool) -> dict:
    data_dir = Path(f"data/{game}")
    matches = sorted(data_dir.glob(f"bsp_labels-{bsps}_[0-9]*.pt"))
    if not matches:
        print(f"  [SKIP] no labels for {bsps}", file=sys.stderr)
        return {}
    labels = torch.load(matches[0], map_location="cpu", weights_only=False)
    schema = json.loads(Path(resolve_schema_path(data_dir, bsps)).read_text(encoding="utf-8"))
    defs = schema["bsps"]
    rates = labels.float().mean(dim=0).numpy()

    by_cat: dict[str, list[tuple[str, float]]] = {}
    for i, b in enumerate(defs):
        by_cat.setdefault(b["category"], []).append((b["id"], float(rates[i])))

    print(f"\n### {bsps}   N={labels.shape[0]}   {len(defs)} BSPs")
    print(f"{'category':<32}{'n':>4}{'min':>9}{'median':>9}{'max':>9}"
          f"{'trivF1':>8}  flags")
    print("-" * 88)
    cats = {}
    for cat in sorted(by_cat, key=lambda c: np.median([v for _, v in by_cat[c]])):
        vals = np.array([v for _, v in by_cat[cat]])
        med = float(np.median(vals))
        fl = sorted({f for v in vals for f in flags_for(float(v))})
        print(f"{cat:<32}{len(vals):>4}{vals.min():>9.4f}{med:>9.4f}"
              f"{vals.max():>9.4f}{2 * med / (1 + med):>8.3f}  {','.join(fl)}")
        cats[cat] = {"n": len(vals), "min": float(vals.min()), "median": med,
                     "max": float(vals.max()),
                     "trivial_f1_at_median": round(2 * med / (1 + med), 4),
                     "flags": fl}
        if per_bsp:
            for bid, v in sorted(by_cat[cat], key=lambda x: x[1]):
                mark = ",".join(flags_for(v))
                print(f"    {bid:<52}{v:>9.4f}  {mark}")
    print("-" * 88)
    allv = rates
    print(f"{'ALL':<32}{len(allv):>4}{allv.min():>9.4f}"
          f"{float(np.median(allv)):>9.4f}{allv.max():>9.4f}")
    n_rare = int((allv < 0.01).sum())
    n_deg = int(((allv < 0.001) | (allv > 0.999)).sum())
    n_triv = int((allv > 0.4).sum())
    print(f"  VERY RARE (<0.01): {n_rare}/{len(allv)}   "
          f"DEGENERATE: {n_deg}   TRIVIAL-F1 (>0.4): {n_triv}")
    return {"bsp_set": bsps, "n_rows": int(labels.shape[0]),
            "n_bsps": len(defs), "categories": cats,
            "n_very_rare": n_rare, "n_degenerate": n_deg, "n_trivial_f1": n_triv}


def main():
    args = docopt(__doc__)
    results = {b: analyse(args["--game"], b, args["--per-bsp"])
               for b in args["--bsps"]}
    out = args["--output"]
    if out == "auto":
        out = Path(f"saes/{args['--game']}/analysis/bsp_prevalence.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
