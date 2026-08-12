"""Compare BSP bases (gorilla / hawk / tiger) at MATCHED prevalence.

The 2026-05-22 reframing audit chose tiger over hawk as the supervision target
and redirected the programme. That comparison used prevalence-dependent metrics
across bases whose base rates differ by ~7x (hawk `completable` ~0.003 vs tiger
conjunctions ~0.022), so its numerical margin cannot be trusted as stated.

This re-runs the comparison on ONE SAE and ONE set of codes -- so the model, the
positions and the dictionary are identical and only the CONCEPT FRAMING changes
-- reporting raw MCC beside the two prevalence-fair numbers (Youden's J, and MCC
standardised to p_ref). Features are selected by `mcc_at_pref` -- the SELECTION
criterion must match the reported metric. Selecting by J instead walks into J's
blind spot: at base rate 0.003 it picks a latent firing on 10% of positions
(J = 0.85, precision = 0.07), which flatters the rarest families.

It also prints the cross-basis TRIADS: the same underlying game fact described at
three levels of agent-relativity (state / recount / agent-relative). Those triads,
not the whole-basis averages, are the meaningful comparison -- a basis is a
packaging convention, whereas a category is a concept.

Usage:
    basis_comparison.py --run-id=<id> --champ=<tag> [options]
    basis_comparison.py (-h | --help)

Options:
    -h --help        Show this help message.
    --run-id=<id>    SAE run id (needs its _h cache).
    --champ=<tag>    Champion suffix, e.g. Yb (selects gorillaYb/hawkYb/tigerYb).
    --game=<name>    Game name [default: quarto]
    --select-by=<m>  Feature-selection criterion: mcc_at_pref | j | mcc
                     [default: mcc_at_pref]. Must match what you report --
                     selecting by J picks liberal, low-precision latents.
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

from lib.sae.eval import (
    P_REF_DEFAULT,
    derive_triads,
    mcc_from_rates,
    resolve_schema_path,
)

# The triads -- the same underlying game fact described at three levels of
# agent-relativity -- are DERIVED from the `concept_family` / `family_role`
# fields the schemas carry, not hardcoded here. That keeps the cross-basis
# correspondence a property of the data (one mapping, in the game module's
# CONCEPT_FAMILIES, stamped onto every schema), so it cannot drift between this
# script and anything else that groups by family.
#
# If a schema predates the stamp, run:
#     python scripts/stamp_concept_families.py


def per_bsp_stats(fires: torch.Tensor, labels: torch.Tensor,
                  select_by: str = "mcc_at_pref") -> list[dict]:
    """For each BSP: the best feature under ``select_by``, and its rates."""
    n = float(fires.shape[0])
    y = labels.float()
    tp = fires.T @ y
    fp = fires.sum(0, keepdim=True).T - tp
    fn = y.sum(0, keepdim=True) - tp
    tn = n - tp - fp - fn
    eps = 1e-12
    tpr = tp / (tp + fn + eps)
    fpr = fp / (fp + tn + eps)
    j = tpr - fpr
    mcc_num = tp * tn - fp * fn
    mcc = mcc_num / torch.sqrt(((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)).clamp(min=eps))
    std = mcc_from_rates(tpr, 1.0 - fpr, P_REF_DEFAULT)
    precision = tp / (tp + fp + eps)
    freq = fires.mean(dim=0)

    # SELECT BY mcc_at_pref, not by J. Selecting by J walks straight into J's
    # documented blind spot: J weights sensitivity and specificity equally
    # regardless of prevalence, so at base rate 0.003 it happily picks a latent
    # that fires on 10% of positions (J = 0.87, precision = 0.07). The selection
    # criterion must match the reported metric, and mcc_at_pref is both
    # prevalence-fair AND precision-aware -- so it is used for both.
    best = {"mcc_at_pref": std, "j": j, "mcc": mcc}[select_by].argmax(dim=0)
    out = []
    for k in range(labels.shape[1]):
        i = int(best[k])
        out.append({
            "base_rate": float(y[:, k].mean()),
            "j": float(j[i, k]),
            "mcc": float(mcc[i, k]),
            "mcc_at_pref": float(std[i, k]),
            "precision": float(precision[i, k]),
            "fire_rate": float(freq[i]),
        })
    return out


def main():
    args = docopt(__doc__)
    game, run_id, champ = args["--game"], args["--run-id"], args["--champ"]
    data_dir = Path(f"data/{game}")

    h_path = Path(f"saes/{game}/cache/{run_id}_h.pt")
    if not h_path.exists():
        sys.exit(f"error: no _h cache for {run_id}")
    h = torch.load(h_path, map_location="cpu", weights_only=True)
    fires = (h > 0).float()
    del h

    print(f"run: {run_id}")
    print(f"ONE SAE, ONE set of codes -- only the concept framing changes.")
    print(f"Features selected by: {args['--select-by']}")
    print(f"p_ref = {P_REF_DEFAULT}\n")

    by_cat: dict[str, dict] = {}
    schemas: dict[str, dict] = {}
    for basis in ("gorilla", "hawk", "tiger"):
        bsps = f"{basis}{champ}"
        lbl = sorted(data_dir.glob(f"bsp_labels-{bsps}_[0-9]*.pt"))
        if not lbl:
            print(f"  [SKIP] no labels for {bsps}", file=sys.stderr)
            continue
        labels = torch.load(lbl[0], map_location="cpu", weights_only=False)
        schema = json.loads(Path(resolve_schema_path(data_dir, bsps)).read_text(encoding="utf-8"))
        schemas[basis] = schema
        stats = per_bsp_stats(fires, labels, args["--select-by"])
        for b, st in zip(schema["bsps"], stats):
            by_cat.setdefault(b["category"], {"basis": basis, "rows": []})["rows"].append(st)

    print(f"{'category':<30}{'basis':<8}{'n':>3}{'base':>8}{'MCC':>7}"
          f"{'J':>7}{'MCC@pref':>10}{'prec':>7}{'fire':>7}")
    print("-" * 87)
    summary = {}
    for cat in sorted(by_cat, key=lambda c: (by_cat[c]["basis"], c)):
        rows = by_cat[cat]["rows"]
        rec = {k: float(np.mean([r[k] for r in rows]))
               for k in ("base_rate", "mcc", "j", "mcc_at_pref",
                         "precision", "fire_rate")}
        rec["n"] = len(rows)
        rec["basis"] = by_cat[cat]["basis"]
        summary[cat] = rec
        print(f"{cat:<30}{rec['basis']:<8}{rec['n']:>3}{rec['base_rate']:>8.4f}"
              f"{rec['mcc']:>7.3f}{rec['j']:>7.3f}{rec['mcc_at_pref']:>10.4f}"
              f"{rec['precision']:>7.3f}{rec['fire_rate']:>7.3f}")
    print("-" * 87)

    triads = derive_triads(schemas)
    if not triads:
        print("\nNo cross-basis triads: the schemas carry no concept_family "
              "stamp.\nRun: python scripts/stamp_concept_families.py",
              file=sys.stderr)

    print("\nCROSS-BASIS TRIADS -- the same game fact, three framings.")
    print("This is the comparison the 2026-05-22 audit was really making.")
    print("Families derived from the schemas' concept_family stamp.\n")
    triad_out = {}
    for name, mapping in triads.items():
        print(f"  {name}")
        print(f"    {'framing':<28}{'base':>8}{'MCC':>8}{'J':>8}{'MCC@pref':>10}")
        vals = {}
        for basis, cat in mapping.items():
            if cat not in summary:
                continue
            r = summary[cat]
            vals[basis] = r
            print(f"    {basis + ' / ' + cat:<30}{r['base_rate']:>8.4f}"
                  f"{r['mcc']:>7.3f}{r['j']:>7.3f}{r['mcc_at_pref']:>10.4f}"
                  f"{r['precision']:>7.3f}")
        if len(vals) > 1:
            raw_win = max(vals, key=lambda b: vals[b]["mcc"])
            std_win = max(vals, key=lambda b: vals[b]["mcc_at_pref"])
            j_win = max(vals, key=lambda b: vals[b]["j"])
            flag = "" if raw_win == std_win else "   <-- WINNER CHANGES once prevalence is matched"
            print(f"    winner by raw MCC: {raw_win} | by MCC@pref: {std_win} "
                  f"| by J: {j_win}{flag}")
        triad_out[name] = vals
        print()

    out = args["--output"]
    if out == "auto":
        out = Path(f"saes/{game}/analysis/{run_id}_basis-comparison.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(
        {"run_id": run_id, "champ": champ, "p_ref": P_REF_DEFAULT,
         "select_by": args["--select-by"],
         "triad_source": "schema concept_family/family_role",
         "triad_categories": triads,
         "categories": summary, "triads": triad_out}, indent=2), encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
