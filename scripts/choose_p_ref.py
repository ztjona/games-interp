"""Choose the reference prevalence p_ref for standardised MCC, empirically.

MCC@p_ref re-expresses a feature's MCC at a declared prevalence, using the
feature's own measured (TPR, TNR). It is the noise-free analytic equivalent of
subsample matching -- but p_ref is a SUBSTANTIVE choice, not a normalisation:
the ranking of two detectors can flip with it (high-recall detectors win at high
prevalence, high-specificity detectors at low prevalence).

This script picks p_ref on evidence, against three criteria:

  1. RANK STABILITY -- over what range of p_ref does the ordering of our actual
     runs stay fixed? A p_ref inside a wide stable band is a safe choice.
  2. REGIME RELEVANCE -- p_ref should sit in the base-rate regime we actually
     care about (tiger conjunctions, ~0.02).
  3. EXTRAPOLATION DISTANCE -- standardising from p=0.05 out to p_ref=0.001 is a
     long extrapolation from the measured operating point; prefer a p_ref inside
     the observed spread of base rates.

Features are selected by Youden's J, which is prevalence-invariant, so the
SELECTION step cannot itself smuggle prevalence back in.

Usage:
    choose_p_ref.py [options]
    choose_p_ref.py (-h | --help)

Options:
    -h --help        Show this help message.
    --game <name>    Game name [default: quarto]
    --runs <list>    Comma-separated "run_id:bsps" pairs [default: auto]
    --output <path>  Output JSON [default: auto]
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
from scripts.per_mode_coverage import AUTO_MARKERS
from scripts.prevalence_audit import metrics_at

GRID = (0.100, 0.075, 0.050, 0.035, 0.025, 0.020, 0.015, 0.013, 0.010, 0.005, 0.002)

DEFAULT_RUNS = [
    "F04-champYb-s42-jumprelu-t64-exp8-s4.fc1:tigerYb",
    "E05-champYb-s42-batchtopk-k32-exp8-s4.conv2:tigerYb",
    "I04-champYb-lh100-s42-anchored-jumprelu-t64-exp8-s4.fc1:tigerYb",
    "E05-champVe-s42-batchtopk-k32-exp8-s4.conv2:tigerVe",
    "I04-champVe-lh100-s42-anchored-jumprelu-t64-exp8-s4.fc1:tigerVe",
]


def best_by_j(h_path: Path, labels: torch.Tensor) -> list[dict]:
    """Per BSP: the J-maximising feature's measured (TPR, TNR) and base rate."""
    h = torch.load(h_path, map_location="cpu", weights_only=True)
    fires = (h > 0).float()
    del h
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
    best = j.argmax(dim=0)
    out = []
    for k in range(labels.shape[1]):
        i = int(best[k])
        out.append({"tpr": float(tpr[i, k]), "tnr": float(1.0 - fpr[i, k]),
                    "base_rate": float(y[:, k].mean()), "j": float(j[i, k])})
    return out


def main():
    args = docopt(__doc__)
    game = args["--game"]
    pairs = (DEFAULT_RUNS if args["--runs"] == "auto"
             else [p.strip() for p in args["--runs"].split(",") if p.strip()])
    data_dir = Path(f"data/{game}")

    per_run = {}
    for pair in pairs:
        run_id, bsps = pair.rsplit(":", 1)
        h_path = Path(f"saes/{game}/cache/{run_id}_h.pt")
        if not h_path.exists():
            print(f"  [SKIP] no _h cache: {run_id}", file=sys.stderr)
            continue
        lbl = sorted(data_dir.glob(f"bsp_labels-{bsps}_[0-9]*.pt"))
        if not lbl:
            print(f"  [SKIP] no labels: {bsps}", file=sys.stderr)
            continue
        labels = torch.load(lbl[0], map_location="cpu", weights_only=False)
        schema = json.loads(Path(resolve_schema_path(data_dir, bsps)).read_text(encoding="utf-8"))
        keep = [i for i, b in enumerate(schema["bsps"])
                if any(m in b["category"] for m in AUTO_MARKERS)]
        per_run[f"{run_id.split('-')[0]}-{run_id.split('champ')[1][:2]}/{bsps}"] = \
            best_by_j(h_path, labels[:, keep])
        print(f"  loaded {run_id}  ({len(keep)} concepts)")

    if not per_run:
        sys.exit("no runs loaded")

    rates = [c["base_rate"] for v in per_run.values() for c in v]
    print(f"\nObserved base rates across all concepts: "
          f"min {min(rates):.4f}  median {np.median(rates):.4f}  max {max(rates):.4f}")

    # coverage_mcc@p_ref per run over the grid
    names = list(per_run)
    print(f"\n{'p_ref':>8}" + "".join(f"{n[:16]:>18}" for n in names) + "   ranking")
    print("-" * (8 + 18 * len(names) + 12))
    rankings, table = [], {}
    for p in GRID:
        vals = [float(np.mean([metrics_at(c["tpr"], c["tnr"], p)["MCC"]
                               for c in per_run[n]])) for n in names]
        order = tuple(int(i) for i in np.argsort(-np.array(vals)))
        rankings.append(order)
        table[p] = dict(zip(names, [round(v, 4) for v in vals]))
        print(f"{p:>8.3f}" + "".join(f"{v:>18.4f}" for v in vals)
              + "   " + ">".join(str(o + 1) for o in order))
    print("-" * (8 + 18 * len(names) + 12))

    # Rank stability: the widest contiguous band of p_ref sharing one ordering.
    best_band, cur = (0, None, None), None
    for idx, order in enumerate(rankings):
        if order != cur:
            cur, start = order, idx
        span = idx - start + 1
        if span > best_band[0]:
            best_band = (span, GRID[start], GRID[idx])
    print(f"\n[1] RANK STABILITY: widest band with one fixed ordering spans "
          f"p_ref {best_band[2]:.3f}..{best_band[1]:.3f} ({best_band[0]}/{len(GRID)} grid points)")
    flips = sum(1 for a, b in zip(rankings, rankings[1:]) if a != b)
    print(f"    ordering changes {flips} time(s) across the whole grid")

    print(f"[2] REGIME RELEVANCE: the tiger conjunctions sit at "
          f"{np.percentile(rates, 25):.4f}..{np.percentile(rates, 75):.4f} (IQR)")
    print(f"[3] EXTRAPOLATION: keeping p_ref inside "
          f"[{min(rates):.4f}, {max(rates):.4f}] avoids extrapolating beyond "
          f"any measured operating point")

    lo, hi = float(np.percentile(rates, 25)), float(np.percentile(rates, 75))
    inside = [p for p in GRID if lo <= p <= hi]
    rec = min(inside, key=lambda p: abs(p - float(np.median(rates)))) if inside \
        else float(np.median(rates))
    print(f"\nRECOMMENDATION: p_ref = {rec:.3f}  "
          f"(median observed base rate {np.median(rates):.4f}; inside the stable "
          f"band and inside the measured range)")

    out = args["--output"]
    if out == "auto":
        out = Path(f"saes/{game}/analysis/p_ref_selection.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({
        "grid": list(GRID), "runs": names, "table": table,
        "base_rate_stats": {"min": min(rates), "median": float(np.median(rates)),
                            "max": max(rates), "p25": lo, "p75": hi},
        "rank_flips": flips, "recommended_p_ref": rec}, indent=2), encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
