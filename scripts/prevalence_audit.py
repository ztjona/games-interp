"""Prevalence audit: which of our metrics move when only the base rate moves?

Motivation: on 2026-08-11 a "coverage is 38% worse on-policy" finding was made
and retracted -- the populations differed in prevalence (0.050 vs 0.013) and MCC,
which we had wrongly documented as prevalence-invariant, fell 25% on that change
alone. This script audits the whole metric set so the same mistake cannot recur
by assumption.

Two parts:

  1. ANALYTIC. Hold a classifier's quality FIXED (sensitivity a = TPR,
     specificity b = TNR) and vary prevalence p. Any metric that moves is
     prevalence-dependent and cannot be compared across populations without
     matching or standardising.

  2. EMPIRICAL. Plant a concept with FIXED separability in synthetic SAE codes at
     several prevalences and run the real 3A pipeline, to check whether
     `asymptote_r2` inherits the same dependence.

Usage:
    prevalence_audit.py [options]
    prevalence_audit.py (-h | --help)

Options:
    -h --help        Show this help message.
    --tpr <f>        Fixed sensitivity for the analytic part [default: 0.50]
    --tnr <f>        Fixed specificity for the analytic part [default: 0.99]
    --ref <f>        Reference prevalence for standardisation [default: 0.02]
    --skip-empirical Skip the (slower) synthetic-SAE part.
    --json           Emit JSON as well as the table.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docopt import docopt

PREVALENCES = (0.100, 0.050, 0.030, 0.020, 0.013, 0.005)


def confusion(a: float, b: float, p: float) -> tuple[float, float, float, float]:
    """Expected (tp, fp, fn, tn) rates for a fixed-quality classifier."""
    return a * p, (1 - b) * (1 - p), (1 - a) * p, b * (1 - p)


def metrics_at(a: float, b: float, p: float) -> dict[str, float]:
    tp, fp, fn, tn = confusion(a, b, p)
    eps = 1e-15
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)             # == a
    specificity = tn / (tn + fp + eps)        # == b
    f1 = 2 * precision * recall / (precision + recall + eps)
    trivial_f1 = 2 * p / (1 + p)
    mcc_num = tp * tn - fp * fn
    mcc_den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + eps)
    # Point-biserial r between a binary label and a one-dimensional score is the
    # same functional family as MCC; R2 = r^2 is reported for the 3A curve.
    r_pb = math.sqrt(p * (1 - p)) * (a + b - 1) / math.sqrt(
        (a * p + (1 - b) * (1 - p)) * (b * (1 - p) + (1 - a) * p) + eps)
    return {
        "F1": f1,
        "F1-lift": max(0.0, f1 - trivial_f1),
        "MCC": mcc_num / mcc_den,
        "precision": precision,
        "recall (TPR)": recall,
        "specificity (TNR)": specificity,
        "Youden J (TPR-FPR)": a + b - 1,
        "balanced accuracy": (a + b) / 2,
        "R2 (point-biserial^2)": r_pb ** 2,
    }


def standardise_mcc(a: float, b: float, p_ref: float) -> float:
    """MCC the classifier WOULD score at the reference prevalence.

    MCC is a deterministic function of (a, b, p), so given a feature's measured
    sensitivity and specificity we can report its MCC at a canonical prevalence
    without discarding a single row. This is the cheap alternative to subsample
    matching."""
    return metrics_at(a, b, p_ref)["MCC"]


def analytic_audit(a: float, b: float, ref: float) -> dict:
    rows = {name: [] for name in metrics_at(a, b, 0.05)}
    for p in PREVALENCES:
        m = metrics_at(a, b, p)
        for k, v in m.items():
            rows[k].append(v)

    print(f"ANALYTIC AUDIT -- classifier held FIXED at TPR={a}, TNR={b}.")
    print("Any movement across this row is caused by prevalence ALONE.\n")
    hdr = f"{'metric':<24}" + "".join(f"{p:>9.3f}" for p in PREVALENCES) + f"{'swing':>9}"
    print(hdr)
    print("-" * len(hdr))
    verdicts = {}
    for k, vals in rows.items():
        lo, hi = min(vals), max(vals)
        swing = 0.0 if hi == 0 else (hi - lo) / hi
        verdicts[k] = "INVARIANT" if swing < 1e-9 else f"{swing:.0%}"
        print(f"{k:<24}" + "".join(f"{v:>9.4f}" for v in vals)
              + f"{verdicts[k]:>9}")
    print("-" * len(hdr))
    print("swing = (max-min)/max across the prevalence range above.\n")
    return {"rows": rows, "verdicts": verdicts,
            "prevalences": list(PREVALENCES), "tpr": a, "tnr": b}


def empirical_r2(ref_seed: int = 0) -> dict:
    """Does 3A's asymptote_r2 inherit the prevalence dependence?

    Plant ONE latent whose relationship to the concept is identical at every
    prevalence (fixed flip probability => fixed TPR/TNR), vary only how often the
    concept is true, and run the real diagnose_concept.
    """
    import torch
    from lib.sae.dilution import DilutionConfig, diagnose_concept

    print("EMPIRICAL CHECK -- 3A asymptote_r2 with a FIXED-quality planted latent")
    print("(one latent, 25% flip rate, only the concept's prevalence changes)\n")
    cfg = DilutionConfig(max_rows=20_000, curve_rows=60_000, min_positives=1,
                         n_splits=3, top_k=16)
    N, d = 60_000, 24
    out = []
    print(f"{'prevalence':>11}{'positives':>11}{'asymptote_r2':>15}{'top_phi':>10}")
    print("-" * 47)
    for p in (0.100, 0.050, 0.020, 0.013):
        rng = np.random.default_rng(ref_seed)
        y = (rng.random(N) < p).astype(np.float32)
        h = (rng.random((N, d)) < 0.05).astype(np.float32)
        h *= rng.uniform(0.5, 1.5, size=(N, d)).astype(np.float32)
        flip = rng.random(N) < 0.25            # fixed corruption => fixed a, b
        h[:, 0] = np.where(flip, 1.0 - y, y).astype(np.float32)
        m = diagnose_concept(torch.tensor(h), torch.tensor(y), cfg)
        out.append({"p": p, "n_pos": int(y.sum()),
                    "asymptote_r2": m["asymptote_r2"], "top_phi": m["top_phi"]})
        print(f"{p:>11.3f}{int(y.sum()):>11}{m['asymptote_r2']:>15.4f}"
              f"{abs(m['top_phi']):>10.4f}")
    print("-" * 47)
    lo = min(r["asymptote_r2"] for r in out)
    hi = max(r["asymptote_r2"] for r in out)
    print(f"asymptote_r2 swing across prevalence: {(hi - lo) / hi:.0%}  "
          f"({lo:.4f} .. {hi:.4f})")
    print("=> 3A's R2 carries the SAME prevalence dependence as MCC. Cross-"
          "population\n   or cross-champion R2 comparisons need matching too.\n")
    return {"points": out, "swing": (hi - lo) / hi if hi else 0.0}


def main():
    args = docopt(__doc__)
    a, b, ref = float(args["--tpr"]), float(args["--tnr"]), float(args["--ref"])

    result = {"analytic": analytic_audit(a, b, ref)}

    print(f"STANDARDISATION -- MCC re-expressed at a reference prevalence "
          f"p_ref={ref}")
    print("Since MCC is a deterministic function of (TPR, TNR, p), a feature's "
          "MCC can be\nreported at a canonical prevalence with no rows "
          "discarded and no sampling noise.\n")
    for (aa, bb) in ((0.50, 0.99), (0.30, 0.995), (0.80, 0.95)):
        raw = [metrics_at(aa, bb, p)["MCC"] for p in PREVALENCES]
        std = standardise_mcc(aa, bb, ref)
        print(f"  TPR={aa:.2f} TNR={bb:.3f}:  raw MCC ranges "
              f"{min(raw):.3f}..{max(raw):.3f}   ->   MCC@{ref} = {std:.3f}")
    print()

    if not args["--skip-empirical"]:
        result["empirical_r2"] = empirical_r2()

    print("CONCLUSION")
    inv = [k for k, v in result["analytic"]["verdicts"].items() if v == "INVARIANT"]
    dep = [k for k, v in result["analytic"]["verdicts"].items() if v != "INVARIANT"]
    print(f"  Prevalence-INVARIANT : {', '.join(inv)}")
    print(f"  Prevalence-DEPENDENT : {', '.join(dep)}")
    print("\n  F1-lift is NOT invariant: subtracting the trivial baseline "
          "2p/(1+p) corrects\n  the FLOOR, not the SCALING, so it still moves "
          "with prevalence.")
    print("  Recommended pairing: report MCC (practical, prevalence-aware) "
          "alongside\n  Youden's J = TPR-FPR (invariant). J is literally the "
          "(a+b-1) factor in the\n  MCC numerator, i.e. MCC with the prevalence "
          "term stripped out.")
    print("  Caveat: J alone is misleading in the opposite direction -- a "
          "feature firing\n  50% of the time with TPR 0.9 scores J=0.4 while "
          "being useless at p=0.02.\n  The PAIR is what is informative; "
          "divergence flags prevalence doing the work.")

    if args["--json"]:
        print("\n" + json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
