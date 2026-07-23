"""Linear probe baseline: logistic regression on raw activations for BSP classification.

Trains per-BSP L2-regularized logistic regression probes on raw activations and
reports per-BSP F1, MCC and F1-lift, plus per-category and overall coverage
metrics. Establishes the upper bound any SAE can achieve on these activations.

Output format mirrors ``sae_eval`` so LP vs SAE numbers are directly comparable:

    coverage          mean of per-BSP best F1 (literature standard)
    coverage_mcc      mean of per-BSP best MCC (base-rate-invariant; 0 for any
                      constant predictor; collapses the F1 ~= 0.667 artifact)
    coverage_f1_lift  mean of max(0, F1 - 2p/(1+p)), where p is the population
                      base rate. Headline metric for trained-vs-random gaps.

Usage:
    linear_probe_baseline.py <activations> <bsp_labels> <bsp_schema> [options]
    linear_probe_baseline.py (-h | --help)

Arguments:
    <activations>    Activation .pt file (N, d_act)
    <bsp_labels>     BSP label .pt file (N, num_bsps)
    <bsp_schema>     BSP schema JSON file

Options:
    -h --help              Show this help message
    --test-frac=<f>        Fraction held out for test [default: 0.2]
    --seed=<int>           Random seed [default: 42]
    --max-iter=<int>       Max iterations for solver [default: 1000]
    --C=<float>            Inverse regularization strength [default: 1.0]
    --output=<path>        Save results JSON to this path [default: auto]

Examples:
    linear_probe_baseline.py data/quarto/fc1_amalgam_activations.pt data/quarto/bsp_labels-gorilla_164.pt data/quarto/bsp_schema-gorilla_164.json

    linear_probe_baseline.py data/quarto/s4.conv2_amalgam_s4_activations.pt data/quarto/bsp_labels-gorillaS4_164.pt data/quarto/bsp_schema-gorillaS4_164.json
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm

try:
    from docopt import docopt
except ImportError:
    print("Install docopt: pip install docopt", file=sys.stderr)
    sys.exit(1)


def _infer_bsp_set_name(bsp_path: Path, schema: dict) -> str:
    """Infer a stable BSP set name for metadata and auto-generated outputs."""
    bsp_set_name = schema.get("bsp_set_name")
    if bsp_set_name:
        return str(bsp_set_name)

    stem = bsp_path.stem
    prefix = "bsp_labels-"
    if stem.startswith(prefix):
        return stem[len(prefix) :]
    return stem


def _default_output_path(act_path: Path, bsp_path: Path, schema: dict) -> Path:
    """Build the default result path.

    Including the BSP set name avoids collisions when the same activation file is
    probed against multiple BSP subsets.
    """
    bsp_set_name = _infer_bsp_set_name(bsp_path, schema)
    return act_path.parent / f"linear_probe_{bsp_set_name}_{act_path.stem}_results.json"


def _scores_from_confusion(
    tn: int, fp: int, fn: int, tp: int
) -> tuple[float, float]:
    """Return (F1, MCC) from confusion-matrix counts; matches lib/sae/eval.py."""
    if tp + fp + fn + tn == 0:
        return 0.0, 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    mcc_den = math.sqrt(
        max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0.0)
    )
    mcc = (tp * tn - fp * fn) / mcc_den if mcc_den > 0 else 0.0
    return float(f1), float(mcc)


def main():
    args = docopt(__doc__)

    act_path = Path(args["<activations>"])
    bsp_path = Path(args["<bsp_labels>"])
    schema_path = Path(args["<bsp_schema>"])
    test_frac = float(args["--test-frac"])
    seed = int(args["--seed"])
    max_iter = int(args["--max-iter"])
    C = float(args["--C"])
    output_path = args["--output"]

    # --- Load data ---
    print("Loading activations...", file=sys.stderr)
    act_data = torch.load(act_path, map_location="cpu", weights_only=False)
    if isinstance(act_data, dict) and "activations" in act_data:
        X = act_data["activations"].numpy()
    elif isinstance(act_data, torch.Tensor):
        X = act_data.numpy()
    else:
        print(f"ERROR: Unexpected activation format: {type(act_data)}", file=sys.stderr)
        sys.exit(1)

    print("Loading BSP labels...", file=sys.stderr)
    bsp_data = torch.load(bsp_path, map_location="cpu", weights_only=False)
    if isinstance(bsp_data, dict) and "labels" in bsp_data:
        Y = bsp_data["labels"].numpy()
    elif isinstance(bsp_data, torch.Tensor):
        Y = bsp_data.numpy()
    else:
        print(f"ERROR: Unexpected BSP label format: {type(bsp_data)}", file=sys.stderr)
        sys.exit(1)

    with open(schema_path) as f:
        schema = json.load(f)

    bsp_defs = schema["bsps"]
    categories = schema["categories"]
    num_bsps = len(bsp_defs)

    assert X.shape[0] == Y.shape[0], f"Sample mismatch: {X.shape[0]} vs {Y.shape[0]}"
    assert Y.shape[1] == num_bsps, f"BSP mismatch: {Y.shape[1]} vs {num_bsps}"

    print(f"Activations: {X.shape}, BSPs: {Y.shape}", file=sys.stderr)

    # Population base rate per BSP (used for the trivial-baseline F1).
    # Using the full Y matches sae_eval's labels.mean(dim=0) convention so that
    # trivial_f1 is a population property, not a sample of the test split.
    base_rates = Y.mean(axis=0)

    # --- Train/test split ---
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_frac, random_state=seed
    )
    print(
        f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}",
        file=sys.stderr,
    )

    # --- Fit per-BSP logistic regression ---
    results_per_bsp = []
    t0 = time.time()

    for i in tqdm(range(num_bsps), desc="Probing BSPs", file=sys.stderr):
        bsp_id = bsp_defs[i]["id"]
        category = bsp_defs[i]["category"]
        y_train_i = Y_train[:, i]
        y_test_i = Y_test[:, i]

        base_rate = float(base_rates[i])
        trivial_f1 = 2.0 * base_rate / (1.0 + base_rate) if base_rate > 0 else 0.0

        # Constant-label case: a constant predictor scores MCC = 0 and
        # F1-lift = 0 by definition. F1 stays the literature value.
        if y_train_i.sum() == 0 or y_train_i.sum() == len(y_train_i):
            f1 = 0.0 if y_test_i.sum() == 0 else 1.0
            results_per_bsp.append(
                {
                    "bsp_id": bsp_id,
                    "category": category,
                    "f1": round(f1, 4),
                    "mcc": 0.0,
                    "f1_lift": 0.0,
                    "base_rate": round(base_rate, 4),
                    "trivial_f1": round(trivial_f1, 4),
                    "skipped": True,
                    "train_pos_rate": round(float(y_train_i.mean()), 4),
                }
            )
            continue

        clf = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver="lbfgs",
            random_state=seed,
        )
        clf.fit(X_train, y_train_i)
        y_pred = clf.predict(X_test)

        cm = confusion_matrix(y_test_i, y_pred, labels=[0, 1])
        tn, fp, fn, tp = (int(x) for x in cm.ravel())
        f1, mcc = _scores_from_confusion(tn, fp, fn, tp)
        f1_lift = max(0.0, f1 - trivial_f1)

        results_per_bsp.append(
            {
                "bsp_id": bsp_id,
                "category": category,
                "f1": round(f1, 4),
                "mcc": round(mcc, 4),
                "f1_lift": round(f1_lift, 4),
                "base_rate": round(base_rate, 4),
                "trivial_f1": round(trivial_f1, 4),
                "skipped": False,
                "train_pos_rate": round(float(y_train_i.mean()), 4),
            }
        )

    total_time = time.time() - t0
    print(f"Done in {total_time:.1f}s", file=sys.stderr)

    # --- Aggregate by category ---
    per_category: dict[str, dict[str, float]] = {}
    for cat in categories:
        f1s = np.array([r["f1"] for r in results_per_bsp if r["category"] == cat])
        if f1s.size == 0:
            continue
        mccs = np.array([r["mcc"] for r in results_per_bsp if r["category"] == cat])
        lifts = np.array(
            [r["f1_lift"] for r in results_per_bsp if r["category"] == cat]
        )
        brs = np.array(
            [r["base_rate"] for r in results_per_bsp if r["category"] == cat]
        )
        per_category[cat] = {
            "count": int(f1s.size),
            "mean_f1": round(float(f1s.mean()), 4),
            "min_f1": round(float(f1s.min()), 4),
            "max_f1": round(float(f1s.max()), 4),
            "median_f1": round(float(np.median(f1s)), 4),
            "mean_mcc": round(float(mccs.mean()), 4),
            "median_mcc": round(float(np.median(mccs)), 4),
            "mean_f1_lift": round(float(lifts.mean()), 4),
            "mean_base_rate": round(float(brs.mean()), 4),
        }

    all_f1s = np.array([r["f1"] for r in results_per_bsp])
    all_mccs = np.array([r["mcc"] for r in results_per_bsp])
    all_lifts = np.array([r["f1_lift"] for r in results_per_bsp])
    overall = {
        "coverage": round(float(all_f1s.mean()), 4),
        "coverage_above_50": round(float((all_f1s > 0.50).mean()), 4),
        "coverage_above_75": round(float((all_f1s > 0.75).mean()), 4),
        "num_bsps": num_bsps,
        "min_f1": round(float(all_f1s.min()), 4),
        "max_f1": round(float(all_f1s.max()), 4),
        "median_f1": round(float(np.median(all_f1s)), 4),
        "coverage_mcc": round(float(all_mccs.mean()), 4),
        "coverage_mcc_above_25": round(float((all_mccs > 0.25).mean()), 4),
        "coverage_mcc_above_50": round(float((all_mccs > 0.50).mean()), 4),
        "median_mcc": round(float(np.median(all_mccs)), 4),
        "min_mcc": round(float(all_mccs.min()), 4),
        "max_mcc": round(float(all_mccs.max()), 4),
        "coverage_f1_lift": round(float(all_lifts.mean()), 4),
        "coverage_f1_lift_above_10": round(float((all_lifts > 0.10).mean()), 4),
        "coverage_f1_lift_above_25": round(float((all_lifts > 0.25).mean()), 4),
        "median_f1_lift": round(float(np.median(all_lifts)), 4),
    }

    # --- Output ---
    output = {
        "overall": overall,
        "per_category": per_category,
        "per_bsp": results_per_bsp,
        "config": {
            "activations": str(act_path),
            "bsp_labels": str(bsp_path),
            "bsp_set": _infer_bsp_set_name(bsp_path, schema),
            "C": C,
            "test_frac": test_frac,
            "seed": seed,
            "max_iter": max_iter,
            "n_train": X_train.shape[0],
            "n_test": X_test.shape[0],
        },
    }

    if output_path == "auto" or output_path is None:
        output_path = _default_output_path(act_path, bsp_path, schema)
    else:
        output_path = Path(output_path)

    with open(output_path, "w") as f:
        # sort_keys for reproducibility: the per_category dict is otherwise
        # emitted in a non-deterministic key order and churns git on re-runs.
        json.dump(output, f, indent=2, sort_keys=True)

    # --- Print summary ---
    print("\n=== Linear Probe Baseline ===", file=sys.stderr)
    print(
        f"coverage (F1)       : {overall['coverage']:.4f}  "
        f"(above50={overall['coverage_above_50']:.1%}, "
        f"above75={overall['coverage_above_75']:.1%})",
        file=sys.stderr,
    )
    print(
        f"coverage_mcc        : {overall['coverage_mcc']:.4f}  "
        f"(above25={overall['coverage_mcc_above_25']:.1%}, "
        f"above50={overall['coverage_mcc_above_50']:.1%})",
        file=sys.stderr,
    )
    print(
        f"coverage_f1_lift    : {overall['coverage_f1_lift']:.4f}  "
        f"(above10={overall['coverage_f1_lift_above_10']:.1%}, "
        f"above25={overall['coverage_f1_lift_above_25']:.1%})",
        file=sys.stderr,
    )
    print("\nPer-category (F1 / MCC / F1-lift):", file=sys.stderr)
    for cat, stats in per_category.items():
        print(
            f"  {cat:25s}  n={stats['count']:3d}  "
            f"F1={stats['mean_f1']:.4f}  "
            f"MCC={stats['mean_mcc']:.4f}  "
            f"lift={stats['mean_f1_lift']:.4f}",
            file=sys.stderr,
        )
    print(f"\nResults saved to: {output_path}", file=sys.stderr)

    # Also print JSON to stdout for programmatic consumption
    json.dump(output["overall"], sys.stdout)
    print()


if __name__ == "__main__":
    main()
