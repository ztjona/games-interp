"""Linear probe baseline: logistic regression on raw activations for BSP classification.

Trains per-BSP L2-regularized logistic regression probes on raw activations and
reports per-BSP F1, per-category mean F1, and overall coverage. This establishes
an upper bound for what any SAE can achieve on these activations.

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
    python scripts/linear_probe_baseline.py \\
        data/quarto/fc1_amalgam_activations.pt \\
        data/quarto/bsp_labels-gorilla_164.pt \\
        data/quarto/bsp_schema-gorilla_164.json

    python scripts/linear_probe_baseline.py \\
        data/quarto/fc1_amalgam_random_activations.pt \\
        data/quarto/bsp_labels-gorilla_164.pt \\
        data/quarto/bsp_schema-gorilla_164.json \\
        --output data/quarto/linear_probe_random_results.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

try:
    from docopt import docopt
except ImportError:
    print("Install docopt: pip install docopt", file=sys.stderr)
    sys.exit(1)


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

    for i in range(num_bsps):
        bsp_id = bsp_defs[i]["id"]
        category = bsp_defs[i]["category"]
        y_train_i = Y_train[:, i]
        y_test_i = Y_test[:, i]

        # Skip constant labels (all 0 or all 1)
        if y_train_i.sum() == 0 or y_train_i.sum() == len(y_train_i):
            f1 = 0.0 if y_test_i.sum() == 0 else 1.0
            results_per_bsp.append(
                {
                    "bsp_id": bsp_id,
                    "category": category,
                    "f1": round(f1, 4),
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
        f1 = f1_score(y_test_i, y_pred, zero_division=0.0)

        results_per_bsp.append(
            {
                "bsp_id": bsp_id,
                "category": category,
                "f1": round(float(f1), 4),
                "skipped": False,
                "train_pos_rate": round(float(y_train_i.mean()), 4),
            }
        )

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(
                f"  [{i+1}/{num_bsps}] elapsed={elapsed:.1f}s",
                file=sys.stderr,
            )

    total_time = time.time() - t0
    print(f"Done in {total_time:.1f}s", file=sys.stderr)

    # --- Aggregate by category ---
    cat_f1s = {}
    for r in results_per_bsp:
        cat = r["category"]
        if cat not in cat_f1s:
            cat_f1s[cat] = []
        cat_f1s[cat].append(r["f1"])

    per_category = {}
    for cat in categories:
        f1s = cat_f1s.get(cat, [])
        if f1s:
            arr = np.array(f1s)
            per_category[cat] = {
                "count": len(f1s),
                "mean_f1": round(float(arr.mean()), 4),
                "min_f1": round(float(arr.min()), 4),
                "max_f1": round(float(arr.max()), 4),
                "median_f1": round(float(np.median(arr)), 4),
            }

    all_f1s = np.array([r["f1"] for r in results_per_bsp])
    overall = {
        "coverage": round(float(all_f1s.mean()), 4),
        "coverage_above_50": round(float((all_f1s > 0.50).mean()), 4),
        "coverage_above_75": round(float((all_f1s > 0.75).mean()), 4),
        "num_bsps": num_bsps,
        "min_f1": round(float(all_f1s.min()), 4),
        "max_f1": round(float(all_f1s.max()), 4),
        "median_f1": round(float(np.median(all_f1s)), 4),
    }

    # --- Output ---
    output = {
        "overall": overall,
        "per_category": per_category,
        "per_bsp": results_per_bsp,
        "config": {
            "activations": str(act_path),
            "bsp_labels": str(bsp_path),
            "C": C,
            "test_frac": test_frac,
            "seed": seed,
            "max_iter": max_iter,
            "n_train": X_train.shape[0],
            "n_test": X_test.shape[0],
        },
    }

    if output_path == "auto" or output_path is None:
        output_path = act_path.parent / f"linear_probe_{act_path.stem}_results.json"
    else:
        output_path = Path(output_path)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # --- Print summary ---
    print("\n=== Linear Probe Baseline ===", file=sys.stderr)
    print(f"Overall coverage (mean F1): {overall['coverage']:.4f}", file=sys.stderr)
    print(
        f"BSPs above 0.50 F1: {overall['coverage_above_50']:.1%}",
        file=sys.stderr,
    )
    print(
        f"BSPs above 0.75 F1: {overall['coverage_above_75']:.1%}",
        file=sys.stderr,
    )
    print(f"\nPer-category breakdown:", file=sys.stderr)
    for cat, stats in per_category.items():
        print(
            f"  {cat:25s}  n={stats['count']:3d}  "
            f"mean_F1={stats['mean_f1']:.4f}  "
            f"[{stats['min_f1']:.4f}, {stats['max_f1']:.4f}]",
            file=sys.stderr,
        )
    print(f"\nResults saved to: {output_path}", file=sys.stderr)

    # Also print JSON to stdout for programmatic consumption
    json.dump(output["overall"], sys.stdout)
    print()


if __name__ == "__main__":
    main()
