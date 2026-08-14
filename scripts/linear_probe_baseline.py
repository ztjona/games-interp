"""Linear probe baseline: logistic regression on raw activations for BSP classification.

Trains per-BSP L2-regularized logistic regression probes on raw activations and
reports per-BSP F1, MCC and F1-lift, plus per-category and overall coverage
metrics. Establishes the upper bound any SAE can achieve on these activations.

Output format mirrors ``sae_eval`` so LP vs SAE numbers are directly comparable:

    coverage_mcc          mean per-BSP MCC. HEADLINE metric. Zero for any
                          constant predictor, so it collapses the F1 ~= 0.667
                          offered_piece artefact. Not base-rate invariant --
                          see the two below before comparing populations.
    coverage_youden_j     mean per-BSP Youden's J = TPR + TNR - 1. Prevalence-
                          INVARIANT companion; MCC-vs-J divergence reads off
                          how much of a gap is base rate rather than quality.
    coverage_mcc_at_pref  mean per-BSP MCC restated at p_ref = 0.025. This is
                          the number to compare ACROSS bases or champions
                          whose base rates differ (e.g. hawk vs tiger).
    coverage              mean per-BSP F1. Retained for comparison with the
                          published literature ONLY -- never rank on it.

Per-category numbers use the same key names as
``lib/sae/eval.compute_per_category_coverage``, and ``per_family`` rolls them up
by the schema's ``concept_family`` stamp, so an LP report and an SAE registry
row are directly diffable field-by-field and family-by-family.

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
    --max-train=<n>        Cap the TRAINING split at n rows (0 = use all)
                           [default: 0]. The probe is an upper bound, not a
                           precision estimate, and with 512 features a 232k-row
                           train split is far past diminishing returns -- at
                           ~33 s/BSP the full panel is ~21 h of CPU. The test
                           split is never subsampled. Rare BSPs suffer first;
                           the script names any with <50 positives left.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.sae.eval import (  # noqa: E402
    P_REF_DEFAULT,
    aggregate_per_category_by_family,
    mcc_from_rates,
)


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

    # Optional train-set subsample. The probe is an UPPER BOUND on decodability,
    # not a precision estimate, and with d=512 features a 232k-row training set
    # is far past the point where more rows change the fit. lbfgs cost is roughly
    # linear in rows, so capping the train split is close to a pure time saving:
    # ~33 s/BSP at 232k rows means 373 BSPs x 6 champion-hook combos ~= 21 h CPU.
    #
    # The TEST split is deliberately left at full size -- the metric should still
    # be measured against the real distribution, and prediction is cheap.
    n_train_full = X_train.shape[0]
    max_train = int(args["--max-train"])
    subsampled = 0 < max_train < n_train_full
    if subsampled:
        rng = np.random.default_rng(seed)
        keep = rng.choice(n_train_full, size=max_train, replace=False)
        X_train, Y_train = X_train[keep], Y_train[keep]
        print(
            f"Subsampled train split: {n_train_full} -> {max_train} rows "
            f"(seed={seed}); test split left at {X_test.shape[0]}.",
            file=sys.stderr,
        )
        # Rare BSPs are where subsampling actually costs something: at base rate
        # 0.003 a 50k subsample holds ~150 positives. Name them rather than let
        # the reader assume every column is equally well estimated.
        pos = Y_train.sum(axis=0)
        thin = int((pos < 50).sum())
        if thin:
            print(
                f"WARNING: {thin}/{Y_train.shape[1]} BSPs have <50 positives in "
                f"the subsampled train split; their probes are noisy upper "
                f"bounds. Raise --max-train if those concepts matter.",
                file=sys.stderr,
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
                    "youden_j": 0.0,
                    "mcc_at_pref": 0.0,
                    "tpr": 0.0,
                    "tnr": 0.0,
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

        # Choose the decision threshold that maximises MCC, FIT ON TRAIN.
        #
        # `clf.predict` cuts at p = 0.5, which is badly calibrated for rare
        # classes: at base rate 0.025 it scored MCC 0.265 where the best
        # threshold reaches 0.493 (measured 2026-08-14). That makes the "upper
        # bound" beatable -- an SAE latent whose effective threshold happens to
        # sit better was scoring efficiency > 1 in 11 cells -- and it understates
        # LP worst on exactly the rare threat concepts this project is about.
        #
        # The threshold is selected on the TRAINING split and only then applied
        # to test, so this adds no leakage; it just stops the probe from being
        # crippled by an arbitrary cut.
        p_train = clf.predict_proba(X_train)[:, 1]
        best_t, best_m = 0.5, -2.0
        for t in np.unique(np.quantile(p_train, np.linspace(0.001, 0.999, 200))):
            tn_, fp_, fn_, tp_ = confusion_matrix(
                y_train_i, (p_train >= t).astype(int), labels=[0, 1]).ravel()
            den = math.sqrt(
                float(tp_ + fp_) * (tp_ + fn_) * (tn_ + fp_) * (tn_ + fn_))
            m = ((tp_ * tn_ - fp_ * fn_) / den) if den > 0 else 0.0
            if m > best_m:
                best_t, best_m = float(t), m
        y_pred = (clf.predict_proba(X_test)[:, 1] >= best_t).astype(int)

        cm = confusion_matrix(y_test_i, y_pred, labels=[0, 1])
        tn, fp, fn, tp = (int(x) for x in cm.ravel())
        f1, mcc = _scores_from_confusion(tn, fp, fn, tp)
        f1_lift = max(0.0, f1 - trivial_f1)

        # Prevalence-fair companions, identical in definition to the SAE side
        # (lib/sae/eval.py). Without them an LP-vs-SAE efficiency ratio can only
        # be formed on prevalence-DEPENDENT numbers, so it would confound
        # "hawk is harder to probe" with "hawk concepts are rarer".
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        tnr = tn / (tn + fp) if (tn + fp) else 0.0
        youden_j = tpr + tnr - 1.0
        mcc_at_pref = float(mcc_from_rates(
            torch.tensor(tpr), torch.tensor(tnr), P_REF_DEFAULT))

        results_per_bsp.append(
            {
                "bsp_id": bsp_id,
                "category": category,
                "f1": round(f1, 4),
                "mcc": round(mcc, 4),
                "youden_j": round(youden_j, 4),
                "mcc_at_pref": round(mcc_at_pref, 4),
                "tpr": round(tpr, 4),
                "tnr": round(tnr, 4),
                # Chosen on TRAIN; recorded so a report is auditable.
                "threshold": round(best_t, 4),
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
        js = np.array(
            [r["youden_j"] for r in results_per_bsp if r["category"] == cat]
        )
        stds = np.array(
            [r["mcc_at_pref"] for r in results_per_bsp if r["category"] == cat]
        )
        # Key names mirror lib/sae/eval.compute_per_category_coverage exactly, so
        # an LP report and an SAE registry row can be diffed field-by-field.
        per_category[cat] = {
            "count": int(f1s.size),
            "mean_f1": round(float(f1s.mean()), 4),
            "mean_mcc": round(float(mccs.mean()), 4),
            "median_mcc": round(float(np.median(mccs)), 4),
            "min_mcc": round(float(mccs.min()), 4),
            "max_mcc": round(float(mccs.max()), 4),
            "sd_mcc": round(float(mccs.std()), 4),
            "heterogeneous": bool((mccs.max() - mccs.min()) > 0.30),
            "mean_youden_j": round(float(js.mean()), 4),
            "mean_mcc_at_pref": round(float(stds.mean()), 4),
            "mean_f1_lift": round(float(lifts.mean()), 4),
            "mean_base_rate": round(float(brs.mean()), 4),
        }

    all_f1s = np.array([r["f1"] for r in results_per_bsp])
    all_mccs = np.array([r["mcc"] for r in results_per_bsp])
    all_lifts = np.array([r["f1_lift"] for r in results_per_bsp])
    all_js = np.array([r["youden_j"] for r in results_per_bsp])
    all_stds = np.array([r["mcc_at_pref"] for r in results_per_bsp])
    all_brs = np.array([r["base_rate"] for r in results_per_bsp])
    overall = {
        # MCC is the headline; J is the prevalence-INVARIANT companion and
        # mcc_at_pref the prevalence-STANDARDISED one. The F1 family is kept
        # only for comparison with the published literature -- do not rank on
        # it. Same contract as the SAE registry (see CLAUDE.md).
        "coverage_mcc": round(float(all_mccs.mean()), 4),
        "coverage_mcc_above_25": round(float((all_mccs > 0.25).mean()), 4),
        "coverage_mcc_above_50": round(float((all_mccs > 0.50).mean()), 4),
        "median_mcc": round(float(np.median(all_mccs)), 4),
        "min_mcc": round(float(all_mccs.min()), 4),
        "max_mcc": round(float(all_mccs.max()), 4),
        "coverage_youden_j": round(float(all_js.mean()), 4),
        "median_youden_j": round(float(np.median(all_js)), 4),
        "coverage_mcc_at_pref": round(float(all_stds.mean()), 4),
        "median_mcc_at_pref": round(float(np.median(all_stds)), 4),
        "p_ref": P_REF_DEFAULT,
        "mean_base_rate": round(float(all_brs.mean()), 4),
        "median_base_rate": round(float(np.median(all_brs)), 4),
        "min_base_rate": round(float(all_brs.min()), 4),
        "max_base_rate": round(float(all_brs.max()), 4),
        "frac_bsps_very_rare": round(float((all_brs < 0.01).mean()), 4),
        "num_bsps": num_bsps,
        # --- demoted: literature comparison only ---
        "coverage": round(float(all_f1s.mean()), 4),
        "coverage_f1_lift": round(float(all_lifts.mean()), 4),
    }

    # Concept-family rollup, read off the schema's stamp -- the LP side must
    # group exactly the way the SAE side does, or a hawk-vs-tiger LP/SAE ratio
    # is comparing different partitions of the concept menu.
    per_family = aggregate_per_category_by_family(per_category, schema)
    if not per_family:
        print(
            "WARNING: schema carries no concept_family stamp; no per-family "
            "rollup. Run: python scripts/stamp_concept_families.py",
            file=sys.stderr,
        )

    # --- Output ---
    output = {
        "overall": overall,
        "per_category": per_category,
        "per_family": per_family,
        "per_bsp": results_per_bsp,
        "config": {
            "activations": str(act_path),
            "bsp_labels": str(bsp_path),
            "bsp_set": _infer_bsp_set_name(bsp_path, schema),
            "C": C,
            "test_frac": test_frac,
            # Recorded so a subsampled report is never mistaken for a full one.
            "max_train": max_train,
            "train_subsampled": subsampled,
            "n_train_full": n_train_full,
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
        f"coverage_mcc        : {overall['coverage_mcc']:.4f}  "
        f"(above25={overall['coverage_mcc_above_25']:.1%}, "
        f"above50={overall['coverage_mcc_above_50']:.1%})   <- headline",
        file=sys.stderr,
    )
    print(
        f"coverage_youden_j   : {overall['coverage_youden_j']:.4f}  "
        f"(prevalence-invariant)",
        file=sys.stderr,
    )
    print(
        f"coverage_mcc_at_pref: {overall['coverage_mcc_at_pref']:.4f}  "
        f"(p_ref={overall['p_ref']}; compare ACROSS bases with this)",
        file=sys.stderr,
    )
    print(
        f"mean_base_rate      : {overall['mean_base_rate']:.4f}  "
        f"({overall['frac_bsps_very_rare']:.1%} of BSPs below p=0.01)",
        file=sys.stderr,
    )
    print(
        f"coverage (F1)       : {overall['coverage']:.4f}  "
        f"(literature comparison only -- do not rank on it)",
        file=sys.stderr,
    )
    print("\nPer-category (MCC / J / MCC@pref / base):", file=sys.stderr)
    for cat, stats in per_category.items():
        print(
            f"  {cat:25s}  n={stats['count']:3d}  "
            f"MCC={stats['mean_mcc']:.4f}  "
            f"J={stats['mean_youden_j']:.4f}  "
            f"MCC@pref={stats['mean_mcc_at_pref']:.4f}  "
            f"base={stats['mean_base_rate']:.4f}",
            file=sys.stderr,
        )
    if per_family:
        print("\nPer-concept-family (count-weighted over BSPs):",
              file=sys.stderr)
        for family, stats in sorted(per_family.items()):
            print(
                f"  {family:25s}  n={stats['count']:3d}  "
                f"MCC={stats.get('mean_mcc', float('nan')):.4f}  "
                f"MCC@pref={stats.get('mean_mcc_at_pref', float('nan')):.4f}",
                file=sys.stderr,
            )
    print(f"\nResults saved to: {output_path}", file=sys.stderr)

    # Also print JSON to stdout for programmatic consumption
    json.dump(output["overall"], sys.stdout)
    print()


if __name__ == "__main__":
    main()
