"""SAE evaluation metrics: coverage and board reconstruction.

Coverage measures how well SAE features align with known Board State Properties
(BSPs).  Board reconstruction measures whether high-precision features can
recover the full board state.

All functions operate on pre-computed tensors and are game-agnostic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from tqdm import tqdm

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature–BSP matching
# ---------------------------------------------------------------------------


@dataclass
class FeatureBSPMatching:
    """Result of matching SAE features to BSPs.

    Attributes:
        precision: (d_dict, num_bsps) float tensor
        recall:    (d_dict, num_bsps) float tensor
        f1:        (d_dict, num_bsps) float tensor
        best_f1_per_bsp:      (num_bsps,) best F1 for each BSP
        best_feature_per_bsp: (num_bsps,) index of best feature for each BSP
    """

    precision: torch.Tensor
    recall: torch.Tensor
    f1: torch.Tensor
    best_f1_per_bsp: torch.Tensor
    best_feature_per_bsp: torch.Tensor


def match_features_to_bsps(
    h: torch.Tensor,
    bsp_labels: torch.Tensor,
) -> FeatureBSPMatching:
    """Compute precision, recall, F1 between every SAE feature and every BSP.

    Args:
        h:          (N, d_dict) SAE hidden activations (non-negative).
        bsp_labels: (N, num_bsps) binary BSP labels (0.0 or 1.0).

    Returns:
        FeatureBSPMatching with all pairwise metrics and best-feature assignments.
    """
    if h.shape[0] != bsp_labels.shape[0]:
        raise ValueError(
            f"Sample count mismatch: h has {h.shape[0]}, "
            f"bsp_labels has {bsp_labels.shape[0]}"
        )

    with torch.no_grad():
        # Binarize feature activations
        fires = (h > 0).float()  # (N, d_dict)
        labels = bsp_labels.float()  # (N, num_bsps)

        # Pairwise counts via matrix multiplication
        # TP[i,j] = sum over samples where feature i fires AND BSP j is true
        tp = fires.T @ labels  # (d_dict, num_bsps)
        fp = fires.T @ (1.0 - labels)  # (d_dict, num_bsps)
        fn = (1.0 - fires).T @ labels  # (d_dict, num_bsps)

        eps = 1e-8
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2.0 * precision * recall / (precision + recall + eps)

        best_f1_per_bsp, best_feature_per_bsp = f1.max(dim=0)

    return FeatureBSPMatching(
        precision=precision,
        recall=recall,
        f1=f1,
        best_f1_per_bsp=best_f1_per_bsp,
        best_feature_per_bsp=best_feature_per_bsp,
    )


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def compute_coverage(matching: FeatureBSPMatching) -> dict[str, float]:
    """Compute coverage: mean of best-F1 across BSPs.

    Also reports fraction of BSPs above standard thresholds.

    Args:
        matching: Output of match_features_to_bsps().

    Returns:
        Dict with keys: coverage, coverage_above_50, coverage_above_75,
        num_bsps, min_f1, max_f1, median_f1.
    """
    best_f1 = matching.best_f1_per_bsp
    num_bsps = best_f1.shape[0]

    return {
        "coverage": round(best_f1.mean().item(), 4),
        "coverage_above_50": round((best_f1 > 0.50).float().mean().item(), 4),
        "coverage_above_75": round((best_f1 > 0.75).float().mean().item(), 4),
        "num_bsps": num_bsps,
        "min_f1": round(best_f1.min().item(), 4),
        "max_f1": round(best_f1.max().item(), 4),
        "median_f1": round(best_f1.median().item(), 4),
    }


# ---------------------------------------------------------------------------
# Board Reconstruction
# ---------------------------------------------------------------------------


def compute_board_reconstruction(
    matching: FeatureBSPMatching,
    h: torch.Tensor,
    bsp_labels: torch.Tensor,
    precision_threshold: float = 0.9,
) -> dict[str, Any]:
    """Compute board reconstruction from high-precision SAE features.

    For each BSP, select the feature with highest precision above the threshold.
    Then compute classification accuracy using that feature's firing pattern as
    the BSP prediction.

    Args:
        matching:             Output of match_features_to_bsps().
        h:                    (N, d_dict) SAE hidden activations.
        bsp_labels:           (N, num_bsps) binary BSP labels.
        precision_threshold:  Minimum precision to qualify a feature (default 0.9).

    Returns:
        Dict with keys: board_reconstruction, num_reconstructable_bsps,
        fraction_reconstructable, mean_accuracy, per_bsp_accuracy (list).
    """
    num_bsps = bsp_labels.shape[1]

    with torch.no_grad():
        fires = (h > 0).float()  # (N, d_dict)
        labels = bsp_labels.float()  # (N, num_bsps)

        # For each BSP, find the feature with highest precision above threshold
        precision = matching.precision  # (d_dict, num_bsps)

        # Mask features below threshold
        above_thresh = precision >= precision_threshold  # (d_dict, num_bsps)

        # For each BSP: best precision feature among those above threshold
        # Set below-threshold precisions to -1 so they can't win argmax
        masked_precision = precision.clone()
        masked_precision[~above_thresh] = -1.0
        best_prec_per_bsp, best_prec_feature = masked_precision.max(dim=0)

        # Which BSPs have at least one qualifying feature?
        reconstructable = best_prec_per_bsp >= precision_threshold  # (num_bsps,)
        num_reconstructable = int(reconstructable.sum().item())

        if num_reconstructable == 0:
            return {
                "board_reconstruction": 0.0,
                "num_reconstructable_bsps": 0,
                "fraction_reconstructable": 0.0,
                "mean_accuracy": 0.0,
                "per_bsp_accuracy": [],
            }

        # Compute accuracy for each reconstructable BSP
        per_bsp_accuracy = []
        accuracy_sum = 0.0

        for j in range(num_bsps):
            if not reconstructable[j]:
                per_bsp_accuracy.append(None)
                continue

            feat_idx = int(best_prec_feature[j].item())
            pred = fires[:, feat_idx]  # (N,)
            true = labels[:, j]  # (N,)
            correct = (pred == true).float().mean().item()
            per_bsp_accuracy.append(round(correct, 4))
            accuracy_sum += correct

        mean_acc = accuracy_sum / num_reconstructable

    return {
        "board_reconstruction": round(mean_acc, 4),
        "num_reconstructable_bsps": num_reconstructable,
        "fraction_reconstructable": round(num_reconstructable / num_bsps, 4),
        "mean_accuracy": round(mean_acc, 4),
        "per_bsp_accuracy": per_bsp_accuracy,
    }


# ---------------------------------------------------------------------------
# Combined evaluation
# ---------------------------------------------------------------------------


def evaluate_sae(
    sae,
    activations: torch.Tensor,
    bsp_labels: torch.Tensor,
    precision_threshold: float = 0.9,
    batch_size: int = 4096,
) -> dict:
    """Run full Layer 1 evaluation: coverage + board reconstruction.

    Args:
        sae:                  A BaseSAE instance (should be in eval mode).
        activations:          (N, d_input) raw activation tensor.
        bsp_labels:           (N, num_bsps) binary BSP labels.
        precision_threshold:  Threshold for board reconstruction (default 0.9).
        batch_size:           Process activations in chunks to control memory.

    Returns:
        Dict combining coverage and reconstruction metrics plus structural
        metrics (FVU, L0, dead_features_pct, mse).
    """
    from .train import compute_metrics

    N = activations.shape[0]
    device = next(sae.parameters()).device

    # Encode all activations to get hidden representations
    h_parts = []
    n_batches = (N + batch_size - 1) // batch_size
    with torch.no_grad():
        for i in tqdm(
            range(0, N, batch_size), total=n_batches, desc="Encoding", unit="batch"
        ):
            batch = activations[i : i + batch_size].to(device)
            result = sae(batch)
            h_parts.append(result["h"].cpu())

    h = torch.cat(h_parts, dim=0)  # (N, d_dict) on CPU
    bsp_labels_cpu = bsp_labels.cpu()

    # Structural metrics (on a sample, like training)
    eval_sample = activations[: min(N, batch_size)].to(device)
    structural = compute_metrics(sae, eval_sample)

    # Feature-BSP matching
    matching = match_features_to_bsps(h, bsp_labels_cpu)

    # Coverage
    coverage = compute_coverage(matching)

    # Board reconstruction
    reconstruction = compute_board_reconstruction(
        matching, h, bsp_labels_cpu, precision_threshold=precision_threshold
    )

    return {
        **structural,
        **coverage,
        **reconstruction,
    }
