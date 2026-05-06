"""Public API for lib.sae — Sparse Autoencoder training library."""

from .architectures import (
    ARCHITECTURES,
    BaseSAE,
    BatchTopKSAE,
    GatedSAE,
    JumpReLUSAE,
    PAnnealingSAE,
    TopKSAE,
    VanillaSAE,
)
from .eval import (
    FeatureBSPMatching,
    compute_board_reconstruction,
    compute_coverage,
    compute_feature_sharing,
    compute_per_category_coverage,
    evaluate_sae,
    match_features_to_bsps,
)
from .hooks import ActivationStore, list_hooks, load_game_model
from .registry import register_training_run
from .train import (
    compute_metrics,
    iter_batches,
    load_activation_data,
    load_checkpoint,
    save_checkpoint,
    train_sae,
)

__all__ = [
    # Architectures
    "BaseSAE",
    "VanillaSAE",
    "TopKSAE",
    "BatchTopKSAE",
    "GatedSAE",
    "JumpReLUSAE",
    "PAnnealingSAE",
    "ARCHITECTURES",
    # Evaluation
    "FeatureBSPMatching",
    "match_features_to_bsps",
    "compute_coverage",
    "compute_feature_sharing",
    "compute_per_category_coverage",
    "compute_board_reconstruction",
    "evaluate_sae",
    # Hooks
    "ActivationStore",
    "load_game_model",
    "list_hooks",
    # Training
    "train_sae",
    "compute_metrics",
    "iter_batches",
    "load_activation_data",
    "save_checkpoint",
    "load_checkpoint",
    # Registry
    "register_training_run",
]
