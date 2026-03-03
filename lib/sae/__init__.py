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
from .hooks import ActivationStore, list_hooks, load_game_model
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
]
