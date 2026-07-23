# -*- coding: utf-8 -*-

"""Sa-series architecture variants on top of QuartoCNNAutoregUnified.

All three variants preserve the OA contract:
- 32-d phase-stable aux input (offered_one_hot ⊕ available_mask)
- two output heads ``fc2_place`` / ``fc2_select`` mapping to 16-d Q values
- tanh output activation (bounded)
- phase-agnostic trunk (phase only routes which head to read)

Only the trunk capacity / layout differs:

- ``QuartoCNNAutoregUnifiedS1``: deeper conv stack (3 conv layers, 32-64-128 channels)
- ``QuartoCNNAutoregUnifiedS2``: wider + deeper fc head (fc1 → fc1b, 512 → 256)
- ``QuartoCNNAutoregUnifiedS4``: uniform 512-wide trunk (n_neurons=512 in fc1)

See ``docs/diary/series-R-S.md``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.quarto.CNN_autoreg import _QuartoCNNAutoregUnifiedBase
from models.NN_abstract import NN_abstract

# ---------------------------------------------------------------------------
# S1: deeper conv (32-64-128) with the same aux contract and fc head shape
# ---------------------------------------------------------------------------


class QuartoCNNAutoregUnifiedS1(_QuartoCNNAutoregUnifiedBase):
    """Deeper-conv variant: conv1(32) → conv2(64) → conv3(128)."""

    def __init__(self):
        # Skip the base __init__: we redefine the entire trunk and heads.
        NN_abstract.__init__(self)

        fc_aux_size = 32
        self.fc_in_aux = nn.Linear(self.AUX_DIM, fc_aux_size)

        trunk_in_channels = 16 + (fc_aux_size // 16)
        k1, k2, k3 = 32, 64, 128
        n_neurons = 128

        self.conv1 = nn.Conv2d(trunk_in_channels, k1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(k1, k2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(k2, k3, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(k3 * 4 * 4, n_neurons)
        self.fc2_place = nn.Linear(n_neurons, 16)
        self.fc2_select = nn.Linear(n_neurons, 16)
        self.dropout = nn.Dropout(0.5)

    @property
    def name(self) -> str:
        return "QuartoCNN_autoreg_unified_S1_deepConv"

    def _apply_output_activation(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.tanh(logits)

    def _shared_trunk(self, x_board, x_aux):
        import numpy as np  # local import; matches base file's pattern

        if isinstance(x_board, np.ndarray):
            x_board = torch.from_numpy(x_board).float()
        if isinstance(x_aux, np.ndarray):
            x_aux = torch.from_numpy(x_aux).float()

        x_board = x_board.to(self.device)
        x_aux = x_aux.to(self.device)

        if x_aux.shape[-1] != self.AUX_DIM:
            raise ValueError(
                f"Unified-aux model expects aux of size {self.AUX_DIM}, "
                f"got {tuple(x_aux.shape)}."
            )

        aux_feat = F.relu(self.fc_in_aux(x_aux))
        aux_channels = aux_feat.shape[-1] // 16
        aux_map = aux_feat.view(-1, aux_channels, 4, 4)

        x = torch.cat([x_board, aux_map], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return x


# ---------------------------------------------------------------------------
# S2: wide + deep fc head (512 → 256), same conv stack as OA base
# ---------------------------------------------------------------------------


class QuartoCNNAutoregUnifiedS2(_QuartoCNNAutoregUnifiedBase):
    """Wider+deeper fc head: fc1(→512) → fc1b(→256) before heads."""

    def __init__(self):
        NN_abstract.__init__(self)

        fc_aux_size = 32
        self.fc_in_aux = nn.Linear(self.AUX_DIM, fc_aux_size)

        trunk_in_channels = 16 + (fc_aux_size // 16)
        k1, k2 = 16, 32
        hidden1, hidden2 = 512, 256

        self.conv1 = nn.Conv2d(trunk_in_channels, k1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(k1, k2, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(k2 * 4 * 4, hidden1)
        self.fc1b = nn.Linear(hidden1, hidden2)
        self.fc2_place = nn.Linear(hidden2, 16)
        self.fc2_select = nn.Linear(hidden2, 16)
        self.dropout = nn.Dropout(0.5)

    @property
    def name(self) -> str:
        return "QuartoCNN_autoreg_unified_S2_wideFC"

    def _apply_output_activation(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.tanh(logits)

    def _shared_trunk(self, x_board, x_aux):
        import numpy as np

        if isinstance(x_board, np.ndarray):
            x_board = torch.from_numpy(x_board).float()
        if isinstance(x_aux, np.ndarray):
            x_aux = torch.from_numpy(x_aux).float()

        x_board = x_board.to(self.device)
        x_aux = x_aux.to(self.device)

        if x_aux.shape[-1] != self.AUX_DIM:
            raise ValueError(
                f"Unified-aux model expects aux of size {self.AUX_DIM}, "
                f"got {tuple(x_aux.shape)}."
            )

        aux_feat = F.relu(self.fc_in_aux(x_aux))
        aux_channels = aux_feat.shape[-1] // 16
        aux_map = aux_feat.view(-1, aux_channels, 4, 4)

        x = torch.cat([x_board, aux_map], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc1b(x))
        return x


# ---------------------------------------------------------------------------
# S4: uniform 512-wide trunk
# ---------------------------------------------------------------------------


class QuartoCNNAutoregUnifiedS4(_QuartoCNNAutoregUnifiedBase):
    """Uniform 512-neuron fc1; conv widths unchanged from OA base."""

    def __init__(self):
        NN_abstract.__init__(self)

        fc_aux_size = 32
        self.fc_in_aux = nn.Linear(self.AUX_DIM, fc_aux_size)

        trunk_in_channels = 16 + (fc_aux_size // 16)
        k1, k2 = 16, 32
        n_neurons = 512

        self.conv1 = nn.Conv2d(trunk_in_channels, k1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(k1, k2, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(k2 * 4 * 4, n_neurons)
        self.fc2_place = nn.Linear(n_neurons, 16)
        self.fc2_select = nn.Linear(n_neurons, 16)
        self.dropout = nn.Dropout(0.5)

    @property
    def name(self) -> str:
        return "QuartoCNN_autoreg_unified_S4_uniform512"

    def _apply_output_activation(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.tanh(logits)


# ---------------------------------------------------------------------------
# S4Hot: S4 trunk + auxiliary depth-1 hot-piece head (select-safety shaping)
# ---------------------------------------------------------------------------


class QuartoCNNAutoregUnifiedS4Hot(QuartoCNNAutoregUnifiedS4):
    """S4 trunk + an auxiliary ``fc_hot`` head predicting the depth-1
    hot-piece mask (1 = giving this piece loses to an immediate completion).

    Purpose: a dense BCE signal that forces the *shared trunk* to encode
    piece-safety, which the unchanged ``fc2_select`` head then reads. The
    ``forward`` pass is unchanged and still returns ``(q_place, q_select)``,
    so every existing bot / diagnostic / hook works as-is; the aux head is a
    training-time scaffold, exposed via :meth:`hot_logits` (its own trunk
    forward) and NOT consulted at inference.

    The trunk (``conv1``, ``conv2``, ``fc1``=512, ``fc2_place``,
    ``fc2_select``) is byte-identical to :class:`QuartoCNNAutoregUnifiedS4`,
    so all SAE hook points (``s4.conv1``, ``s4.conv2``, ``s4.fc1`` ...) and
    activation dimensions are unchanged. The only extra parameters are
    ``fc_hot.{weight,bias}``, which the checkpoint carries but the pipeline
    never reads. See champYb onboarding diary 2026-06-18.
    """

    def __init__(self):
        super().__init__()
        self.fc_hot = nn.Linear(self.fc1.out_features, 16)

    @property
    def name(self) -> str:
        return "QuartoCNN_autoreg_unified_S4_hot"

    def hot_logits(self, x_board, x_aux) -> torch.Tensor:
        """Raw logits (B, 16) for the depth-1 hot-piece mask (BCE target side)."""
        x = self._shared_trunk(x_board, x_aux)
        return self.fc_hot(x)
