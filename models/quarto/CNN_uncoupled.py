# -*- coding: utf-8 -*-

"""
CNN_toy - Is the basic CNN model for the Quarto board game.
"""

"""
Python 3
15 / 05 / 2025
@author: z_tjona

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth
"""
from models.NN_abstract import NN_abstract

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ----------------------------- logging --------------------------


class QuartoCNN(NN_abstract):
    """
    QuartoCNN is a Convolutional Neural Network (CNN) model for the Quarto board game that predicts the action value function.
    It has a sequencial output architecture:
        * The first output predicts the board position to place a piece,
        * The second output uses the information of first output to predict the piece to place.
    # Input:
    * batchx16x4x4 input tensors representing different positions of the game board.
    * batchx16 dims for each piece

    # Output:
    * batch-16 [-1,1] tensor representing the action value of the board position
    * batch-by-16 [-1,1] tensor representing the action value of the piece
    """

    @property
    def name(self) -> str:
        return "QuartoCNN_uncoupled"

    def __init__(self):
        super().__init__()
        # Input shape: (batch_size, 16, 4, 4)
        # (batch_size, dims, height, width)
        fc_inpiece_size = 16  # must be multiple of 16

        assert fc_inpiece_size % 16 == 0, "fc_inpiece_size must be a multiple of 16"
        self.fc_in_piece = nn.Linear(
            16, fc_inpiece_size
        )  # Input layer for piece features

        k1_size = 16
        self.conv1 = nn.Conv2d(
            16 + fc_inpiece_size // 16, k1_size, kernel_size=3, padding=1
        )
        k2_size = 32
        self.conv2 = nn.Conv2d(k1_size, k2_size, kernel_size=3, padding=1)
        n_neurons = 128
        self.fc1 = nn.Linear(k2_size * 4 * 4, n_neurons)

        # Predicts piece selection (now independent of board prediction)
        self.fc2_board = nn.Linear(n_neurons, 4 * 4)

        # piece: predicts piece to give
        self.fc2_piece = nn.Linear(n_neurons, 4 * 4)
        self.dropout = nn.Dropout(0.5)  # 0.3 before

    def forward(
        self, x_board: torch.Tensor | np.ndarray, x_piece: torch.Tensor | np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the model.
        Args:
            ``x_board``: Input tensor of the board with placed pieces (batch_size, 16, 4, 4).
            ``x_piece``: Input tensor of selected piece to place (batch_size, 16).
        Returns:
            qav_board: Onehot tensor [-1, 1] of the action value for the board position to place piece (batch_size, 16).
            qav_piece: Onehot tensor [-1, 1] of the action value for the selected piece (batch_size, 16).
        """
        piece_feat = F.relu(self.fc_in_piece(x_piece))
        piece_map = piece_feat.view(-1, 1, 4, 4)
        x = torch.cat([x_board, piece_map], dim=1)  # type: ignore
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Bx128

        # output 1: board position (batch, 16)
        logits_board = self.fc2_board(x)
        qav_board = F.tanh(logits_board)

        # output 2: selected piece (batch, 16)
        # x_qav = torch.cat([x, qav_board], dim=1)
        # logits_piece = self.fc2_piece(x_qav)
        logits_piece = self.fc2_piece(x)
        qav_piece = F.tanh(logits_piece)
        return qav_board, qav_piece
