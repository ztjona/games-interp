# -*- coding: utf-8 -*-
"""Abstract base class for neural network models in the Quarto game.
This class defines the interface for neural network models used in the Quarto game.
It defines the interfaces for model initialization, forward pass and prediction, and it implements model export and import.
"""

"""
Python 3
26 / 05 / 2025
@author: z_tjona

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth
"""
from abc import ABC, abstractmethod

import torch

from datetime import datetime
from os import path, makedirs
import torch.nn.functional as F


class NN_abstract(ABC, torch.nn.Module):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def __init__(self, *args, **kwargs):
        super().__init__()
        # Set device - use CUDA if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pass

    @abstractmethod
    def forward(
        self, x_board: torch.Tensor, x_piece: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    def predict(
        self,
        x_board: torch.Tensor,
        x_piece: torch.Tensor,
        TEMPERATURE: float = 1.0,
        DETERMINISTIC: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts the preferred order of all board positions and pieces, with optional ``TEMPERATURE`` for randomness.

        Args:
            ``x_board``: Input tensor of shape (batch_size, 16, 4, 4).
            ``x_piece``: Input tensor of shape (batch_size, 16).
            ``TEMPERATURE``: Sampling temperature (>0). Lower values make predictions more deterministic.
                Only used when ``DETERMINISTIC`` is False.
            ``DETERMINISTIC``: If True, returns sorted indices by Q-value (argmax).
                If False, samples from softmax distribution.

        Returns:
            * ``board_indices``: Ordered board position indices (batch_size, 16).
                Index 0 is the best position, index 15 is the worst.
            * ``piece_indices``: Ordered piece indices (batch_size, 16).
                Index 0 is the best piece, index 15 is the worst.
        """
        assert x_board.shape[1:] == (
            16,
            4,
            4,
        ), "Input tensor must have shape (batch_size, 16, 4, 4)"
        assert x_piece.shape[1] == 16, "Input tensor must have shape (batch_size, 16)"
        assert (
            x_board.shape[0] == x_piece.shape[0]
        ), "Input tensors must have the same batch size"

        assert x_board.shape[0] == 1, "Batch size of 1 is required for prediction"
        self.eval()
        with torch.no_grad():
            # Move inputs to the same device as the model
            x_board = x_board.to(self.device)
            x_piece = x_piece.to(self.device)
            qav_board, qav_piece = self.forward(x_board, x_piece)

            # Use tanh outputs directly for deterministic prediction
            if DETERMINISTIC:
                # Independent sorting for board positions and pieces
                board_indices = torch.argsort(qav_board, descending=True, dim=1)
                piece_indices = torch.argsort(qav_piece, descending=True, dim=1)
                return board_indices, piece_indices
            else:
                # For stochastic prediction, use softmax over tanh outputs and sample
                board_probs = F.softmax(qav_board / TEMPERATURE, dim=1)
                piece_probs = F.softmax(qav_piece / TEMPERATURE, dim=1)
                # Independent sorting for board positions and pieces
                board_indices = torch.multinomial(
                    board_probs,
                    board_probs.shape[1],  # all possible combinations
                    replacement=False,
                )
                piece_indices = torch.multinomial(
                    piece_probs,
                    piece_probs.shape[1],  # all possible combinations
                    replacement=False,
                )
                return board_indices, piece_indices

    # ####################################################################
    @classmethod
    def from_file(cls, weights_path: str):
        """
        Load the model from a file.

        Args:
            weights_path: Path to the saved model weights file (.pt file).

        Returns:
            NN_abstract: Model instance with loaded weights.
        """
        model = cls()

        # specifically load only weights, map to appropriate device
        model.load_state_dict(
            torch.load(weights_path, weights_only=True, map_location=model.device)
        )
        model.to(model.device)  # Ensure model is on the correct device

        return model

    def export_model(
        self,
        checkpoint_suffix: str,
        checkpoint_folder: str = "$__filedir__$/weights/",
    ) -> str:
        """
        Export the model to a file with the datetime and suffix in the filename.

        ## Args:
            checkpoint_suffix: Suffix for the checkpoint file name. Usually the epoch number.
            checkpoint_folder: Folder to save the model weights.
            Defaults to "$__filedir__$/weights/", which will be replaced with the directory of this file.
        ## Returns:
            The full path to the saved model file.
        """
        checkpoint_name = (
            f"{datetime.now().strftime('%Y%m%d_%H%M')}-{checkpoint_suffix}.pt"
        )

        if checkpoint_folder.startswith("$__filedir__$/"):
            # Replace the placeholder with the actual directory of this file
            base_dir = path.dirname(path.abspath(__file__))
            checkpoint_folder = checkpoint_folder.replace(
                "$__filedir__$/", base_dir + "/"
            )

        file_path = path.join(checkpoint_folder, checkpoint_name)

        makedirs(path.dirname(file_path), exist_ok=True)
        torch.save(self.state_dict(), file_path)

        return file_path
