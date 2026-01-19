"""Shared encoder for multi-task learning."""

import torch
import torch.nn as nn


class SharedEncoder(nn.Module):
    """
    Shared MLP encoder for multi-task molecular property prediction.

    Architecture: Input -> [Linear -> BatchNorm -> ReLU -> Dropout] x N -> Output

    The encoder learns a shared representation that is used by all task-specific
    heads. Gradient analysis is performed on this shared encoder to understand
    task relationships.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dims: list[int] = [512, 256],
        dropout: float = 0.2,
    ):
        """
        Args:
            input_dim: Input dimension (fingerprint size)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = hidden_dims[-1]

        # Build encoder layers
        layers = []
        prev_dim = input_dim

        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = dim

        self.encoder = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Encoded representation of shape (batch_size, output_dim)
        """
        return self.encoder(x)

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
