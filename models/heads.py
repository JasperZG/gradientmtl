"""Task-specific prediction heads."""

import torch
import torch.nn as nn


class TaskHead(nn.Module):
    """
    Task-specific prediction head.

    Architecture: Input -> Linear -> BatchNorm -> ReLU -> Dropout -> Linear -> Output

    Each task has its own head that maps the shared encoder output to
    a task-specific prediction. The output is a single value (logit for
    classification, value for regression).
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        """
        Args:
            input_dim: Input dimension (encoder output dim)
            hidden_dim: Hidden layer dimension
            dropout: Dropout probability
        """
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

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
        Forward pass through head.

        Args:
            x: Encoder output of shape (batch_size, input_dim)

        Returns:
            Predictions of shape (batch_size,)
        """
        return self.head(x).squeeze(-1)

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
