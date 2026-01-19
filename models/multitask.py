"""Combined multi-task model."""

from typing import Iterator
import torch
import torch.nn as nn

from .encoder import SharedEncoder
from .heads import TaskHead


class MultiTaskModel(nn.Module):
    """
    Multi-task model for molecular property prediction.

    Consists of a shared encoder and task-specific prediction heads.
    The shared encoder learns representations that are useful across
    all tasks, while each head specializes for its specific task.
    """

    def __init__(
        self,
        task_names: list[str],
        input_dim: int = 2048,
        encoder_hidden_dims: list[int] = [512, 256],
        head_hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        """
        Args:
            task_names: List of task names
            input_dim: Input dimension (fingerprint size)
            encoder_hidden_dims: Hidden dimensions for shared encoder
            head_hidden_dim: Hidden dimension for task heads
            dropout: Dropout probability
        """
        super().__init__()

        self.task_names = task_names

        # Shared encoder
        self.encoder = SharedEncoder(
            input_dim=input_dim,
            hidden_dims=encoder_hidden_dims,
            dropout=dropout,
        )

        # Task-specific heads (ModuleDict for proper parameter registration)
        self.heads = nn.ModuleDict({
            task: TaskHead(
                input_dim=self.encoder.output_dim,
                hidden_dim=head_hidden_dim,
                dropout=dropout,
            )
            for task in task_names
        })

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass through model.

        Args:
            x: Input fingerprints of shape (batch_size, input_dim)

        Returns:
            Dict mapping task_name -> predictions (batch_size,)
            Predictions are logits for classification tasks (apply sigmoid for probabilities)
        """
        # Shared encoding
        encoding = self.encoder(x)

        # Task-specific predictions
        return {task: head(encoding) for task, head in self.heads.items()}

    def get_encoder_parameters(self) -> Iterator[nn.Parameter]:
        """
        Get parameters of the shared encoder.

        Used for gradient analysis - we analyze gradients w.r.t. shared parameters
        to understand task relationships.
        """
        return self.encoder.parameters()

    def get_head_parameters(self, task: str) -> Iterator[nn.Parameter]:
        """Get parameters for a specific task head."""
        return self.heads[task].parameters()

    def get_all_parameters(self) -> Iterator[nn.Parameter]:
        """Get all model parameters."""
        return self.parameters()

    def freeze_encoder(self):
        """Freeze encoder parameters (for transfer learning)."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def get_num_parameters(self) -> dict[str, int]:
        """Get number of parameters for each component."""
        encoder_params = self.encoder.get_num_parameters()
        head_params = {task: head.get_num_parameters()
                      for task, head in self.heads.items()}
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'encoder': encoder_params,
            'heads': head_params,
            'total': total_params,
        }

    def summary(self) -> str:
        """Get model summary string."""
        params = self.get_num_parameters()
        lines = [
            "MultiTaskModel Summary",
            "=" * 50,
            f"Tasks: {', '.join(self.task_names)}",
            f"Encoder: {params['encoder']:,} parameters",
            "Heads:",
        ]
        for task, n_params in params['heads'].items():
            lines.append(f"  {task}: {n_params:,} parameters")
        lines.extend([
            "-" * 50,
            f"Total: {params['total']:,} parameters",
        ])
        return '\n'.join(lines)
