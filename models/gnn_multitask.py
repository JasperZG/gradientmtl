"""
GNN-based Multi-task model for molecular property prediction.

Uses a graph neural network encoder with task-specific heads.
"""

import torch
import torch.nn as nn
from torch_geometric.data import Batch

from .gnn_encoder import create_gnn_encoder
from .heads import TaskHead


class GNNMultiTaskModel(nn.Module):
    """
    Multi-task model with GNN encoder and task-specific heads.

    Architecture:
    - GNN encoder (GCN or GAT) for learning molecular representations
    - Separate prediction heads for each task
    """

    def __init__(
        self,
        task_names: list[str],
        atom_feature_dim: int,
        encoder_type: str = 'gcn',
        encoder_hidden_dims: list[int] = [256, 256, 256],
        encoder_output_dim: int = 256,
        head_hidden_dim: int = 128,
        dropout: float = 0.2,
        **encoder_kwargs,
    ):
        """
        Args:
            task_names: List of task names
            atom_feature_dim: Dimension of atom features
            encoder_type: 'gcn' or 'gat'
            encoder_hidden_dims: Hidden dimensions for GNN layers
            encoder_output_dim: Output dimension of encoder
            head_hidden_dim: Hidden dimension for task heads
            dropout: Dropout rate
            **encoder_kwargs: Additional encoder arguments (e.g., heads for GAT)
        """
        super().__init__()

        self.task_names = task_names
        self.encoder_type = encoder_type

        # Create GNN encoder
        self.encoder = create_gnn_encoder(
            encoder_type=encoder_type,
            input_dim=atom_feature_dim,
            hidden_dims=encoder_hidden_dims,
            output_dim=encoder_output_dim,
            dropout=dropout,
            **encoder_kwargs,
        )

        # Create task-specific heads
        self.heads = nn.ModuleDict()
        for task_name in task_names:
            self.heads[task_name] = TaskHead(
                input_dim=encoder_output_dim,
                hidden_dim=head_hidden_dim,
                dropout=dropout,
            )

    def forward(self, batch: Batch) -> dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            batch: PyG Batch object

        Returns:
            Dict mapping task_name -> predictions [batch_size, 1]
        """
        # Encode molecular graphs
        shared_repr = self.encoder(batch)

        # Task-specific predictions
        outputs = {}
        for task_name, head in self.heads.items():
            outputs[task_name] = head(shared_repr)

        return outputs

    def get_encoder_parameters(self):
        """Get iterator over encoder parameters (for gradient logging)."""
        return self.encoder.parameters()

    def get_head_parameters(self, task_name: str):
        """Get iterator over specific task head parameters."""
        return self.heads[task_name].parameters()

    def summary(self) -> str:
        """Return a summary of the model architecture."""
        lines = [
            f"GNNMultiTaskModel ({self.encoder_type.upper()})",
            "=" * 50,
            f"Encoder: {self.encoder_type.upper()}",
        ]

        # Count parameters
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        lines.append(f"  Parameters: {encoder_params:,}")

        lines.append(f"\nTask Heads ({len(self.task_names)} tasks):")
        total_head_params = 0
        for task_name in self.task_names:
            head_params = sum(p.numel() for p in self.heads[task_name].parameters())
            total_head_params += head_params
            lines.append(f"  {task_name}: {head_params:,} params")

        total_params = encoder_params + total_head_params
        lines.append(f"\nTotal Parameters: {total_params:,}")

        return '\n'.join(lines)
