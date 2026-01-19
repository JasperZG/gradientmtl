"""
Graph Neural Network encoders for molecular property prediction.

Supports GCN (Graph Convolutional Network) and GAT (Graph Attention Network).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_add_pool
from torch_geometric.data import Batch


class GCNEncoder(nn.Module):
    """
    Graph Convolutional Network encoder.

    Architecture:
    - Multiple GCN layers with ReLU and dropout
    - Global mean pooling for graph-level representation
    - Final MLP projection
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [256, 256, 256],
        output_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        # GCN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.convs.append(GCNConv(prev_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim

        # Final projection
        self.projection = nn.Sequential(
            nn.Linear(prev_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.dropout = dropout

    def forward(self, batch: Batch) -> torch.Tensor:
        """
        Forward pass through GCN encoder.

        Args:
            batch: PyG Batch object with x, edge_index, batch attributes

        Returns:
            Graph-level representations [batch_size, output_dim]
        """
        x = batch.x
        edge_index = batch.edge_index

        # GCN layers
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling
        x = global_mean_pool(x, batch.batch)

        # Final projection
        x = self.projection(x)

        return x

    def get_output_dim(self) -> int:
        return self.output_dim


class GATEncoder(nn.Module):
    """
    Graph Attention Network encoder.

    Architecture:
    - Multiple GAT layers with multi-head attention
    - Global mean pooling for graph-level representation
    - Final MLP projection
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [256, 256, 256],
        output_dim: int = 256,
        heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.heads = heads

        # GAT layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            # Last layer uses concat=False (average heads)
            concat = (i < len(hidden_dims) - 1)
            out_channels = hidden_dim // heads if concat else hidden_dim

            self.convs.append(GATConv(
                prev_dim,
                out_channels,
                heads=heads,
                concat=concat,
                dropout=dropout,
            ))

            actual_out = out_channels * heads if concat else hidden_dim
            self.batch_norms.append(nn.BatchNorm1d(actual_out))
            prev_dim = actual_out

        # Final projection
        self.projection = nn.Sequential(
            nn.Linear(prev_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.dropout = dropout

    def forward(self, batch: Batch) -> torch.Tensor:
        """
        Forward pass through GAT encoder.

        Args:
            batch: PyG Batch object with x, edge_index, batch attributes

        Returns:
            Graph-level representations [batch_size, output_dim]
        """
        x = batch.x
        edge_index = batch.edge_index

        # GAT layers
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling
        x = global_mean_pool(x, batch.batch)

        # Final projection
        x = self.projection(x)

        return x

    def get_output_dim(self) -> int:
        return self.output_dim


def create_gnn_encoder(
    encoder_type: str,
    input_dim: int,
    hidden_dims: list[int] = [256, 256, 256],
    output_dim: int = 256,
    dropout: float = 0.2,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create a GNN encoder.

    Args:
        encoder_type: 'gcn' or 'gat'
        input_dim: Input feature dimension
        hidden_dims: Hidden layer dimensions
        output_dim: Output dimension
        dropout: Dropout rate
        **kwargs: Additional arguments for specific encoder types

    Returns:
        GNN encoder module
    """
    if encoder_type.lower() == 'gcn':
        return GCNEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout,
        )
    elif encoder_type.lower() == 'gat':
        return GATEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout,
            heads=kwargs.get('heads', 4),
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
