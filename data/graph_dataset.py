"""
PyTorch Geometric Dataset for multi-task molecular property prediction.

Handles missing labels with mask tensors.
"""

import numpy as np
import torch
from torch_geometric.data import Data, Dataset


class MultiTaskGraphDataset(Dataset):
    """
    Dataset for multi-task learning with molecular graphs.

    Each sample contains:
    - Molecular graph (nodes, edges, features)
    - Labels for multiple tasks (NaN for missing)
    - Mask indicating which labels are valid
    """

    def __init__(
        self,
        graphs: list[Data],
        labels: dict[str, np.ndarray],
        task_types: dict[str, str],
    ):
        """
        Args:
            graphs: List of PyG Data objects
            labels: Dict mapping task_name -> array of labels (may contain NaN)
            task_types: Dict mapping task_name -> 'classification' or 'regression'
        """
        super().__init__()
        self.graphs = graphs
        self.task_names = list(task_types.keys())
        self.task_types = task_types
        self.n_tasks = len(self.task_names)

        # Convert labels to tensors
        self.labels = torch.zeros(len(graphs), self.n_tasks)
        self.masks = torch.zeros(len(graphs), self.n_tasks)

        for task_idx, task_name in enumerate(self.task_names):
            if task_name in labels:
                task_labels = labels[task_name]
                for i, val in enumerate(task_labels):
                    if not np.isnan(val):
                        self.labels[i, task_idx] = float(val)
                        self.masks[i, task_idx] = 1.0

        # Print statistics
        print(f"  Samples: {len(graphs)}")
        print(f"  Tasks: {self.n_tasks}")
        for task_idx, task_name in enumerate(self.task_names):
            n_valid = int(self.masks[:, task_idx].sum())
            pct = 100 * n_valid / len(graphs)
            print(f"    {task_name}: {n_valid} labels ({pct:.1f}%)")

    def len(self) -> int:
        return len(self.graphs)

    def get(self, idx: int) -> Data:
        """Get a single sample with labels and mask attached to the graph."""
        graph = self.graphs[idx].clone()

        # Attach labels and mask to the graph
        graph.y = self.labels[idx]
        graph.mask = self.masks[idx]

        return graph


def graph_collate_fn(batch: list[Data]) -> tuple:
    """
    Custom collate function for multi-task graph data.

    Returns a PyG Batch object plus separate label and mask tensors.
    """
    from torch_geometric.data import Batch

    # Create batched graph
    batch_graph = Batch.from_data_list(batch)

    # Stack labels and masks
    labels = torch.stack([data.y for data in batch])
    masks = torch.stack([data.mask for data in batch])

    return batch_graph, labels, masks
