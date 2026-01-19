"""PyTorch Dataset for multi-task molecular property prediction."""

from typing import Any
import numpy as np
import torch
from torch.utils.data import Dataset


class MultiTaskMoleculeDataset(Dataset):
    """
    Dataset for multi-task molecular property prediction.

    Handles missing labels (not all molecules have all properties) by
    using mask tensors to indicate which samples have valid labels.
    """

    def __init__(
        self,
        fingerprints: np.ndarray,
        labels: dict[str, np.ndarray],
        task_types: dict[str, str],
    ):
        """
        Args:
            fingerprints: Array of shape (N, fp_bits) with molecular fingerprints
            labels: Dict mapping task_name -> (N,) array with NaN for missing labels
            task_types: Dict mapping task_name -> 'classification' or 'regression'
        """
        self.fingerprints = torch.tensor(fingerprints, dtype=torch.float32)
        self.task_names = list(labels.keys())
        self.task_types = task_types

        # Process labels: convert NaN to 0 (placeholder) and create masks
        self.labels = {}
        self.masks = {}

        for task, values in labels.items():
            # Find valid (non-NaN) entries
            valid_mask = ~np.isnan(values)

            # Replace NaN with 0 for tensor creation
            values_clean = np.nan_to_num(values, nan=0.0)

            # Store as tensors
            self.labels[task] = torch.tensor(values_clean, dtype=torch.float32)
            self.masks[task] = torch.tensor(valid_mask, dtype=torch.float32)

            # Log statistics
            n_valid = valid_mask.sum()
            print(f"Task {task}: {n_valid}/{len(values)} valid labels "
                  f"({100*n_valid/len(values):.1f}%)")

    def __len__(self) -> int:
        return len(self.fingerprints)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            'fingerprint': self.fingerprints[idx],
            'labels': {task: self.labels[task][idx] for task in self.task_names},
            'masks': {task: self.masks[task][idx] for task in self.task_names},
        }

    def get_task_indices(self, task: str) -> list[int]:
        """Get indices of samples that have valid labels for a specific task."""
        return torch.where(self.masks[task] > 0)[0].tolist()

    def get_task_statistics(self) -> dict[str, dict[str, float]]:
        """Get statistics for each task (for normalization, etc.)."""
        stats = {}
        for task in self.task_names:
            mask = self.masks[task].numpy() > 0
            values = self.labels[task].numpy()[mask]

            if self.task_types[task] == 'classification':
                # For classification: class balance
                pos_rate = values.mean()
                stats[task] = {
                    'pos_rate': pos_rate,
                    'neg_rate': 1 - pos_rate,
                    'n_samples': len(values),
                }
            else:
                # For regression: mean and std
                stats[task] = {
                    'mean': values.mean(),
                    'std': values.std(),
                    'min': values.min(),
                    'max': values.max(),
                    'n_samples': len(values),
                }

        return stats


def multitask_collate_fn(batch: list[dict]) -> dict[str, Any]:
    """
    Collate function for DataLoader.

    Args:
        batch: List of samples from __getitem__

    Returns:
        Batched data with:
        - fingerprints: (B, fp_bits) tensor
        - labels: Dict[task] -> (B,) tensor
        - masks: Dict[task] -> (B,) tensor
    """
    fingerprints = torch.stack([item['fingerprint'] for item in batch])

    # Get task names from first item
    task_names = list(batch[0]['labels'].keys())

    labels = {
        task: torch.stack([item['labels'][task] for item in batch])
        for task in task_names
    }

    masks = {
        task: torch.stack([item['masks'][task] for item in batch])
        for task in task_names
    }

    return {
        'fingerprint': fingerprints,
        'labels': labels,
        'masks': masks,
    }


def create_data_loaders(
    fingerprints: np.ndarray,
    labels: dict[str, np.ndarray],
    task_types: dict[str, str],
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train, validation, and test data loaders.

    Args:
        fingerprints: Full fingerprint array (N, fp_bits)
        labels: Dict of full label arrays
        task_types: Dict mapping task -> type
        train_idx, val_idx, test_idx: Split indices
        batch_size: Batch size
        num_workers: Number of data loading workers

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create subset arrays
    train_fp = fingerprints[train_idx]
    val_fp = fingerprints[val_idx]
    test_fp = fingerprints[test_idx]

    train_labels = {task: arr[train_idx] for task, arr in labels.items()}
    val_labels = {task: arr[val_idx] for task, arr in labels.items()}
    test_labels = {task: arr[test_idx] for task, arr in labels.items()}

    # Create datasets
    print("\nCreating training dataset:")
    train_dataset = MultiTaskMoleculeDataset(train_fp, train_labels, task_types)

    print("\nCreating validation dataset:")
    val_dataset = MultiTaskMoleculeDataset(val_fp, val_labels, task_types)

    print("\nCreating test dataset:")
    test_dataset = MultiTaskMoleculeDataset(test_fp, test_labels, task_types)

    # Create loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=multitask_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=multitask_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=multitask_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
