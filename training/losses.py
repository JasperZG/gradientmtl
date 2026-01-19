"""Task-specific loss functions with masking support for multi-task learning."""

import torch
import torch.nn as nn


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss function with support for missing labels.

    Handles:
    - Classification tasks (BCE with logits)
    - Regression tasks (MSE)
    - Missing labels via masking
    - Uniform or weighted task combination
    """

    def __init__(
        self,
        task_types: dict[str, str],
        task_weights: dict[str, float] | None = None,
    ):
        """
        Args:
            task_types: Dict mapping task_name -> 'classification' or 'regression'
            task_weights: Optional dict mapping task_name -> weight (default: uniform)
        """
        super().__init__()

        self.task_types = task_types
        self.task_names = list(task_types.keys())

        # Default: uniform weighting
        if task_weights is None:
            task_weights = {task: 1.0 / len(task_types) for task in task_types}
        self.task_weights = task_weights

        # Loss functions (no reduction - we handle masking manually)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        labels: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute multi-task loss.

        Args:
            predictions: Dict mapping task -> (B,) logits/values
            labels: Dict mapping task -> (B,) targets
            masks: Dict mapping task -> (B,) binary mask (1 = valid, 0 = missing)

        Returns:
            total_loss: Scalar, weighted sum of task losses
            task_losses: Dict mapping task -> scalar loss (for logging/gradients)
        """
        task_losses = {}

        for task in self.task_names:
            pred = predictions[task]
            target = labels[task]
            mask = masks[task]

            # Compute per-sample loss
            if self.task_types[task] == 'classification':
                loss = self.bce_loss(pred, target)
            else:  # regression
                loss = self.mse_loss(pred, target)

            # Apply mask: only compute loss where labels exist
            valid_count = mask.sum()

            if valid_count > 0:
                # Masked mean loss
                masked_loss = (loss * mask).sum() / valid_count
            else:
                # No valid samples for this task in this batch
                # Return zero loss (will be handled specially in gradient logging)
                masked_loss = torch.tensor(0.0, device=pred.device, requires_grad=True)

            task_losses[task] = masked_loss

        # Weighted sum for total loss
        total_loss = sum(
            self.task_weights[task] * task_losses[task]
            for task in self.task_names
        )

        return total_loss, task_losses

    def get_individual_losses(
        self,
        predictions: dict[str, torch.Tensor],
        labels: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Get individual task losses without computing total.

        This is used for gradient logging where we need separate backward
        passes for each task.

        Args:
            predictions: Dict mapping task -> (B,) logits/values
            labels: Dict mapping task -> (B,) targets
            masks: Dict mapping task -> (B,) binary mask

        Returns:
            Dict mapping task -> scalar loss tensor (with grad_fn)
        """
        task_losses = {}

        for task in self.task_names:
            pred = predictions[task]
            target = labels[task]
            mask = masks[task]

            if self.task_types[task] == 'classification':
                loss = self.bce_loss(pred, target)
            else:
                loss = self.mse_loss(pred, target)

            valid_count = mask.sum()

            if valid_count > 0:
                masked_loss = (loss * mask).sum() / valid_count
            else:
                # Create a zero loss that still has grad_fn connected to predictions
                # This ensures we can still call backward() if needed
                masked_loss = pred.sum() * 0.0

            task_losses[task] = masked_loss

        return task_losses
