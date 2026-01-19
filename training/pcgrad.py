"""
PCGrad (Projecting Conflicting Gradients) optimizer wrapper.

Reference: Yu et al., "Gradient Surgery for Multi-Task Learning" (NeurIPS 2020)
https://arxiv.org/abs/2001.06782

PCGrad projects gradients from conflicting tasks to remove negative interference.
If grad_i · grad_j < 0, project grad_i onto the normal plane of grad_j.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
import copy
import random


class PCGrad:
    """
    PCGrad wrapper that modifies gradients to reduce task interference.

    Usage:
        pcgrad = PCGrad(optimizer)

        for batch in dataloader:
            optimizer.zero_grad()

            # Compute per-task losses
            task_losses = model.compute_task_losses(batch)

            # PCGrad backward (computes projected gradients)
            pcgrad.backward(task_losses)

            # Regular optimizer step
            optimizer.step()
    """

    def __init__(self, optimizer: torch.optim.Optimizer, reduction: str = 'mean'):
        """
        Args:
            optimizer: Base optimizer (e.g., Adam, AdamW)
            reduction: How to combine projected gradients ('mean' or 'sum')
        """
        self.optimizer = optimizer
        self.reduction = reduction
        self._reset_grad_buffer()

    def _reset_grad_buffer(self):
        """Reset gradient storage buffer."""
        self.grad_buffer = {}

    def zero_grad(self):
        """Zero gradients in optimizer."""
        self.optimizer.zero_grad()
        self._reset_grad_buffer()

    def step(self):
        """Perform optimizer step."""
        self.optimizer.step()

    @property
    def param_groups(self):
        """Access optimizer param groups."""
        return self.optimizer.param_groups

    def _get_grad_vector(self, loss: torch.Tensor,
                         shared_params: List[torch.nn.Parameter]) -> torch.Tensor:
        """
        Compute gradient vector for a single task loss.

        Args:
            loss: Scalar loss for one task
            shared_params: List of shared encoder parameters

        Returns:
            Flattened gradient vector
        """
        grads = torch.autograd.grad(
            outputs=loss,
            inputs=shared_params,
            retain_graph=True,
            allow_unused=True
        )

        # Handle None gradients (unused parameters)
        grad_list = []
        for g, p in zip(grads, shared_params):
            if g is None:
                grad_list.append(torch.zeros_like(p).flatten())
            else:
                grad_list.append(g.flatten())

        return torch.cat(grad_list)

    def _set_grad_from_vector(self, grad_vector: torch.Tensor,
                               shared_params: List[torch.nn.Parameter]):
        """
        Set parameter gradients from a flattened gradient vector.

        Args:
            grad_vector: Flattened gradient vector
            shared_params: List of shared encoder parameters
        """
        offset = 0
        for param in shared_params:
            param_size = param.numel()
            param_grad = grad_vector[offset:offset + param_size].view_as(param)

            if param.grad is None:
                param.grad = param_grad.clone()
            else:
                param.grad.copy_(param_grad)

            offset += param_size

    def _project_gradient(self, grad_i: torch.Tensor,
                          grad_j: torch.Tensor) -> torch.Tensor:
        """
        Project grad_i onto the normal plane of grad_j if they conflict.

        If grad_i · grad_j < 0:
            grad_i' = grad_i - (grad_i · grad_j / ||grad_j||²) * grad_j

        Args:
            grad_i: Gradient to potentially project
            grad_j: Reference gradient

        Returns:
            Projected gradient (or original if no conflict)
        """
        dot_product = torch.dot(grad_i, grad_j)

        if dot_product < 0:
            # Conflict detected - project grad_i
            grad_j_norm_sq = torch.dot(grad_j, grad_j)
            if grad_j_norm_sq > 1e-12:  # Avoid division by zero
                projection = (dot_product / grad_j_norm_sq) * grad_j
                return grad_i - projection

        return grad_i

    def backward(self, task_losses: Dict[str, torch.Tensor],
                 shared_params: Optional[List[torch.nn.Parameter]] = None,
                 head_params: Optional[Dict[str, List[torch.nn.Parameter]]] = None):
        """
        Compute PCGrad-modified gradients and set them on parameters.

        Args:
            task_losses: Dict mapping task names to scalar losses
            shared_params: List of shared encoder parameters (required)
            head_params: Optional dict mapping task names to head parameters
                        (these get normal gradients, not PCGrad)
        """
        if shared_params is None:
            raise ValueError("shared_params must be provided for PCGrad")

        if len(task_losses) == 0:
            return

        task_names = list(task_losses.keys())
        n_tasks = len(task_names)

        # Compute per-task gradients for shared parameters
        task_grads = {}
        for task_name in task_names:
            loss = task_losses[task_name]
            if loss is not None and loss.requires_grad:
                task_grads[task_name] = self._get_grad_vector(loss, shared_params)

        if len(task_grads) == 0:
            return

        # Apply PCGrad projection
        # Random order for fairness (as in original paper)
        task_order = list(task_grads.keys())
        random.shuffle(task_order)

        projected_grads = {}
        for task_i in task_order:
            grad_i = task_grads[task_i].clone()

            # Project against all other task gradients
            for task_j in task_order:
                if task_i != task_j:
                    grad_j = task_grads[task_j]
                    grad_i = self._project_gradient(grad_i, grad_j)

            projected_grads[task_i] = grad_i

        # Combine projected gradients
        stacked_grads = torch.stack(list(projected_grads.values()))
        if self.reduction == 'mean':
            final_grad = stacked_grads.mean(dim=0)
        else:  # sum
            final_grad = stacked_grads.sum(dim=0)

        # Set gradients on shared parameters
        self._set_grad_from_vector(final_grad, shared_params)

        # Handle head parameters with regular backprop
        if head_params is not None:
            for task_name, params in head_params.items():
                if task_name in task_losses and task_losses[task_name] is not None:
                    loss = task_losses[task_name]
                    if loss.requires_grad:
                        # Compute and accumulate head gradients
                        head_grads = torch.autograd.grad(
                            outputs=loss,
                            inputs=params,
                            retain_graph=True,
                            allow_unused=True
                        )
                        for param, grad in zip(params, head_grads):
                            if grad is not None:
                                if param.grad is None:
                                    param.grad = grad.clone()
                                else:
                                    param.grad.add_(grad)


class PCGradTrainer:
    """
    Trainer that uses PCGrad for gradient conflict resolution.

    Compares performance with and without PCGrad to validate
    that gradient conflicts (detected by our G matrix) are real.
    """

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 task_names: List[str], use_pcgrad: bool = True,
                 device: str = 'cuda'):
        """
        Args:
            model: Multi-task model with encoder and task heads
            optimizer: Base optimizer
            task_names: List of task names
            use_pcgrad: Whether to use PCGrad (False = baseline)
            device: Device to use
        """
        self.model = model
        self.task_names = task_names
        self.use_pcgrad = use_pcgrad
        self.device = device

        if use_pcgrad:
            self.pcgrad = PCGrad(optimizer)
        else:
            self.pcgrad = None
            self.optimizer = optimizer

        # Get shared parameters (encoder)
        self.shared_params = list(model.encoder.parameters())

        # Get head parameters
        self.head_params = {}
        for task_name in task_names:
            if hasattr(model, 'heads') and task_name in model.heads:
                self.head_params[task_name] = list(model.heads[task_name].parameters())

    def train_step(self, batch_data, batch_labels: torch.Tensor,
                   batch_masks: torch.Tensor) -> Tuple[float, Dict[str, float]]:
        """
        Perform one training step with optional PCGrad.

        Args:
            batch_data: Input features or graph data
            batch_labels: (batch_size, n_tasks) tensor of labels
            batch_masks: (batch_size, n_tasks) tensor of valid label masks

        Returns:
            total_loss: Combined loss value
            task_losses_dict: Dict of individual task losses
        """
        self.model.train()

        if self.use_pcgrad:
            self.pcgrad.zero_grad()
        else:
            self.optimizer.zero_grad()

        # Forward pass
        outputs = self.model(batch_data)

        # Compute per-task losses
        task_losses = {}
        loss_values = {}

        for i, task_name in enumerate(self.task_names):
            if task_name not in outputs:
                continue

            pred = outputs[task_name]

            # Handle label dimensions
            if batch_labels.dim() == 1:
                labels = batch_labels
                masks = batch_masks
            else:
                labels = batch_labels[:, i]
                masks = batch_masks[:, i]

            # Skip if no valid labels
            if masks.sum() == 0:
                continue

            # Masked loss computation
            pred_masked = pred[masks.bool()].squeeze()
            labels_masked = labels[masks.bool()]

            if len(pred_masked) == 0:
                continue

            # Use appropriate loss (BCE for classification, MSE for regression)
            # Determine by checking if labels are binary
            unique_labels = torch.unique(labels_masked)
            is_classification = len(unique_labels) <= 2 and all(l in [0, 1] for l in unique_labels.tolist())

            if is_classification:
                loss = nn.functional.binary_cross_entropy_with_logits(
                    pred_masked, labels_masked.float()
                )
            else:
                loss = nn.functional.mse_loss(pred_masked, labels_masked.float())

            task_losses[task_name] = loss
            loss_values[task_name] = loss.item()

        if len(task_losses) == 0:
            return 0.0, {}

        # Backward pass
        if self.use_pcgrad:
            self.pcgrad.backward(
                task_losses,
                shared_params=self.shared_params,
                head_params=self.head_params
            )
            self.pcgrad.step()
        else:
            # Standard multi-task backward
            total_loss = sum(task_losses.values())
            total_loss.backward()
            self.optimizer.step()

        total_loss_value = sum(loss_values.values())
        return total_loss_value, loss_values


def compute_gradient_conflict_stats(task_grads: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Compute gradient conflict statistics for logging.

    Args:
        task_grads: Dict mapping task names to gradient vectors

    Returns:
        Dict with conflict statistics
    """
    task_names = list(task_grads.keys())
    n_tasks = len(task_names)

    if n_tasks < 2:
        return {'n_conflicts': 0, 'avg_conflict_magnitude': 0.0}

    n_conflicts = 0
    conflict_magnitudes = []

    for i, task_i in enumerate(task_names):
        for j, task_j in enumerate(task_names):
            if i >= j:
                continue

            grad_i = task_grads[task_i]
            grad_j = task_grads[task_j]

            # Cosine similarity
            norm_i = grad_i.norm()
            norm_j = grad_j.norm()

            if norm_i > 1e-12 and norm_j > 1e-12:
                cos_sim = torch.dot(grad_i, grad_j) / (norm_i * norm_j)

                if cos_sim < 0:
                    n_conflicts += 1
                    conflict_magnitudes.append(-cos_sim.item())

    avg_magnitude = sum(conflict_magnitudes) / len(conflict_magnitudes) if conflict_magnitudes else 0.0

    return {
        'n_conflicts': n_conflicts,
        'avg_conflict_magnitude': avg_magnitude,
        'total_pairs': n_tasks * (n_tasks - 1) // 2,
        'conflict_ratio': n_conflicts / (n_tasks * (n_tasks - 1) // 2) if n_tasks > 1 else 0
    }
