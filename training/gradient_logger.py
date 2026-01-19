"""
Gradient Conflict Logger - Core innovation for gradient-based causal discovery.

This module computes per-task gradients and measures their pairwise cosine
similarities to construct a gradient conflict matrix. This matrix reveals
task relationships:
- Positive similarity: tasks are synergistic (gradients align)
- Near-zero similarity: tasks are independent (gradients orthogonal)
- Negative similarity: tasks conflict (gradients oppose)
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn


class GradientConflictLogger:
    """
    Logs gradient conflicts between tasks during multi-task learning.

    The gradient conflict matrix G has entries:
        G[i,j] = cos(∇L_i, ∇L_j) = (∇L_i · ∇L_j) / (||∇L_i|| ||∇L_j||)

    where ∇L_k is the gradient of task k's loss w.r.t. shared encoder parameters.
    """

    def __init__(
        self,
        model: nn.Module,
        task_names: list[str],
        log_interval: int = 10,
        device: torch.device | None = None,
    ):
        """
        Args:
            model: MultiTaskModel with get_encoder_parameters() method
            task_names: List of task names (determines matrix order)
            log_interval: Log gradients every N training steps
            device: Device for computations (default: same as model)
        """
        self.model = model
        self.task_names = task_names
        self.log_interval = log_interval
        self.n_tasks = len(task_names)

        if device is None:
            # Get device from model
            device = next(model.parameters()).device
        self.device = device

        # Running sum for time-averaged conflict matrix
        self.conflict_sum = np.zeros((self.n_tasks, self.n_tasks))
        self.log_count = 0

        # Store history for analysis
        self.conflict_history = []

        # Track which steps we logged
        self.logged_steps = []

    def compute_task_gradients(
        self,
        task_losses: dict[str, torch.Tensor],
        retain_graph_after: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Compute gradient of each task loss w.r.t. shared encoder parameters.

        Uses torch.autograd.grad with retain_graph=True to compute
        gradients for multiple tasks from the same forward pass.

        Args:
            task_losses: Dict mapping task -> scalar loss tensor (with grad_fn)
            retain_graph_after: If True, keep graph for subsequent backward()

        Returns:
            Dict mapping task -> flattened gradient vector
        """
        encoder_params = list(self.model.get_encoder_parameters())
        task_gradients = {}

        # Sort tasks for consistent ordering
        tasks_to_process = [t for t in self.task_names if t in task_losses]

        for i, task in enumerate(tasks_to_process):
            loss = task_losses[task]

            # Skip if loss is zero or has no grad_fn
            if not loss.requires_grad:
                continue

            # Check if loss is effectively zero (no valid samples)
            if loss.item() == 0.0:
                continue

            # Always retain graph since we need it for the main backward() call
            # after gradient logging, or for subsequent tasks
            try:
                grads = torch.autograd.grad(
                    outputs=loss,
                    inputs=encoder_params,
                    retain_graph=True,  # Always retain for main backward()
                    create_graph=False,
                    allow_unused=True,
                )

                # Flatten and concatenate all parameter gradients
                grad_parts = []
                for g, p in zip(grads, encoder_params):
                    if g is not None:
                        grad_parts.append(g.flatten())
                    else:
                        # Parameter not used by this task - use zeros
                        grad_parts.append(torch.zeros_like(p).flatten())

                grad_vec = torch.cat(grad_parts)
                task_gradients[task] = grad_vec

            except RuntimeError as e:
                # Graph might have been freed - skip this task
                print(f"Warning: Could not compute gradient for {task}: {e}")
                continue

        return task_gradients

    def compute_conflict_matrix(
        self,
        task_gradients: dict[str, torch.Tensor],
    ) -> np.ndarray:
        """
        Compute pairwise cosine similarity matrix.

        G[i,j] = cos(grad_i, grad_j)

        Interpretation:
        - G[i,j] ≈ 1: Tasks strongly aligned (synergistic)
        - G[i,j] ≈ 0: Tasks orthogonal (independent)
        - G[i,j] ≈ -1: Tasks strongly opposed (conflicting)

        Args:
            task_gradients: Dict mapping task -> flattened gradient vector

        Returns:
            (n_tasks, n_tasks) cosine similarity matrix
        """
        matrix = np.zeros((self.n_tasks, self.n_tasks))

        for i, task_i in enumerate(self.task_names):
            if task_i not in task_gradients:
                # Task had no valid samples - leave row/col as zeros
                continue

            grad_i = task_gradients[task_i]
            norm_i = grad_i.norm().item()

            for j, task_j in enumerate(self.task_names):
                if task_j not in task_gradients:
                    continue

                grad_j = task_gradients[task_j]
                norm_j = grad_j.norm().item()

                # Compute cosine similarity with numerical stability
                eps = 1e-8
                if norm_i > eps and norm_j > eps:
                    cosine = torch.dot(grad_i, grad_j).item() / (norm_i * norm_j)
                    # Clamp to [-1, 1] for numerical stability
                    cosine = max(-1.0, min(1.0, cosine))
                    matrix[i, j] = cosine
                else:
                    # One or both gradients are zero
                    matrix[i, j] = 0.0

        return matrix

    def log_step(
        self,
        step: int,
        task_losses: dict[str, torch.Tensor],
    ) -> np.ndarray | None:
        """
        Log gradient conflicts at this training step.

        Only computes gradients every log_interval steps.

        Args:
            step: Current training step
            task_losses: Dict mapping task -> scalar loss tensor

        Returns:
            Conflict matrix if logged at this step, None otherwise
        """
        if step % self.log_interval != 0:
            return None

        # Compute per-task gradients
        task_gradients = self.compute_task_gradients(task_losses)

        if len(task_gradients) < 2:
            # Need at least 2 tasks with valid gradients
            return None

        # Compute conflict matrix
        matrix = self.compute_conflict_matrix(task_gradients)

        # Update running average
        self.conflict_sum += matrix
        self.log_count += 1
        self.conflict_history.append(matrix.copy())
        self.logged_steps.append(step)

        return matrix

    def get_averaged_conflict_matrix(self) -> np.ndarray:
        """
        Get time-averaged conflict matrix.

        Returns:
            (n_tasks, n_tasks) matrix of average cosine similarities
        """
        if self.log_count == 0:
            return np.zeros((self.n_tasks, self.n_tasks))
        return self.conflict_sum / self.log_count

    def get_conflict_history(self) -> np.ndarray:
        """
        Get full history of conflict matrices.

        Returns:
            (n_logged, n_tasks, n_tasks) array
        """
        if not self.conflict_history:
            return np.array([])
        return np.stack(self.conflict_history)

    def get_pairwise_conflict(self, task_a: str, task_b: str) -> float:
        """
        Get average conflict between two specific tasks.

        Args:
            task_a, task_b: Task names

        Returns:
            Average cosine similarity between task gradients
        """
        if task_a not in self.task_names or task_b not in self.task_names:
            raise ValueError(f"Unknown task: {task_a} or {task_b}")

        i = self.task_names.index(task_a)
        j = self.task_names.index(task_b)

        return self.get_averaged_conflict_matrix()[i, j]

    def get_conflict_evolution(self, task_a: str, task_b: str) -> np.ndarray:
        """
        Get evolution of conflict between two tasks over training.

        Args:
            task_a, task_b: Task names

        Returns:
            Array of cosine similarities at each logged step
        """
        i = self.task_names.index(task_a)
        j = self.task_names.index(task_b)

        history = self.get_conflict_history()
        if len(history) == 0:
            return np.array([])

        return history[:, i, j]

    def save(self, output_path: Path):
        """
        Save conflict matrices to disk.

        Args:
            output_path: Path for .npz file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            output_path,
            averaged=self.get_averaged_conflict_matrix(),
            history=self.get_conflict_history(),
            task_names=np.array(self.task_names),
            logged_steps=np.array(self.logged_steps),
            log_count=self.log_count,
        )
        print(f"Saved gradient conflict data to {output_path}")

    @classmethod
    def load(cls, path: Path) -> dict:
        """
        Load saved conflict matrices.

        Args:
            path: Path to .npz file

        Returns:
            Dict with 'averaged', 'history', 'task_names', etc.
        """
        data = np.load(path, allow_pickle=True)
        return {
            'averaged': data['averaged'],
            'history': data['history'],
            'task_names': data['task_names'].tolist(),
            'logged_steps': data['logged_steps'],
            'log_count': data['log_count'].item(),
        }

    def summary(self) -> str:
        """Get summary of logged conflicts."""
        matrix = self.get_averaged_conflict_matrix()

        lines = [
            f"Gradient Conflict Summary",
            f"=" * 50,
            f"Tasks: {', '.join(self.task_names)}",
            f"Logged steps: {self.log_count}",
            f"",
            f"Average Conflict Matrix:",
        ]

        # Format matrix
        header = "          " + "  ".join(f"{t[:8]:>8}" for t in self.task_names)
        lines.append(header)

        for i, task_i in enumerate(self.task_names):
            row = f"{task_i[:8]:>8}  "
            row += "  ".join(f"{matrix[i,j]:>8.3f}" for j in range(self.n_tasks))
            lines.append(row)

        # Key pairs
        lines.extend(["", "Key Relationships:"])
        pairs = []
        for i, task_i in enumerate(self.task_names):
            for j, task_j in enumerate(self.task_names):
                if i < j:
                    pairs.append((task_i, task_j, matrix[i, j]))

        # Sort by absolute value
        pairs.sort(key=lambda x: -abs(x[2]))

        for task_i, task_j, val in pairs[:5]:
            if val > 0.3:
                relation = "synergistic"
            elif val < -0.3:
                relation = "conflicting"
            else:
                relation = "independent"
            lines.append(f"  {task_i} vs {task_j}: {val:+.3f} ({relation})")

        return '\n'.join(lines)
