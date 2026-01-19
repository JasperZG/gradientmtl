"""Training loop for GNN-based multi-task learning with gradient logging."""

from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score

from .losses import MultiTaskLoss
from .gradient_logger import GradientConflictLogger


class GNNMultiTaskTrainer:
    """
    Trainer for GNN-based multi-task molecular property prediction.

    Similar to MultiTaskTrainer but handles PyG Batch objects.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        task_types: dict[str, str],
        config: dict,
        device: torch.device,
        output_dir: Path | None = None,
    ):
        """
        Args:
            model: GNNMultiTaskModel
            train_loader: Training data loader (PyG DataLoader)
            val_loader: Validation data loader
            task_types: Dict mapping task -> 'classification' or 'regression'
            config: Training configuration dict
            device: Device for training
            output_dir: Directory for saving outputs
        """
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.task_types = task_types
        self.task_names = list(task_types.keys())

        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Loss function
        self.loss_fn = MultiTaskLoss(task_types)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs'],
            eta_min=config['learning_rate'] / 100,
        )

        # Gradient logger
        self.gradient_logger = GradientConflictLogger(
            model=model,
            task_names=self.task_names,
            log_interval=config['gradient_log_interval'],
            device=device,
        )

        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None

        # Output directory
        self.output_dir = Path(output_dir) if output_dir else Path('outputs')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'task_metrics': [],
        }

    def _batch_to_device(self, batch_graph):
        """Move PyG batch to device."""
        return batch_graph.to(self.device)

    def _extract_labels_masks(self, batch_graph):
        """Extract labels and masks from batched graph."""
        # Labels and masks are stacked in the batch
        # PyG stacks graph-level attributes into [batch_size, n_tasks]
        labels_tensor = batch_graph.y  # Should be [batch_size, n_tasks]
        masks_tensor = batch_graph.mask  # Should be [batch_size, n_tasks]

        # Handle case where tensors might need reshaping
        batch_size = batch_graph.num_graphs
        n_tasks = len(self.task_names)

        if labels_tensor.dim() == 1:
            # Reshape from [batch_size * n_tasks] to [batch_size, n_tasks]
            labels_tensor = labels_tensor.view(batch_size, n_tasks)
            masks_tensor = masks_tensor.view(batch_size, n_tasks)

        # Convert to dict format expected by loss function
        # TaskHead outputs [batch_size], so labels should be [batch_size]
        labels = {}
        masks = {}
        for i, task in enumerate(self.task_names):
            labels[task] = labels_tensor[:, i]  # Shape: [batch_size]
            masks[task] = masks_tensor[:, i]    # Shape: [batch_size]

        return labels, masks

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {task: [] for task in self.task_names}
        total_losses = []

        for batch_idx, batch_graph in enumerate(self.train_loader):
            global_step = epoch * len(self.train_loader) + batch_idx

            # Move batch to device
            batch_graph = self._batch_to_device(batch_graph)
            labels, masks = self._extract_labels_masks(batch_graph)

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(batch_graph)

            # Compute individual task losses (for gradient logging)
            task_losses = self.loss_fn.get_individual_losses(
                predictions, labels, masks
            )

            # Log gradients BEFORE backward pass
            self.gradient_logger.log_step(global_step, task_losses)

            # Compute total loss and backward
            total_loss, _ = self.loss_fn(predictions, labels, masks)
            total_loss.backward()

            # Gradient clipping
            if self.config.get('gradient_clip_norm'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['gradient_clip_norm']
                )

            # Optimizer step
            self.optimizer.step()

            # Track losses
            total_losses.append(total_loss.item())
            for task, loss in task_losses.items():
                epoch_losses[task].append(loss.item())

        # Average losses
        avg_losses = {task: np.mean(losses) if losses else 0.0
                     for task, losses in epoch_losses.items()}
        avg_losses['total'] = np.mean(total_losses)

        return avg_losses

    @torch.no_grad()
    def validate(self) -> tuple[float, dict[str, Any]]:
        """Validate model and compute metrics."""
        self.model.eval()

        all_preds = {task: [] for task in self.task_names}
        all_labels = {task: [] for task in self.task_names}
        all_masks = {task: [] for task in self.task_names}
        total_losses = []

        for batch_graph in self.val_loader:
            batch_graph = self._batch_to_device(batch_graph)
            labels, masks = self._extract_labels_masks(batch_graph)

            predictions = self.model(batch_graph)
            total_loss, _ = self.loss_fn(predictions, labels, masks)
            total_losses.append(total_loss.item())

            for task in self.task_names:
                all_preds[task].append(predictions[task].cpu())
                all_labels[task].append(labels[task].cpu())
                all_masks[task].append(masks[task].cpu())

        # Concatenate
        for task in self.task_names:
            all_preds[task] = torch.cat(all_preds[task]).numpy()
            all_labels[task] = torch.cat(all_labels[task]).numpy()
            all_masks[task] = torch.cat(all_masks[task]).numpy()

        # Compute metrics
        metrics = {}
        for task in self.task_names:
            mask = all_masks[task].flatten() > 0
            if mask.sum() == 0:
                continue

            preds = all_preds[task].flatten()[mask]
            labels_arr = all_labels[task].flatten()[mask]

            if self.task_types[task] == 'classification':
                probs = 1 / (1 + np.exp(-preds))

                try:
                    if len(np.unique(labels_arr)) > 1:
                        auc = roc_auc_score(labels_arr, probs)
                    else:
                        auc = 0.5
                except ValueError:
                    auc = 0.5

                metrics[task] = {
                    'roc_auc': auc,
                    'n_samples': int(mask.sum()),
                }
            else:
                rmse = np.sqrt(np.mean((preds - labels_arr) ** 2))
                mae = np.mean(np.abs(preds - labels_arr))

                metrics[task] = {
                    'rmse': rmse,
                    'mae': mae,
                    'n_samples': int(mask.sum()),
                }

        avg_val_loss = np.mean(total_losses)
        return avg_val_loss, metrics

    def train(self) -> dict[str, Any]:
        """Full training loop with early stopping."""
        print(f"\nStarting GNN training for {self.config['epochs']} epochs...")
        print(f"Device: {self.device}")
        print(f"Gradient logging every {self.config['gradient_log_interval']} steps")
        print()

        for epoch in range(self.config['epochs']):
            # Train
            train_losses = self.train_epoch(epoch)

            # Validate
            val_loss, val_metrics = self.validate()

            # Update learning rate
            self.scheduler.step()

            # Store history
            self.history['train_loss'].append(train_losses['total'])
            self.history['val_loss'].append(val_loss)
            self.history['task_metrics'].append(val_metrics)

            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
            else:
                self.patience_counter += 1

            # Print progress
            metrics_str = ", ".join(
                f"{task}: {m.get('roc_auc', m.get('rmse', 0)):.3f}"
                for task, m in val_metrics.items()
            )
            print(f"Epoch {epoch+1:3d} | "
                  f"Train: {train_losses['total']:.4f} | "
                  f"Val: {val_loss:.4f} | "
                  f"{metrics_str}")

            # Check early stopping
            if self.patience_counter >= self.config['early_stopping_patience']:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        # Save outputs
        self._save_outputs()

        # Final validation
        final_val_loss, final_metrics = self.validate()

        print(f"\nTraining complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"\nFinal metrics:")
        for task, m in final_metrics.items():
            if self.task_types[task] == 'classification':
                print(f"  {task}: ROC-AUC = {m['roc_auc']:.3f}")
            else:
                print(f"  {task}: RMSE = {m['rmse']:.3f}")

        # Print gradient conflict summary
        print(f"\n{self.gradient_logger.summary()}")

        return {
            'conflict_matrix': self.gradient_logger.get_averaged_conflict_matrix(),
            'task_names': self.task_names,
            'final_metrics': final_metrics,
            'history': self.history,
            'best_val_loss': self.best_val_loss,
        }

    def _save_outputs(self):
        """Save model checkpoint and gradient data."""
        # Save model
        checkpoint_path = self.output_dir / 'checkpoints' / 'best_gnn_model.pt'
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.best_model_state or self.model.state_dict(),
            'config': self.config,
            'task_types': self.task_types,
            'best_val_loss': self.best_val_loss,
        }, checkpoint_path)
        print(f"Saved model checkpoint to {checkpoint_path}")

        # Save gradient conflict data
        gradient_path = self.output_dir / 'gradients' / 'gnn_conflict_matrices.npz'
        self.gradient_logger.save(gradient_path)

    def load_checkpoint(self, path: Path):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {path}")
