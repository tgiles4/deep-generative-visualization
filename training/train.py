"""
Training loop with checkpointing and Manim hook integration.
Uses PyTorch Lightning for efficient training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pytorch_lightning as pl
from dataclasses import dataclass
from typing import Callable, Optional
from torch.utils.data import DataLoader
from training.checkpoint_manager import CheckpointManager


@dataclass
class DataLoaders:
    """Container for train, validation, and test data loaders."""
    train: DataLoader
    test: DataLoader
    val: Optional[DataLoader] = None


class VisualizationLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for models with latent space visualization.
    Handles training, validation, testing, and checkpointing with Manim hooks.
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module | dict[str, tuple[nn.Module, float]] | Callable | None = None,
        optimizer: type[optim.Optimizer] = optim.Adam,
        optimizer_kwargs: dict | None = None,
        metrics: list[tuple[str, Callable]] | None = None,
        checkpoint_dir: str = './checkpoints',
        save_every_n_epochs: int = 1,
        save_images: bool = False,
        learning_rate: float = 0.001,
    ):
        """
        Initialize Lightning module.
        
        Args:
            model: Model to train (should inherit from BaseLatentModel)
            criterion: Loss function. Can be:
                - Single nn.Module (e.g., nn.CrossEntropyLoss())
                - dict[str, tuple[nn.Module, float]]: Multiple criteria with weights
                - Callable: Custom loss function that takes (output, target) and
                  returns either a scalar tensor or dict[str, tensor] for component tracking
            optimizer: Optimizer class (default: optim.Adam)
            optimizer_kwargs: Additional kwargs for optimizer (default: {'lr': learning_rate})
            metrics: list of (name, callable) tuples. Each callable takes (y, y_hat) and returns float.
                    Default: [('accuracy', accuracy_fn)]
            checkpoint_dir: Directory to save checkpoints
            save_every_n_epochs: Save checkpoint every N epochs
            save_images: Whether to save original images for comparison visualizations
            learning_rate: Learning rate
        """
        super().__init__()
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        self.save_every_n_epochs = save_every_n_epochs
        self.save_images = save_images
        self.learning_rate = learning_rate
        
        # Setup criterion
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        self.criterion = criterion
        
        # Setup optimizer
        self.optimizer_class = optimizer
        if optimizer_kwargs is None:
            optimizer_kwargs = {'lr': learning_rate}
        self.optimizer_kwargs = optimizer_kwargs
        
        # Setup metrics (standardized as callables taking (y, y_hat))
        if metrics is None:
            # Default accuracy metric
            def accuracy_fn(y, y_hat):
                if y_hat.dim() > 1:
                    y_hat = y_hat.argmax(dim=1)
                return (y_hat == y).float().mean().item() * 100.0
            metrics = [('accuracy', accuracy_fn)]
        self.metrics = metrics
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        
        # Track predictions for metrics computation
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
    
    def _compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute loss and return both total loss and component dict for metrics.
        
        Args:
            output: Model output
            target: Target labels
            
        Returns:
            tuple: (total_loss_tensor, loss_components_dict)
        """
        criterion = self.criterion
        
        if isinstance(criterion, dict):
            # Multiple criteria with weights
            total_loss = torch.tensor(0.0, device=output.device)
            components = {}
            for name, (loss_fn, weight) in criterion.items():
                loss_value = loss_fn(output, target)
                weighted_loss = loss_value * weight
                total_loss = total_loss + weighted_loss
                components[f'loss_{name}'] = loss_value.item()
                components[f'loss_{name}_weighted'] = weighted_loss.item()
            components['loss_total'] = total_loss.item()
            return total_loss, components
        
        elif isinstance(criterion, nn.Module):
            # Single nn.Module criterion (e.g., nn.CrossEntropyLoss, nn.MSELoss)
            loss_value = criterion(output, target)
            return loss_value, {'loss_total': loss_value.item()}
        
        elif callable(criterion):
            # Custom callable loss function (not an nn.Module)
            loss_result = criterion(output, target)
            
            if isinstance(loss_result, dict):
                # Returns dict of components
                total = sum(loss_result.values())
                components = {f'loss_{k}': v.item() if isinstance(v, torch.Tensor) else v 
                             for k, v in loss_result.items()}
                components['loss_total'] = total.item() if isinstance(total, torch.Tensor) else total
                return total, components
            else:
                # Returns scalar
                return loss_result, {'loss_total': loss_result.item() if isinstance(loss_result, torch.Tensor) else loss_result}
        
        else:
            raise TypeError(f"Unsupported criterion type: {type(criterion)}")
    
    def _compute_metrics(self, y: torch.Tensor, y_hat: torch.Tensor) -> dict[str, float]:
        """
        Compute all metrics using standardized callables.
        
        Args:
            y: True labels
            y_hat: Predicted outputs (logits or predictions)
            
        Returns:
            Dictionary of metric_name -> value
        """
        metrics_dict = {}
        for metric_name, metric_fn in self.metrics:
            try:
                value = metric_fn(y, y_hat)
                metrics_dict[metric_name] = value
            except Exception as e:
                # Skip metrics that fail (e.g., incompatible shapes)
                if self.global_rank == 0:  # Only print on main process
                    print(f"Warning: Metric '{metric_name}' failed: {e}")
        return metrics_dict
    
    def _flatten_if_needed(self, data: torch.Tensor) -> torch.Tensor:
        """Flatten image data if needed."""
        if data.dim() > 2:
            return data.view(data.size(0), -1)
        return data
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        data, target = batch
        data = self._flatten_if_needed(data)
        
        # Forward pass
        output = self.model(data)
        loss, loss_components = self._compute_loss(output, target)
        
        # Store outputs for metrics computation
        self.training_step_outputs.append({
            'output': output.detach(),
            'target': target.detach(),
            'loss': loss.item(),
            'loss_components': loss_components,
        })
        
        # Trigger batch end hook
        if batch_idx % 10 == 0:
            self.model._trigger_hook('on_batch_end', batch_idx, self.current_epoch)
        
        return loss
    
    def on_train_epoch_start(self):
        """Called at the start of training epoch."""
        self.training_step_outputs = []
        self.model._trigger_hook('on_epoch_start', self.current_epoch)
    
    def on_train_epoch_end(self):
        """Called at the end of training epoch."""
        if not self.training_step_outputs:
            return
        
        # Aggregate outputs
        all_outputs = torch.cat([x['output'] for x in self.training_step_outputs], dim=0)
        all_targets = torch.cat([x['target'] for x in self.training_step_outputs], dim=0)
        
        # Compute metrics
        train_metrics = self._compute_metrics(all_targets, all_outputs)
        
        # Aggregate loss components
        train_loss = sum(x['loss'] for x in self.training_step_outputs) / len(self.training_step_outputs)
        train_loss_components = {}
        for output_dict in self.training_step_outputs:
            for key, value in output_dict['loss_components'].items():
                if key not in train_loss_components:
                    train_loss_components[key] = 0.0
                train_loss_components[key] += value
        for key in train_loss_components:
            train_loss_components[key] /= len(self.training_step_outputs)
        
        # Combine all metrics
        metrics = {
            'train_loss': train_loss,
            **{f'train_{k}': v for k, v in train_loss_components.items()},
            **{f'train_{k}': v for k, v in train_metrics.items()},
        }
        
        # Log metrics
        self.log_dict(metrics, on_epoch=True, prog_bar=True)
        
        # Store for checkpointing
        self.train_metrics = metrics
        
        # Handle checkpointing and Manim hooks (only on main process)
        if self.global_rank == 0:
            self._save_checkpoint_and_trigger_hooks()
    
    def _save_checkpoint_and_trigger_hooks(self):
        """Save checkpoint and trigger Manim hooks."""
        # Get all metrics
        all_metrics = {**self.train_metrics}
        if hasattr(self, 'val_metrics'):
            all_metrics.update(self.val_metrics)
        if hasattr(self, 'test_metrics'):
            all_metrics.update(self.test_metrics)
        
        # Get latent vectors for visualization (from training data)
        train_loader = self.trainer.train_dataloader
        latent_vectors, labels = self.model.get_latent_vectors(train_loader, self.device)
        
        # Collect images if requested
        images = None
        if self.save_images:
            images_list = []
            self.model.eval()
            with torch.no_grad():
                for data, _ in train_loader:
                    images_list.append(data.cpu().numpy())
            images = np.concatenate(images_list, axis=0)
        
        # Save checkpoint
        if self.current_epoch % self.save_every_n_epochs == 0:
            self.checkpoint_manager.save_latent_snapshot(
                latent_vectors=latent_vectors,
                labels=labels,
                epoch=self.current_epoch,
                images=images,
            )
            self.checkpoint_manager.save_metrics(all_metrics, self.current_epoch)
        
        # Trigger epoch end hook
        self.model._trigger_hook(
            'on_epoch_end',
            self.current_epoch,
            latent_vectors,
            labels,
            all_metrics,
        )
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        data, target = batch
        data = self._flatten_if_needed(data)
        
        output = self.model(data)
        loss, loss_components = self._compute_loss(output, target)
        
        self.validation_step_outputs.append({
            'output': output.detach(),
            'target': target.detach(),
            'loss': loss.item(),
            'loss_components': loss_components,
        })
        
        return loss
    
    def on_validation_epoch_start(self):
        """Called at the start of validation epoch."""
        self.validation_step_outputs = []
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        if not self.validation_step_outputs:
            return
        
        # Aggregate outputs
        all_outputs = torch.cat([x['output'] for x in self.validation_step_outputs], dim=0)
        all_targets = torch.cat([x['target'] for x in self.validation_step_outputs], dim=0)
        
        # Compute metrics
        val_metrics = self._compute_metrics(all_targets, all_outputs)
        
        # Aggregate loss components
        val_loss = sum(x['loss'] for x in self.validation_step_outputs) / len(self.validation_step_outputs)
        val_loss_components = {}
        for output_dict in self.validation_step_outputs:
            for key, value in output_dict['loss_components'].items():
                if key not in val_loss_components:
                    val_loss_components[key] = 0.0
                val_loss_components[key] += value
        for key in val_loss_components:
            val_loss_components[key] /= len(self.validation_step_outputs)
        
        # Combine all metrics
        metrics = {
            'val_loss': val_loss,
            **{f'val_{k}': v for k, v in val_loss_components.items()},
            **{f'val_{k}': v for k, v in val_metrics.items()},
        }
        
        # Log metrics
        self.log_dict(metrics, on_epoch=True, prog_bar=True)
        
        # Store for checkpointing
        self.val_metrics = metrics
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        data, target = batch
        data = self._flatten_if_needed(data)
        
        output = self.model(data)
        loss, loss_components = self._compute_loss(output, target)
        
        self.test_step_outputs.append({
            'output': output.detach(),
            'target': target.detach(),
            'loss': loss.item(),
            'loss_components': loss_components,
        })
        
        return loss
    
    def on_test_epoch_start(self):
        """Called at the start of test epoch."""
        self.test_step_outputs = []
    
    def on_test_epoch_end(self):
        """Called at the end of test epoch."""
        if not self.test_step_outputs:
            return
        
        # Aggregate outputs
        all_outputs = torch.cat([x['output'] for x in self.test_step_outputs], dim=0)
        all_targets = torch.cat([x['target'] for x in self.test_step_outputs], dim=0)
        
        # Compute metrics
        test_metrics = self._compute_metrics(all_targets, all_outputs)
        
        # Aggregate loss components
        test_loss = sum(x['loss'] for x in self.test_step_outputs) / len(self.test_step_outputs)
        test_loss_components = {}
        for output_dict in self.test_step_outputs:
            for key, value in output_dict['loss_components'].items():
                if key not in test_loss_components:
                    test_loss_components[key] = 0.0
                test_loss_components[key] += value
        for key in test_loss_components:
            test_loss_components[key] /= len(self.test_step_outputs)
        
        # Combine all metrics
        metrics = {
            'test_loss': test_loss,
            **{f'test_{k}': v for k, v in test_loss_components.items()},
            **{f'test_{k}': v for k, v in test_metrics.items()},
        }
        
        # Log metrics
        self.log_dict(metrics, on_epoch=True)
        
        # Store for checkpointing
        self.test_metrics = metrics
    
    def configure_optimizers(self):
        """Configure optimizer."""
        return self.optimizer_class(self.model.parameters(), **self.optimizer_kwargs)
