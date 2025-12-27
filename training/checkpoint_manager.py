"""
Checkpoint management for latent space snapshots and training metrics.
Focuses on data needed for visualization and post-training analysis.
"""

import json
import h5py
import numpy as np
from pathlib import Path
from typing import Any, Optional


class CheckpointManager:
    """
    Manages saving and loading latent space snapshots and training metrics.
    Uses HDF5 for efficient storage of latent vectors, labels, and optional image data.
    """
    
    def __init__(self, checkpoint_dir: str):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save/load checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_latent_snapshot(
        self,
        latent_vectors: np.ndarray,
        labels: Optional[np.ndarray],
        epoch: int,
        sample_nums: Optional[np.ndarray] = None,
        images: Optional[np.ndarray] = None,
    ):
        """
        Save latent space snapshot to HDF5 with per-sample tracking.
        
        Schema:
        - latent_vectors: (n_samples, latent_dim) - the visualization dimensions
        - labels: (n_samples,) - class labels
        - epoch: scalar - training epoch
        - sample_nums: (n_samples,) - sample indices for tracking across epochs
        - images: (n_samples, ...) - optional original images for comparison
        
        Args:
            latent_vectors: Array of shape (n_samples, latent_dim)
            labels: Optional array of shape (n_samples,) with class labels
            epoch: Epoch number
            sample_nums: Optional array of shape (n_samples,) with sample indices.
                        If None, will be auto-generated as [0, 1, 2, ..., n_samples-1]
            images: Optional array of original images for comparison visualizations.
                   Useful for comparing samples at centroid vs edges of distribution.
        """
        filename = f"latent_epoch_{epoch}.h5"
        filepath = self.checkpoint_dir / filename
        
        n_samples = len(latent_vectors)
        latent_dim = latent_vectors.shape[1]
        
        # Auto-generate sample_nums if not provided
        if sample_nums is None:
            sample_nums = np.arange(n_samples, dtype=np.int32)
        
        with h5py.File(filepath, 'w') as f:
            # Store latent vectors (dimensions)
            f.create_dataset('latent_vectors', data=latent_vectors, compression='gzip')
            
            # Store labels
            if labels is not None:
                f.create_dataset('labels', data=labels, compression='gzip')
            
            # Store sample numbers for tracking across epochs
            f.create_dataset('sample_nums', data=sample_nums, compression='gzip')
            
            # Store epoch as attribute and per-sample
            f.attrs['epoch'] = epoch
            f.create_dataset('epochs', data=np.full(n_samples, epoch, dtype=np.int32), compression='gzip')
            
            # Store optional images
            if images is not None:
                f.create_dataset('images', data=images, compression='gzip')
            
            # Store metadata
            f.attrs['n_samples'] = n_samples
            f.attrs['latent_dim'] = latent_dim
            f.attrs['has_images'] = images is not None
    
    def load_latent_snapshot(
        self,
        epoch: int,
        include_images: bool = False,
    ) -> dict[str, np.ndarray]:
        """
        Load latent space snapshot from HDF5.
        
        Args:
            epoch: Epoch number to load
            include_images: Whether to load images if available
            
        Returns:
            Dictionary with keys:
            - 'latent_vectors': (n_samples, latent_dim)
            - 'labels': (n_samples,) or None
            - 'sample_nums': (n_samples,)
            - 'epochs': (n_samples,) - all same value (epoch)
            - 'images': (n_samples, ...) or None if not available/requested
            - 'metadata': dict with epoch, n_samples, latent_dim, etc.
        """
        filename = f"latent_epoch_{epoch}.h5"
        filepath = self.checkpoint_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Latent snapshot not found: {filepath}")
        
        with h5py.File(filepath, 'r') as f:
            data = {
                'latent_vectors': f['latent_vectors'][:],
                'sample_nums': f['sample_nums'][:],
                'epochs': f['epochs'][:],
            }
            
            # Load labels if available
            if 'labels' in f:
                data['labels'] = f['labels'][:]
            else:
                data['labels'] = None
            
            # Load images if requested and available
            if include_images and 'images' in f:
                data['images'] = f['images'][:]
            else:
                data['images'] = None
            
            # Load metadata
            data['metadata'] = dict(f.attrs)
        
        return data
    
    def load_all_snapshots(
        self,
        epochs: Optional[list[int]] = None,
        include_images: bool = False,
    ) -> dict[int, dict[str, np.ndarray]]:
        """
        Load multiple epoch snapshots for comparison visualizations.
        
        Args:
            epochs: List of epochs to load. If None, loads all available epochs.
            include_images: Whether to load images if available
            
        Returns:
            Dictionary mapping epoch -> snapshot data (same format as load_latent_snapshot)
        """
        if epochs is None:
            # Find all available epochs
            snapshot_files = list(self.checkpoint_dir.glob("latent_epoch_*.h5"))
            epochs = []
            for file in snapshot_files:
                try:
                    epoch = int(file.stem.split('_')[-1])
                    epochs.append(epoch)
                except ValueError:
                    continue
            epochs = sorted(epochs)
        
        all_snapshots = {}
        for epoch in epochs:
            try:
                all_snapshots[epoch] = self.load_latent_snapshot(epoch, include_images=include_images)
            except FileNotFoundError:
                continue
        
        return all_snapshots
    
    def save_metrics(self, metrics: dict[str, float], epoch: int):
        """
        Save training metrics per epoch for post-training comparison.
        
        Args:
            metrics: Dictionary of metric name -> value for this epoch
            epoch: Epoch number
        """
        metrics_file = self.checkpoint_dir / "metrics.json"
        
        # Load existing metrics if file exists
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                all_metrics = json.load(f)
        else:
            all_metrics = {}
        
        # Store per-epoch metrics
        all_metrics[f"epoch_{epoch}"] = {
            'epoch': epoch,
            **metrics  # Spread the metrics
        }
        
        # Also maintain a time-series format for easy plotting
        if 'time_series' not in all_metrics:
            all_metrics['time_series'] = {}
        
        for metric_name, value in metrics.items():
            if metric_name not in all_metrics['time_series']:
                all_metrics['time_series'][metric_name] = []
            all_metrics['time_series'][metric_name].append({
                'epoch': epoch,
                'value': value
            })
        
        # Save
        with open(metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)
    
    def load_metrics(self) -> dict[str, Any]:
        """
        Load all training metrics.
        
        Returns:
            Dictionary with:
            - 'epoch_N': metrics for epoch N
            - 'time_series': dict mapping metric_name -> list of {epoch, value}
        """
        metrics_file = self.checkpoint_dir / "metrics.json"
        
        if not metrics_file.exists():
            return {}
        
        with open(metrics_file, 'r') as f:
            return json.load(f)
    
    def get_available_epochs(self) -> list[int]:
        """
        Get list of all available epoch snapshots.
        
        Returns:
            Sorted list of epoch numbers
        """
        snapshot_files = list(self.checkpoint_dir.glob("latent_epoch_*.h5"))
        epochs = []
        for file in snapshot_files:
            try:
                epoch = int(file.stem.split('_')[-1])
                epochs.append(epoch)
            except ValueError:
                continue
        return sorted(epochs)
