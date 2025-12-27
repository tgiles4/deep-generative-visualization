"""
Base classes for models with 2D/3D latent spaces and Manim visualization hooks.
"""

from abc import ABC, abstractmethod
from typing import Callable, Any, Optional
import numpy as np
import torch
import torch.nn as nn


class ManimHook:
    """Callback hook for Manim visualization system."""
    
    def __init__(self):
        self.callbacks: list[Callable] = []
    
    def register(self, callback: Callable):
        """Register a callback function."""
        self.callbacks.append(callback)
    
    def unregister(self, callback: Callable):
        """Unregister a callback function."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def call(self, *args, **kwargs):
        """Call all registered callbacks."""
        for callback in self.callbacks:
            callback(*args, **kwargs)


class BaseLatentModel(nn.Module, ABC):
    """
    Abstract base class for models with 2D/3D latent spaces.
    Provides Manim hooks for visualization.
    """
    
    def __init__(self, latent_dim: int = 2):
        """
        Initialize base model.
        
        Args:
            latent_dim: Dimension of latent space (2 or 3 for visualization)
        """
        super().__init__()
        self.latent_dim = latent_dim
        
        # Manim hooks
        self.hooks = {
            'on_epoch_start': ManimHook(),
            'on_epoch_end': ManimHook(),
            'on_batch_end': ManimHook(),
        }
    
    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent space.
        
        Args:
            x: Input tensor
            
        Returns:
            Latent representation tensor of shape (batch_size, latent_dim)
        """
        pass
    
    def register_manim_hook(self, hook_name: str, callback: Callable):
        """
        Register a Manim visualization callback.
        
        Args:
            hook_name: Name of hook ('on_epoch_start', 'on_epoch_end', 'on_batch_end')
            callback: Callback function
        """
        if hook_name in self.hooks:
            self.hooks[hook_name].register(callback)
        else:
            raise ValueError(f"Unknown hook name: {hook_name}")
    
    def _trigger_hook(self, hook_name: str, *args, **kwargs):
        """Trigger a hook with given arguments."""
        if hook_name in self.hooks:
            self.hooks[hook_name].call(*args, **kwargs)
    
    @staticmethod
    def build_sequential_layers(
        dims: list[int] | int,
        activation: type | str | None = nn.ReLU,
        include_final_activation: bool = False,
    ) -> nn.Sequential:
        """
        Build a sequential layer stack from a list of dimensions.
        General-purpose utility for building encoder/decoder/any feedforward layers.
        
        Args:
            dims: List of dimensions [input_dim, hidden1, hidden2, ..., output_dim]
                  or single int (for single layer)
            activation: Activation function class, string name, or None
                       If string, will be resolved from torch.nn (e.g., "ReLU", "GELU")
                       If None, no activation is added
            include_final_activation: Whether to add activation after final layer
            
        Returns:
            nn.Sequential module with Linear layers and activations
            
        Examples:
            # Encoder: 784 -> 512 -> 256 -> 2
            encoder = build_sequential_layers([784, 512, 256, 2], activation=nn.ReLU)
            
            # Decoder: 2 -> 256 -> 512 -> 784 (symmetric)
            decoder = build_sequential_layers([2, 256, 512, 784], activation=nn.ReLU)
            
            # With string activation (Hydra-friendly)
            layers = build_sequential_layers([784, 512, 10], activation="GELU")
        """
        # Handle single int
        if isinstance(dims, int):
            dims = [dims]
        
        if len(dims) < 2:
            raise ValueError("dims must have at least 2 elements [input_dim, output_dim]")
        
        # Resolve activation from string if needed
        if isinstance(activation, str):
            activation = getattr(nn, activation)
        
        layers = []
        for i in range(len(dims) - 1):
            # Add linear layer
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            
            # Add activation (except after final layer unless specified)
            if activation is not None:
                if i < len(dims) - 2 or include_final_activation:
                    layers.append(activation())
        
        return nn.Sequential(*layers)
    
    @staticmethod
    def get_activation(activation: str | type) -> type:
        """
        Resolve activation function from string or type.
        Useful for Hydra configs where activation is specified as string.
        
        Args:
            activation: String name (e.g., "ReLU", "GELU") or activation class
            
        Returns:
            Activation function class
            
        Examples:
            act = get_activation("ReLU")  # Returns nn.ReLU
            act = get_activation(nn.GELU)  # Returns nn.GELU
        """
        if isinstance(activation, str):
            return getattr(nn, activation)
        return activation
    
    def get_latent_vectors(self, dataloader, device: str = 'cpu') -> tuple:
        """
        Get latent vectors for all samples in dataloader.
        
        Args:
            dataloader: DataLoader with input data
            device: Device to run inference on
            
        Returns:
            Tuple of (latent_vectors, labels) where:
            - latent_vectors: numpy array of shape (n_samples, latent_dim)
            - labels: numpy array of shape (n_samples,)
        """
        self.eval()
        latents = []
        labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x, y = batch[0], batch[1]
                else:
                    x, y = batch, None
                
                x = x.to(device)
                z = self.encode(x)
                
                latents.append(z.cpu().numpy())
                if y is not None:
                    labels.append(y.cpu().numpy())
        
        latents = np.concatenate(latents, axis=0)
        if labels:
            labels = np.concatenate(labels, axis=0)
        else:
            labels = None
        
        return latents, labels
