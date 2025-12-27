"""
Multi-layer Perceptron with visualization layer.
Architecture: Input -> Hidden -> Visualization -> Output
"""

import torch
import torch.nn as nn
from models.base import BaseLatentModel


class MLP(BaseLatentModel):
    """
    MLP with visualization layer in the middle.
    Architecture: Input -> Hidden -> Visualization Layer -> Output
    
    This allows observing how the reconstruction/output layers affect
    the learned representations in the visualization space.
    """
    
    def __init__(
        self,
        input_dim: int = 784,  # 28x28 for MNIST
        hidden_dims: list[int] = [512, 256],
        output_dim: int = 10,  # num_classes for classification
        visualization_dim: int = 2,  # 2D or 3D for visualization
        activation: type | str = "ReLU",  # String for Hydra config compatibility
    ):
        """
        Initialize MLP with visualization layer.
        
        Args:
            input_dim: Input dimension (784 for flattened MNIST)
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (e.g., num_classes for classification)
            visualization_dim: Dimension of visualization layer (2 or 3)
            activation: Activation function class or string name (e.g., "ReLU", "GELU")
                       String format is Hydra-config friendly
        """
        super().__init__(latent_dim=visualization_dim)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Resolve activation (handles both string and type)
        activation_cls = self.get_activation(activation)
        
        # Build layers: Input -> Hidden -> Visualization -> Output
        # Split into two parts to easily extract visualization layer
        # Part 1: Input -> Hidden -> Visualization
        pre_dims = [input_dim] + hidden_dims + [visualization_dim]
        self.pre_visualization = self.build_sequential_layers(
            dims=pre_dims,
            activation=activation_cls,
        )
        
        # Part 2: Visualization -> Output (direct connection, no decoder)
        self.visualization_to_output = self.build_sequential_layers(
            dims=[visualization_dim, output_dim],
            activation=None,  # No activation before final output (standard for classification)
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to visualization space.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Visualization representation of shape (batch_size, visualization_dim)
        """
        # Flatten if needed (for image inputs)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.pre_visualization(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Input -> Hidden -> Visualization -> Output.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Flatten if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        # Input -> Hidden -> Visualization -> Output
        z = self.pre_visualization(x)
        return self.visualization_to_output(z)
