# Hydra Configuration Notes

## Activation Functions

The `build_sequential_layers` method and model classes support activation functions in two ways:

### 1. String Format (Hydra-Friendly)
```yaml
activation: "ReLU"
activation: "GELU"
activation: "Tanh"
```

This is the recommended format for Hydra configs as it's YAML-native and easy to modify.

### 2. Class Format (Programmatic)
```python
activation: nn.ReLU
activation: nn.GELU
```

This works in Python code but requires importing torch.nn in configs.

## Supported Activation Functions

Any activation function available in `torch.nn` can be used:
- `"ReLU"` - Rectified Linear Unit (default)
- `"GELU"` - Gaussian Error Linear Unit (smoother, often better representations)
- `"Tanh"` - Hyperbolic Tangent (bounded)
- `"Sigmoid"` - Sigmoid (bounded, 0-1)
- `"LeakyReLU"` - Leaky ReLU (allows small negative values)
- `"ELU"` - Exponential Linear Unit
- `"Swish"` - Swish/SiLU

## Example Hydra Config

```yaml
# config/model/mlp.yaml
_target_: models.mlp.MLP
input_dim: 784
hidden_dims: [512, 256]
output_dim: 10
visualization_dim: 2
activation: "GELU"  # Easy to change in config!
```

## Why String Format?

1. **YAML Native**: No need to import Python modules in config files
2. **Easy Experimentation**: Change activation with a single line
3. **Hydra Override Friendly**: `python train.py model.activation=GELU`
4. **Version Control**: Clear diffs when changing activations

## Impact on Representations

Different activations can significantly affect learned representations:
- **ReLU**: Sparse activations, dead neurons possible
- **GELU**: Smoother gradients, often better for representation learning
- **Tanh/Sigmoid**: Bounded outputs, can help with stability
- **LeakyReLU**: Prevents dead neurons, maintains gradient flow

Experimenting with activations via Hydra configs makes it easy to see how they affect the visualization space!

