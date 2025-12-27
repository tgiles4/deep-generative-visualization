# Deep Generative Visualization

Visualization pipeline for demonstrating how generative models learn in 2D/3D latent spaces, designed for TikTok/YouTube content production.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Train MLP on MNIST and Generate Visualization

```bash
python train_and_visualize.py
```

This will:
1. Train an MLP with a 2D latent space on MNIST
2. Save checkpoints and latent space snapshots every epoch
3. Generate a Manim animation showing the latent space evolution

The output video will be in `media/videos/` directory.

## Project Structure

```
deep-generative-visualization/
├── models/              # Model implementations with 2D/3D latent spaces
│   ├── base.py         # Base classes with Manim hooks
│   └── mlp.py          # MLP with visualization layer
├── training/           # Training utilities
│   ├── train.py       # PyTorch Lightning module for training
│   ├── checkpoint_manager.py  # Checkpoint management
│   └── datasets.py    # Dataset loaders
├── visualization/     # Visualization system
│   ├── palette.py     # Green branding color palette
│   └── manim_scenes/  # Manim scene implementations
│       ├── base_scene.py
│       └── latent_2d.py
├── config/            # Hydra configuration files
│   ├── config.yaml    # Main config
│   ├── model/         # Model configs
│   ├── training/      # Training configs
│   └── dataset/       # Dataset configs
├── notebooks/         # Colab notebooks (coming soon)
└── train_and_visualize.py  # Main training script with Hydra
```

## Features

- **2D/3D Latent Spaces**: Models with explicit 2D or 3D bottleneck layers
- **Manim Hooks**: Real-time visualization capture during training
- **Green Branding**: Consistent color palette for TikTok/YouTube content
- **Checkpointing**: Save model states and latent snapshots for visualization
- **Flexible**: Easy to extend with new model types

## Current Implementation

- ✅ MLP with 2D/3D latent space
- ✅ MNIST dataset support with validation split
- ✅ PyTorch Lightning training pipeline
- ✅ Checkpointing with latent space snapshots
- ✅ 2D latent space visualization scene
- ✅ Green branding palette
- ✅ Hydra configuration system
- ✅ Multiple loss criteria support
- ✅ Standardized metrics system

## Coming Soon

- More model types (VAE, GAN, Diffusion, Contrastive, Meta-Learning, SentenceBERT)
- 3D latent space visualizations
- Colab notebook integration
- More datasets (Fashion-MNIST, CIFAR-10)

## Usage Example

```python
import pytorch_lightning as pl
from models.mlp import MLP
from training.train import VisualizationLightningModule, DataLoaders
from training.datasets import get_mnist_loaders

# Create model
model = MLP(input_dim=784, hidden_dims=[512, 256], output_dim=10, visualization_dim=2)

# Create data loaders
train_loader, val_loader, test_loader = get_mnist_loaders(batch_size=128, val_split=0.1)
loaders = DataLoaders(train=train_loader, val=val_loader, test=test_loader)

# Create Lightning module
lightning_module = VisualizationLightningModule(
    model=model,
    checkpoint_dir="./checkpoints",
)

# Create Lightning trainer and train
trainer = pl.Trainer(max_epochs=30)
trainer.fit(lightning_module, train_dataloaders=loaders.train, val_dataloaders=loaders.val)
trainer.test(lightning_module, dataloaders=loaders.test)
```

Or use the Hydra-based script:

```bash
# Default config
python train_and_visualize.py

# Override parameters
python train_and_visualize.py training.num_epochs=50 model.visualization_dim=3
```

## Requirements

- Python 3.12.12 (tested on Google Colab)
- PyTorch 2.0+
- PyTorch Lightning 2.0+
- Hydra Core 1.3+
- Manim Community Edition
- See `requirements.txt` for full list

## Google Colab Setup

For Colab users, see [COLAB_SETUP.md](COLAB_SETUP.md) for detailed installation instructions.
