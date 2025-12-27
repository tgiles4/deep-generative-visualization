"""
Simple script to train MLP on MNIST and generate visualization.
Run this to test the full pipeline.
Uses Hydra for configuration management.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig, OmegaConf
from training.train import VisualizationLightningModule, DataLoaders
from training.datasets import get_mnist_loaders


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    """Main function to train and visualize."""
    print("=" * 60)
    print("Training Model with Visualization")
    print("=" * 60)
    print(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    print()
    
    # Instantiate model from config
    print("Creating model...")
    model = hydra.utils.instantiate(cfg.model)
    print(f"Model: {cfg.model._target_}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Create data loaders
    print("Loading data...")
    train_loader, val_loader, test_loader = get_mnist_loaders(
        batch_size=cfg.training.batch_size,
        val_split=cfg.training.val_split,
        data_dir=cfg.dataset.get('data_dir', './data') if 'dataset' in cfg else './data',
    )
    loaders = DataLoaders(train=train_loader, val=val_loader, test=test_loader)
    print()
    
    # Setup criterion (optional in config)
    criterion = None
    if cfg.training.get('criterion') is not None:
        criterion = hydra.utils.instantiate(cfg.training.criterion)
    else:
        criterion = nn.CrossEntropyLoss()  # Default
    
    # Setup optimizer (optional in config)
    optimizer = optim.Adam
    optimizer_kwargs = {'lr': cfg.training.learning_rate}
    if cfg.training.get('optimizer') is not None:
        optimizer = hydra.utils.get_class(cfg.training.optimizer._target_)
        if cfg.training.optimizer.get('kwargs'):
            optimizer_kwargs.update(cfg.training.optimizer.kwargs)
    
    # Create Lightning module
    print("Setting up training...")
    lightning_module = VisualizationLightningModule(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        checkpoint_dir=cfg.checkpointing.checkpoint_dir,
        save_every_n_epochs=cfg.checkpointing.save_every_n_epochs,
        save_images=cfg.checkpointing.save_images,
        learning_rate=cfg.training.learning_rate,
    )
    
    # Create Lightning trainer
    trainer_kwargs = {
        'max_epochs': cfg.training.num_epochs,
        'accelerator': cfg.trainer.accelerator,
        'devices': cfg.trainer.devices,
        'enable_progress_bar': cfg.trainer.enable_progress_bar,
    }
    # Add optional trainer kwargs
    if cfg.trainer.get('precision'):
        trainer_kwargs['precision'] = cfg.trainer.precision
    
    trainer = pl.Trainer(**trainer_kwargs)
    
    # Train model
    print("Starting training...")
    trainer.fit(
        lightning_module,
        train_dataloaders=loaders.train,
        val_dataloaders=loaders.val,
    )
    
    # Test
    print("Running test...")
    trainer.test(lightning_module, dataloaders=loaders.test)
    print()
    
    print("=" * 60)
    print("Training complete! Checkpoints saved to:", cfg.checkpointing.checkpoint_dir)
    print("=" * 60)
    
    # Generate visualization
    print("=" * 60)
    print("Generating Manim visualization...")
    print("=" * 60)
    print()
    print("To render the visualization, run Manim from the command line:")
    print()
    print("  manim -ql visualization/manim_scenes/latent_2d.py Latent2DScene")
    print()
    print("Or use the helper function in a Python script:")
    print()
    print("  from visualization.manim_scenes.latent_2d import create_latent_2d_scene_from_checkpoints")
    print(f"  scene = create_latent_2d_scene_from_checkpoints('{cfg.checkpointing.checkpoint_dir}')")
    print("  scene.render()")
    print()


if __name__ == "__main__":
    main()

