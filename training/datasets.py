"""
Dataset loaders and preprocessing utilities.
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def get_mnist_loaders(
    batch_size: int = 128,
    data_dir: str = './data',
    download: bool = True,
    val_split: float = 0.1,
    random_seed: int = 42,
) -> tuple:
    """
    Get MNIST data loaders.
    
    MNIST has 60k training and 10k test samples. No official validation set.
    This function optionally splits the training set to create a validation set
    for monitoring generalization during training.
    
    Args:
        batch_size: Batch size for data loaders
        data_dir: Directory to store/load data
        download: Whether to download if not present
        val_split: Fraction of training data to use for validation (0.0 to disable)
        random_seed: Random seed for reproducible train/val split
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
        If val_split=0.0, val_loader will be None
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Normalize to [0, 1] (MNIST is already normalized)
    ])
    
    # Full training dataset (60k samples)
    full_train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=download,
        transform=transform,
    )
    
    # Test dataset (10k samples)
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=download,
        transform=transform,
    )
    
    # Split training set into train/val if requested
    if val_split > 0.0:
        val_size = int(len(full_train_dataset) * val_split)
        train_size = len(full_train_dataset) - val_size
        
        # Use random seed for reproducibility
        generator = torch.Generator().manual_seed(random_seed)
        train_dataset, val_dataset = random_split(
            full_train_dataset,
            [train_size, val_size],
            generator=generator
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,  # Don't shuffle validation
            num_workers=2,
            pin_memory=True,
        )
    else:
        train_dataset = full_train_dataset
        val_loader = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader

