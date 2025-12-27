"""
Dataset loaders and preprocessing utilities.
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset


class IndexedDataset(Dataset):
    """
    Wrapper dataset that adds global indices to samples.
    Returns (x, y, idx) where idx is the global MNIST index.
    """
    def __init__(self, base_dataset, indices=None):
        """
        Args:
            base_dataset: Base dataset (e.g., MNIST or Subset)
            indices: Optional list of indices to use. If None, uses all indices.
                    For Subset, this should be the subset's indices.
        """
        self.base_dataset = base_dataset
        # If base_dataset is a Subset, get the original indices
        if hasattr(base_dataset, 'indices'):
            # It's a Subset from random_split
            # Convert to list if it's a tensor
            idx = base_dataset.indices
            if isinstance(idx, torch.Tensor):
                self.global_indices = idx.tolist()
            else:
                self.global_indices = list(idx)
        elif indices is not None:
            self.global_indices = list(indices)
        else:
            # Full dataset - indices are just 0..len-1
            self.global_indices = list(range(len(base_dataset)))
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        x, y = self.base_dataset[idx]
        global_idx = self.global_indices[idx]
        return x, y, global_idx


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
        train_subset, val_subset = random_split(
            full_train_dataset,
            [train_size, val_size],
            generator=generator
        )
        
        # Wrap with IndexedDataset to preserve global indices
        train_dataset = IndexedDataset(train_subset)
        val_dataset = IndexedDataset(val_subset)
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,  # Don't shuffle validation
            num_workers=2,
            pin_memory=True,
        )
    else:
        train_dataset = IndexedDataset(full_train_dataset)
        val_loader = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    
    # Wrap test dataset with IndexedDataset
    # For test set, global indices start from 60000 (after training set)
    test_dataset_wrapped = IndexedDataset(test_dataset, indices=list(range(60000, 60000 + len(test_dataset))))
    test_loader = DataLoader(
        test_dataset_wrapped,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader

