import numpy as np
from .MyDataset import MyDataset
from torch.utils.data import DataLoader, random_split
from logger import logging

def get_dataloader(
        train_dir,
        val_dir,
        train_csv,
        val_csv,
        class_mapping,
        train_transform=None,
        val_transform=None,
        split=(0.6, 0.4),
        batch_size=32,
        *args, **kwargs):
    """
    This function returns the train, val, and test dataloaders, using specified mappings for class labels.

    :param train_dir: Directory where training images are stored.
    :param val_dir: Directory where validation images are stored.
    :param train_csv: Path to the CSV file for training data.
    :param val_csv: Path to the CSV file for validation data.
    :param class_mapping: Dictionary mapping class labels to numeric values.
    :param train_transform: Optional transformations to apply to training images.
    :param val_transform: Optional transformations to apply to validation images.
    :param split: Tuple indicating how to split validation data into val and test sets.
    :param batch_size: Number of samples in each batch.
    """
    # Create the datasets using the provided CSV files and class mapping
    train_ds = MyDataset(csv_file=train_csv, img_dir=train_dir, class_mapping=class_mapping, transform=train_transform)
    val_ds = MyDataset(csv_file=val_csv, img_dir=val_dir, class_mapping=class_mapping, transform=val_transform)

    # Now we want to split the val_ds into validation and test datasets
    lengths = np.array(split) * len(val_ds)
    lengths = lengths.astype(int)
    left = len(val_ds) - lengths.sum()
    # We need to add the difference due to float approximation to int
    lengths[-1] += left

    val_ds, test_ds = random_split(val_ds, lengths.tolist())
    logging.info(f'Train samples={len(train_ds)}, Validation samples={len(val_ds)}, Test samples={len(test_ds)}')

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, *args, **kwargs)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, *args, **kwargs)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, *args, **kwargs)

    return train_dl, val_dl, test_dl
