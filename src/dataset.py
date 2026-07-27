from torch.utils.data import random_split, DataLoader
from torchvision.datasets import ImageFolder

from config import DATASET_PATH, BATCH_SIZE
from transforms import train_transforms, test_transforms

def get_dataloaders(train_ratio=0.8):

    full_dataset = ImageFolder(DATASET_PATH)

    train_size = (train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size]
    )

    train_dataset.dataset.transform = train_transforms
    val_dataset.dataset.transform = test_transforms

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    return train_loader, val_loader