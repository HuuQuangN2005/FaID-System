from torchvision import datasets
from torch.utils.data import DataLoader


def create_dataloader(
    data_dir: str,
    train_transform: str,
    val_transform: str,
    test_transform: str,
    batch_size: int = 128,
    num_workers: int = 4,
):
    train_dataset = datasets.ImageFolder(
        root=f"{data_dir}/train", transform=train_transform
    )
    val_dataset = datasets.ImageFolder(root=f"{data_dir}/val", transform=val_transform)
    test_dataset = datasets.ImageFolder(
        root=f"{data_dir}/test", transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers // 2,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, len(train_dataset.classes)
