import os
import sys
import gc

import pandas as pd
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.models import EmbeddingModel
from src.models.backbone import Resnet
from src.models.metrics import ArcMarginProduct
from src.utils.dataset import WebFaceDataset
from src.trainning.trainer import Trainer

NUM_CLASSES = 1780


def init(seed=42, benchmark=False) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        print("device: GPU")
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = benchmark


if __name__ == "__main__":

    gc.collect()
    torch.cuda.empty_cache()

    root_path = os.path.abspath(os.path.join(os.getcwd()))
    if root_path not in sys.path:
        sys.path.append(root_path)

    init(benchmark=True)

    DATA_DIR = os.path.join(root_path, "data", "processed", "webface_112x112")
    METADATA_DIR = os.path.join(DATA_DIR, "metadata.xlsx")
    CHECKPOINTS = os.path.join(root_path, "checkpoints")

    metadata = pd.read_excel(METADATA_DIR)
    le = LabelEncoder()

    train_df = metadata[metadata["split"] == "train"].reset_index(drop=True)
    val_df = metadata[metadata["split"] == "val"].reset_index(drop=True)
    # test_df = metadata[metadata["split"] == "test"].reset_index(drop=True)

    train_df["label"] = le.fit_transform(train_df["label"].astype(str))
    val_df["label"] = le.transform(val_df["label"].astype(str))

    num_classes = len(le.classes_)

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            #transforms.RandomRotation(degrees=15),
            transforms.RandomResizedCrop(size=112, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)],
                p=0.2,
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))], p=0.2
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    train_dataset = WebFaceDataset(train_df, DATA_DIR, transform=train_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    val_dataset = WebFaceDataset(val_df, DATA_DIR, transform=test_transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # test_dataset = WebFaceDataset(test_df, DATA_DIR, transform=test_transform)
    # test_loader = DataLoader(
    #     test_dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True
    # )

    model = EmbeddingModel(
        backbone=Resnet(
            "resnet18", image_size=112, out_channels=512, weights="IMAGENET1K_V1"
        ),
        metric=ArcMarginProduct(512, NUM_CLASSES, s=64, m=0.25),
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": 1e-4},
            {
                "params": model.metric.parameters(),
                "lr": 1e-3,
            },
        ],
        weight_decay=5e-2,
    )

    scheduler = ReduceLROnPlateau(
        optimizer=optimizer, mode="min", patience=5, threshold=1e-2
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        checkpoints_dir="checkpoints/recognition",
        logs_dir="logs/recognition",
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        epochs=200,
        patience=8,
    )
