import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random
import os
import sys
import gc
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm

from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from torch.amp import GradScaler, autocast
from src.models.models import EmbeddingModel
from src.models.backbone import Resnet
from src.models.metrics import ArcMarginProduct
from src.utils.dataset import WebFaceDataset


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


scaler = GradScaler()


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs=50,
    patience=5,
    device="cuda",
    save_dir="checkpoints",
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    epochs_no_improve = 0
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{num_epochs} [Train]",
            unit="batch",
            leave=False,
        )

        for images, labels in train_pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda"):
                logits = model(images, labels)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            train_pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{100 * train_correct / train_total:.2f}%",
                }
            )

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        epoch_train_acc = 100 * train_correct / train_total

        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0

        val_pbar = tqdm(
            val_loader,
            desc=f"Epoch {epoch+1}/{num_epochs} [Val]",
            unit="batch",
            leave=False,
        )

        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                with autocast(device_type="cuda"):
                    logits = model(images, labels)
                    loss = criterion(logits, labels)

                running_val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc = 100 * val_correct / val_total

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"| Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.2f}% "
            f"| Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.2f}%"
        )

        print(f"VRAM Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(
                model.state_dict(), os.path.join(save_dir, "best_arcface_model.pth")
            )
            epochs_no_improve = 0
            print("--> Model improved & saved!")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early Stopping triggered after {patience} epochs.")
                break

        gc.collect()
        torch.cuda.empty_cache()

    return history


def save(history, save_dir):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    plt.plot(epochs, history["val_loss"], "r--", label="Val Loss")

    plt.title("ArcFace Training & Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (CrossEntropy)")
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(save_dir, "loss_chart.png")
    plt.savefig(plot_path)
    plt.show()


if __name__ == "__main__":

    gc.collect()
    torch.cuda.empty_cache()

    root_path = os.path.abspath(os.path.join(os.getcwd()))
    if root_path not in sys.path:
        sys.path.append(root_path)

    init()

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
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.3
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
        batch_size=52,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8,
    )

    val_dataset = WebFaceDataset(val_df, DATA_DIR, transform=test_transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=52,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8,
    )

    # test_dataset = WebFaceDataset(test_df, DATA_DIR, transform=test_transform)
    # test_loader = DataLoader(
    #     test_dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True
    # )

    num_classes = 1780

    model = EmbeddingModel(
        backbone=Resnet("resnet50", image_size=112, out_channels=512),
        metric=ArcMarginProduct(512, num_classes, s=64, m=0.3),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": 1e-4},
            {
                "params": model.metric.parameters(),
                "lr": 1e-4,
            },
        ],
        weight_decay=1e-4,
    )

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=100,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_dir=os.path.join(CHECKPOINTS, "FaceEmbedding", "weights"),
    )

    save(
        history=history, save_dir=os.path.join(CHECKPOINTS, "FaceEmbedding", "results")
    )
