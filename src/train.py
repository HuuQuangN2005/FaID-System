import os
import sys
import torch
import random
import numpy as np

import pandas as pd
from tqdm import tqdm
import time

from src.models.models import FaceRecogntionModel
from src.utils.utils import vggface2_test_transform, vggface2_train_transform
from src.utils.dataloader import create_dataloader
from torchsummary import summary

def set_seed(seed: int = 42):
    random.seed(seed)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


VGGFACE2_DATASET = os.path.join(os.getcwd(), "data", "VGGFACE2_split")
CHECKPOINTS = os.path.join(os.getcwd(), "checkpoints", "HybridModel")


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(dataloader, desc="Training", unit="batch", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.2f}%")

    return running_loss / len(dataloader), 100.0 * correct / total


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in tqdm(
            dataloader, desc="Validating", unit="batch", leave=False
        ):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(dataloader), 100.0 * correct / total


def run_training(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    epochs=50,
    save_path="checkpoints",
    device="cuda",
):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    device = torch.device(device=device)
    model.to(device)

    history = []
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        start_t = time.time()

        t_loss, t_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        v_loss, v_acc = validate(model, val_loader, criterion, device)

        dt = time.time() - start_t

        log_entry = {
            "epoch": epoch,
            "train_loss": round(t_loss, 4),
            "train_acc": round(t_acc, 2),
            "val_loss": round(v_loss, 4),
            "val_acc": round(v_acc, 2),
            "time": round(dt, 2),
        }
        history.append(log_entry)
        pd.DataFrame(history).to_csv(os.path.join(save_path, "logs.csv"), index=False)

        print(
            f"Ep {epoch:02d} | L:{t_loss:.3f}/{v_loss:.3f} | Acc:{t_acc:.1f}%/{v_acc:.1f}% | {dt:.1f}s"
        )

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), os.path.join(save_path, "best.pth"))
            print(f"--> Saved best model with Acc: {v_acc:.2f}%")

    torch.save(model.state_dict(), os.path.join(save_path, "last.pth"))


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(42)

    train_dataloader, val_dataloader, test_dataloader, num_classes = create_dataloader(
        VGGFACE2_DATASET,
        vggface2_train_transform,
        vggface2_test_transform,
        vggface2_test_transform,
        batch_size=128,
    )

    model = FaceRecogntionModel(num_classes=num_classes, device=device)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)

    summary(model, (3, 112, 112))
    print('-----------------------------------------------------------------')
    
    run_training(
        model=model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        save_path=CHECKPOINTS,
    )

