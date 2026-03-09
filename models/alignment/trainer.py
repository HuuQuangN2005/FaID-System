import os
import time
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import CelebALandmarkDataset
from model import LandmarkModel

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

IMG_DIR = os.path.join(PROJECT_ROOT, "data", "alignment", "raw", "img_align_celeba")
SPLITS_DIR = os.path.join(PROJECT_ROOT, "data", "alignment", "splits")
SAVE_DIR = os.path.join(PROJECT_ROOT, "models", "alignment", "weights")
os.makedirs(SAVE_DIR, exist_ok=True)

assert os.path.exists(IMG_DIR), f"Image folder not found: {IMG_DIR}"
assert os.path.exists(SPLITS_DIR), f"Splits folder not found: {SPLITS_DIR}"
#

BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE_EARLY_STOP = 5

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def compute_nme(preds, targets):

    left_eye = targets[:, 0, :]
    right_eye = targets[:, 1, :]
    ocular_dist = torch.norm(left_eye - right_eye, dim=1)
    ocular_dist = torch.clamp(ocular_dist, min=1e-6)

    diff = preds - targets
    dist = torch.norm(diff, dim=2)
    mean_dist = dist.mean(dim=1)

    nme = mean_dist / ocular_dist
    return nme.mean().item()

def train():

    print(f"Using device: {DEVICE}")
    # print("Current working directory:", os.getcwd())

    train_split = os.path.join(SPLITS_DIR, "train.txt")
    val_split = os.path.join(SPLITS_DIR, "val.txt")

    # if not os.path.exists(train_split) or not os.path.exists(val_split):
    #     exit(1)

    train_dataset = CelebALandmarkDataset(img_dir=IMG_DIR, split_file=train_split)
    val_dataset = CelebALandmarkDataset(img_dir=IMG_DIR, split_file=val_split)

    pin_memory = torch.cuda.is_available()
    num_workers = 4 if pin_memory else 0

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)

    model = LandmarkModel().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    best_epoch = 0
    no_improve_epochs = 0

    for epoch in range(EPOCHS):
        start_time = time.time()

        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for images, targets in train_pbar:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            nme = compute_nme(preds, targets)
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}", 'nme': f"{nme:.4f}"})

        train_loss /= len(train_loader.dataset)

        # val
        model.eval()
        val_loss = 0.0
        val_nme = 0.0
        val_batches = 0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", leave=False):
                images, targets = images.to(DEVICE), targets.to(DEVICE)
                preds = model(images)
                loss = criterion(preds, targets)
                val_loss += loss.item() * images.size(0)

                nme = compute_nme(preds, targets)
                val_nme += nme * images.size(0)
                val_batches += images.size(0)

        val_loss /= len(val_loader.dataset)
        val_nme /= val_batches

        scheduler.step(val_loss)

        epoch_time = time.time() - start_time
        print("----------------------------------------")
        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
              f"Val NME: {val_nme:.6f} | LR: {optimizer.param_groups[0]['lr']:.6f} | "
              f"Time: {epoch_time:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
            print(f"### Saved best at {best_epoch} (Val Loss: {val_loss:.6f}, NME: {val_nme:.6f})")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= PATIENCE_EARLY_STOP: # 5
            print(f"No improvement")
            break

    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "final_model.pth"))

if __name__ == "__main__":
    # print(torch.cuda.is_available())
    # print(torch.cuda.get_device_name(0))
    train()