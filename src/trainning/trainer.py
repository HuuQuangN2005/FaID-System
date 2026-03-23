import torch
import gc

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast
from pathlib import Path


class Trainer:
    def __init__(
        self, model, optimizer, criterion, checkpoints_dir, logs_dir, *args, **kwargs
    ):
        super(Trainer, self).__init__(*args, **kwargs)
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        self.checkpoints = Path(checkpoints_dir)
        self.logs = Path(logs_dir)
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

        self.checkpoints.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self.scaler = GradScaler() if self.device == "cuda" else None
        self.writer = SummaryWriter(str(self.logs))

    def clean(self):
        if self.writer:
            self.writer.close()

        self.model = None
        self.optimizer = None

        gc.collect()
        torch.cuda.empty_cache()

    def _calculate_acc(self, outputs, labels):
        _, predicted = torch.max(outputs.data, 1)
        batch_total = labels.size(0)
        batch_correct = (predicted == labels).sum().item()
        return predicted, batch_total, batch_correct

    def _save_learning_rate(self, epoch):
        lr_backbone = self.optimizer.param_groups[0]["lr"]
        lr_metric = self.optimizer.param_groups[1]["lr"]

        self.writer.add_scalar("LR/Backbone", lr_backbone, epoch)
        self.writer.add_scalar("LR/Metric", lr_metric, epoch)

    def _run_epoch(self, loader, epoch, phase="train"):
        self.model.train() if phase == "train" else self.model.eval()
        running_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(
            loader, desc=f"Epoch {epoch} [{phase.upper()}]", unit="batch", leave=False
        )

        with torch.set_grad_enabled(phase == "train"):
            for images, labels in pbar:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                if phase == "train":
                    self.optimizer.zero_grad(set_to_none=True)

                device_type = "cuda" if "cuda" in str(self.device) else "cpu"
                with autocast(device_type=device_type):
                    try:
                        outputs = self.model(images, labels)
                    except TypeError:
                        outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                if phase == "train" and self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                elif phase == "train":
                    loss.backward()
                    self.optimizer.step()

                running_loss += loss.item() * images.size(0)
                pred, b_total, b_correct = self._calculate_acc(outputs, labels)
                total += b_total
                correct += b_correct

                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "acc": f"{100*correct/total:.2f}%"}
                )

        return running_loss / len(loader.dataset), 100 * correct / total

    def fit(
        self,
        train_loader,
        val_loader,
        scheduler=None,
        epochs=100,
        patience=8,
        start_epoch=1,
    ):
        best_val_loss = float("inf")
        epochs_no_improve = 0

        if start_epoch == 1:
            self.history = {
                "train_loss": [],
                "val_loss": [],
                "train_acc": [],
                "val_acc": [],
            }

        for epoch in range(start_epoch, epochs + 1):
            t_loss, t_acc = self._run_epoch(train_loader, epoch, "train")
            v_loss, v_acc = self._run_epoch(val_loader, epoch, "val")

            for k, v in zip(
                ["train_loss", "val_loss", "train_acc", "val_acc"],
                [t_loss, v_loss, t_acc, v_acc],
            ):
                self.history[k].append(v)

            self.writer.add_scalars("Loss", {"train": t_loss, "val": v_loss}, epoch)
            self.writer.add_scalars("Accuracy", {"train": t_acc, "val": v_acc}, epoch)

            print(
                f"Epoch {epoch} | Train Loss: {t_loss:.4f}, Acc: {t_acc:.2f}% | Val Loss: {v_loss:.4f}, Acc: {v_acc:.2f}%"
            )

            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(v_loss)
                else:
                    scheduler.step()

                self._save_learning_rate(epoch=epoch)

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                torch.save(self.model.state_dict(), self.checkpoints / "best_model.pth")
                epochs_no_improve = 0
                print("--> Improved & Saved!")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early Stopping at epoch {epoch}")
                    break

            gc.collect()
            torch.cuda.empty_cache()

        self.clean()
        return self.history
