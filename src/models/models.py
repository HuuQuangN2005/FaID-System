import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.backbone import HybridNet
from src.models.head import ClassificationHead


class FaceRecogntionModel(nn.Module):
    def __init__(
        self, embedding_size=512, num_classes=1000, dropout=0.3, device="cuda"
    ):
        super(FaceRecogntionModel, self).__init__()
        self.backbone = HybridNet(embedding_size=embedding_size)

        self.head = ClassificationHead(
            embedding_size=embedding_size,
            num_classes=num_classes,
            dropout=dropout,
            device=device,
        )

    def forward(self, x):
        embs = self.backbone(x)
        logits = self.head(embs)
        return logits
