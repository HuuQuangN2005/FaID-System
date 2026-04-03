import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    def __init__(
        self, embedding_size=512, num_classes=1000, dropout=0.4, device="cuda"
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(embedding_size, num_classes, bias=False, device=device)
        nn.init.normal_(self.fc.weight, std=0.01)

    def forward(self, embeddings, labels=None):
        x = self.dropout(embeddings)
        logits = self.fc(x)
        return logits
