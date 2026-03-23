import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingModel(nn.Module):
    def __init__(self, backbone, metric):
        super(EmbeddingModel, self).__init__()
        self.backbone = backbone
        self.metric = metric

    def forward(self, x, y=None):
        embs = self.backbone(x)
        embs = F.normalize(embs, p=2, dim=1)

        if y is not None:
            logits = self.metric(embs, y)
            return logits

        return embs
