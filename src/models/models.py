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

        if y is not None:
            logits = self.metric(embs, y)
            return logits

        return F.normalize(embs)
