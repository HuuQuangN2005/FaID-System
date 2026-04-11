import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch
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

class LandmarkModel(nn.Module):

    def __init__(self):
        super().__init__()

        backbone = models.resnet18(weights=None)

        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.fc = nn.Linear(512, 10)

    def forward(self, x):

        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x.view(-1,5,2)

