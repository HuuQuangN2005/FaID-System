import torch
import torch.nn as nn
import torchvision.models as models
import math
import torch.nn.functional as F


class Resnet(nn.Module):
    def __init__(self, name, image_size=224, out_channels=512, weights=None):
        super(Resnet, self).__init__()

        self.out_channels = out_channels

        if name == "resnet18":
            self.backbone = models.resnet18(weights=weights)
        elif name == "resnet34":
            self.backbone = models.resnet34(weights=weights)
        elif name == "resnet50":
            self.backbone = models.resnet50(weights=weights)
        else:
            raise ValueError(f"Model not supported")

        if image_size == 112:
            self.backbone.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.backbone.maxpool = nn.Identity()

        elif image_size > 224:
            raise ValueError(f"Model not supported")

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, self.out_channels)

    def forward(self, x):
        return self.backbone(x)
