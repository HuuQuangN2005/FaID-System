import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from src.models.blocks import *


class SimpleDetectionModel(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.1),
            )

        self.model = nn.Sequential(
            block(3, 32),
            block(32, 32),
            nn.MaxPool2d(2),
            block(32, 64),
            block(64, 64),
            nn.MaxPool2d(2),
            block(64, 128),
            block(128, 128),
            nn.MaxPool2d(2),
            block(128, 256),
            block(256, 256),
            nn.MaxPool2d(2),
            block(256, 512),
            block(512, 512),
            nn.MaxPool2d(2),
            block(512, 256),
        )
        self.head = nn.Conv2d(256, 5, kernel_size=1)

    def forward(self, x):
        x = self.model(x)
        x = self.head(x)
        x = x.permute(0, 2, 3, 1)

        boxes = torch.sigmoid(x[..., :4])
        conf = x[..., 4:5]

        return torch.cat([boxes, conf], dim=-1)


class DetectionModel(nn.Module):
    def __init__(self, configs: list = None):
        super(DetectionModel, self).__init__()

        if configs is None:
            configs = [
                (3, 32, 3, 1, 1),
                (32, 64, 3, 1, 1),
                (64, 128, 3, 1, 1),
                (128, 256, 3, 1, 1),
                (256, 512, 3, 1, 1),
            ]
        elif not isinstance(configs, list):
            raise ValueError("configs error!!")

        layers = []
        for i, (in_c, out_c, kernel, stride, padding) in enumerate(configs):
            layers.append(
                ConvBlock(
                    in_c=in_c,
                    out_c=out_c,
                    kernel=kernel,
                    stride=stride,
                    padding=padding,
                )
            )
            layers.append(
                ConvBlock(
                    in_c=out_c,
                    out_c=out_c,
                    kernel=kernel,
                    stride=stride,
                    padding=padding,
                )
            )
            layers.append(nn.MaxPool2d(2))

        self.backbone = nn.Sequential(*layers)

        layers = []

        if configs[-1][1] > 512:
            layers.append(
                ConvBlock(in_c=configs[-1][1], out_c=512, kernel=3, stride=1, padding=1)
            )
            layers.append(ConvBlock(in_c=512, out_c=256, kernel=3, stride=1, padding=1))
            layers.append(nn.Conv2d(256, 5, kernel_size=1))

        else:
            layers.append(
                ConvBlock(in_c=configs[-1][1], out_c=256, kernel=3, stride=1, padding=1)
            )
            layers.append(nn.Conv2d(256, 5, kernel_size=1))

        self.head = nn.Sequential(*layers)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        x = x.permute(0, 2, 3, 1)
        boxes = torch.sigmoid(x[..., :4])
        conf = x[..., 4:5]

        return torch.cat([boxes, conf], dim=-1)


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

        return x.view(-1, 5, 2)


class RecognitionModel(nn.Module):
    def __init__(self, num_classes: int = 100):
        super().__init__()

        backbone = models.resnet18(weights=None)

        self.features = nn.Sequential(*list(backbone.children())[:-1])

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class MyRecognitionModel(nn.Module):
    def __init__(self, num_classes: int = 105):
        super(MyRecognitionModel, self).__init__()

        self.block1 = nn.Sequential(
            ConvBlock(3, 32, padding=1),
            ConvBlock(32, 32, padding=1),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
        )

        self.block2 = nn.Sequential(
            ConvBlock(32, 64, padding=1),
            ConvBlock(64, 64, padding=1),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
        )

        self.block3 = nn.Sequential(
            ConvBlock(64, 128, padding=1),
            ConvBlock(128, 128, padding=1),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
        )

        self.block4 = nn.Sequential(
            ConvBlock(128, 256, padding=1),
            ConvBlock(256, 256, padding=1),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.global_pool(x)
        x = self.fc(x)
        return x
