import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.modules import ConvBlock, HybridBlock


class HybridNet(nn.Module):
    def __init__(
        self, in_channels: int = 3, embedding_size=512, dropout=0.4, device="cuda"
    ):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        self.stem = ConvBlock(
            in_channels=in_channels,
            out_channels=64,
            kernel=3,
            stride=1,
            padding=1,
            device=device,
        )

        self.stage1 = HybridBlock(64, 128, stride=2, local_blocks=2, device=device)
        self.stage2 = HybridBlock(128, 256, stride=2, local_blocks=2, device=device)
        self.stage3 = HybridBlock(
            256,
            512,
            stride=2,
            local_blocks=3,
            global_kernel=5,
            global_dilation=1,
            device=device,
        )
        self.stage4 = HybridBlock(
            512,
            512,
            stride=2,
            local_blocks=2,
            global_kernel=5,
            global_dilation=1,
            device=device,
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, embedding_size, bias=False, device=device)
        self.bn_final = nn.BatchNorm1d(embedding_size, device=device)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.avg_pool(x).view(x.size(0), -1)
        x = self.dropout(x)
        return self.bn_final(self.fc(x))
