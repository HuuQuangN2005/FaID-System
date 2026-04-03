import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel=3,
        stride=1,
        padding=0,
        act="prelu",
        device="cuda",
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            bias=False,
            device=device,
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels, device=device)

        if act == "prelu":
            self.act = nn.PReLU(num_parameters=out_channels, device=device)
        elif act == "silu":
            self.act = nn.SiLU(inplace=True)
        else:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        return self.act(out)


class DConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel=3,
        stride=1,
        padding=0,
        dilation=1,
        act="prelu",
        device="cuda",
    ):
        super().__init__()

        if dilation != 1:
            self.padding = (dilation * (kernel - 1)) // 2
        else:
            self.padding = padding

        self.dw = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel,
            stride=stride,
            padding=self.padding,
            groups=in_channels,
            dilation=dilation,
            bias=False,
            device=device,
        )

        self.pw = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            device=device,
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels, device=device)
        if act == "prelu":
            self.act = nn.PReLU(num_parameters=out_channels, device=device)
        elif act == "silu":
            self.act = nn.SiLU(inplace=True)
        else:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.dw(x)
        out = self.pw(out)
        out = self.bn(out)
        return self.act(out)


class LocalBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel=3, stride=1, num_blocks=3, device="cuda"
    ):
        super(LocalBlock, self).__init__()
        padding = (kernel - 1) // 2

        layers = []
        layers.append(
            DConvBlock(
                in_channels, out_channels, kernel, stride, padding, device=device
            )
        )

        for _ in range(num_blocks - 1):
            layers.append(
                DConvBlock(
                    out_channels, out_channels, kernel, 1, padding, device=device
                )
            )

        self.layers = nn.Sequential(*layers)

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, 1, stride, bias=False, device=device
                ),
                nn.BatchNorm2d(out_channels, device=device),
            )

    def forward(self, x):
        return self.layers(x) + self.shortcut(x)


class GlobalBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel=7,
        stride=1,
        padding=0,
        dilation=1,
        device="cuda",
    ):
        super(GlobalBlock, self).__init__()

        self.padding = (dilation * (kernel - 1)) // 2

        self.dconv = DConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=kernel,
            stride=stride,
            padding=self.padding,
            dilation=dilation,
            device=device,
        )

    def forward(self, x):
        return self.dconv(x)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16, device="cuda"):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False, device=device),
            nn.PReLU(channels // reduction, device=device),
            nn.Linear(channels // reduction, channels, bias=False, device=device),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class HybridBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, local_blocks = 3,global_dilation = 2, global_kernel = 7, device="cuda"):
        super().__init__()
        self.project = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=1,
            stride=stride,
            padding=0,
            device=device,
        )

        self.mid_channels = out_channels // 2

        self.local_branch = LocalBlock(
            in_channels=self.mid_channels,
            out_channels=self.mid_channels,
            num_blocks=local_blocks,
            device=device,
        )
        self.global_branch = GlobalBlock(
            in_channels=self.mid_channels,
            out_channels=self.mid_channels,
            kernel=global_kernel,
            dilation=global_dilation,
            device=device,
        )

        self.se = SEBlock(channels=out_channels, device=device)

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, 1, stride, bias=False, device=device
                ),
                nn.BatchNorm2d(out_channels, device=device),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        x_proj = self.project(x)
        x_l, x_g = torch.chunk(x_proj, 2, dim=1)
        out = torch.cat([self.local_branch(x_l), self.global_branch(x_g)], dim=1)

        return self.se(out) + identity

