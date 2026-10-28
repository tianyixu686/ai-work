"""Lightweight depthwise-separable residual CNN."""
from __future__ import annotations

import torch
from torch import nn


class DSResidual(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels), nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = (nn.Identity() if stride == 1 and in_channels == out_channels else
                     nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                                   nn.BatchNorm2d(out_channels)))
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.body(x) + self.skip(x))


class AnimalLiteNet(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.SiLU(),
            DSResidual(32, 48, 2), DSResidual(48, 48),
            DSResidual(48, 96, 2), DSResidual(96, 96),
            DSResidual(96, 160, 2), DSResidual(160, 160),
        )
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                        nn.Dropout(0.20), nn.Linear(160, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


if __name__ == "__main__":
    model = AnimalLiteNet()
    n = sum(p.numel() for p in model.parameters())
    print(model)
    print(f"parameters={n:,}")

