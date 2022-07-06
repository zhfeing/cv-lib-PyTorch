import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        return self.relu(self.bn(self.conv(x)))


class AlexNet(nn.Module):

    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            Conv(3, 64, kernel_size=11, stride=4, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Conv(64, 192, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Conv(192, 384, kernel_size=3, padding=1),
            Conv(384, 256, kernel_size=3, padding=1),
            Conv(256, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def alexnet(**kwargs) -> AlexNet:
    return AlexNet(**kwargs)


MODEL_DICT = {
    "alexnet": alexnet
}
