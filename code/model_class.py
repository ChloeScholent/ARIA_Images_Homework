import torch
from torch import nn


class MNISTCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
                    )

        self.linear_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64*24*24, out_features=num_classes)
        )

    def forward(self, x):
        out = self.cnn_layers(x)
        out = self.linear_layers(out)
        return out


