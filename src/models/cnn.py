import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioCNN(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(
            kernel_size=2
        )

        self.dropout = nn.Dropout(0.3)

        # Force the CNN output to a fixed spatial size.
        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (1, 32)
        )

        self.fc1 = nn.Linear(
            64 * 1 * 32,
            128
        )

        self.fc2 = nn.Linear(
            128,
            num_classes
        )

    def forward(self, x):

        # [batch, 6, 140]
        x = x.unsqueeze(1)

        # [batch, 1, 6, 140]

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        # Make output consistently:
        # [batch, 64, 1, 32]
        x = self.adaptive_pool(x)

        x = self.dropout(x)

        x = torch.flatten(
            x,
            start_dim=1
        )

        # [batch, 2048]

        x = self.fc1(x)
        x = F.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        return x