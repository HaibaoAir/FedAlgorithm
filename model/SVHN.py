import torch
import torch.nn as nn
import torch.nn.functional as F


class SVHN_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SVHN_CNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                3, 64, kernel_size=3, padding=1
            ),  # [B, 3, 32, 32] -> [B, 64, 32, 32]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # [B, 64, 32, 32]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> [B, 64, 16, 16]
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # -> [B, 128, 16, 16]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # -> [B, 128, 16, 16]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> [B, 128, 8, 8]
        )

        self.fc_block = nn.Sequential(
            nn.Flatten(),  # -> [B, 8*8*128]
            nn.Linear(8 * 8 * 128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),  # -> [B, 10]
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.fc_block(x)
        return x
