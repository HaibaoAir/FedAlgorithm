import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize


class Cifar10_CNN(torch.nn.Module):
    def __init__(self):
        super(Cifar10_CNN, self).__init__()
        # 卷积层：图片大小不变，通道数翻倍
        self.conv1 = torch.nn.Conv2d(in_channels=3, # channel部分
                                    out_channels=16,
                                    kernel_size=3, # size部分
                                    stride=1,
                                    padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=16,
                                    out_channels=32,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=32,
                                    out_channels=64,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
        # 池化层：通道数不变，图片规模减半
        self.pool = nn.MaxPool2d(kernel_size=2,
                                 stride=2,
                                 padding=0)
        
        # 线性层
        self.fc1 = nn.Linear(in_features=64 * 4 * 4, 
                             out_features=512)
        self.fc2 = nn.Linear(in_features=512,
                             out_features=10)
        # dropout层
        self.dropout = nn.Dropout(p=0.3)
 
    def forward(self, x):
        # [50000, 3, 32, 32] -> [50000, 16, 32, 32]
        x = F.relu(self.conv1(x))
        # [50000, 16, 32, 32] -> [50000, 16, 16, 16]
        x = self.pool(x)
        # [50000, 16, 16, 16] -> [50000, 32, 16, 16]
        x = F.relu(self.conv2(x))
        # [50000, 32, 16, 16] -> [50000, 32, 8, 8]
        x = self.pool(x)
        # [50000, 32, 8, 8] -> [50000, 64, 8, 8]
        x = F.relu(self.conv3(x))
        # [50000, 64, 8, 8] -> [50000, 64, 4, 4]
        x = self.pool(x)
        
        x = x.view(-1, 64 * 4 * 4)
        # [50000, 64 * 4 * 4] -> [50000, 512]
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        # [50000, 512] -> [50000, 10]
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CIFAR10_Model(nn.Module):
    def __init__(self, in_planes=3, num_classes=10):
        super().__init__()
        self.cnn_block_1 = nn.Sequential(*[
            nn.Conv2d(in_planes, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25)
        ])
        self.cnn_block_2 = nn.Sequential(*[
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25)
        ])
        self.flatten = lambda inp: torch.flatten(inp, 1)
        self.head = nn.Sequential(*[
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        ])

    def forward(self, X):
        X = X.view(-1, 3, 32, 32)
        X = self.cnn_block_1(X)
        X = self.cnn_block_2(X)
        X = self.flatten(X)
        X = self.head(X)
        return X