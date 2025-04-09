import torch
import torch.nn as nn
import torch.nn.functional as F


class FMNIST_LR(nn.Module):
    def __init__(self):
        super(FMNIST_LR, self).__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc(x)
        return x


class FMNIST_MLP(nn.Module):
    def __init__(self):
        super(FMNIST_MLP, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)  # view是重构，permute是换位置，reshape是啥
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class FMNIST_CNN(torch.nn.Module):
    def __init__(self):
        super(FMNIST_CNN, self).__init__()
        # 卷积层：图片大小不变，通道数翻倍
        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.conv3 = torch.nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        # 池化层：通道数不变，图片规模减半
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # 线性层
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 10)
        # 归一化层
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)

    def forward(self, x):
        # [60000, 1, 28, 28] -> [60000, 32, 28, 28]
        x = F.relu(
            self.batch_norm1(self.conv1(x))
        )  # Conv1 + BatchNorm + ReLU + MaxPool
        # [60000, 32, 28, 28] -> [60000, 32, 14, 14]
        x = self.pool(x)
        # [60000, 32, 14, 14] -> [60000, 64, 14, 14]
        x = F.relu(self.batch_norm2(self.conv2(x)))
        # [60000, 64, 14, 14] -> [60000, 64, 7, 7]
        x = self.pool(x)
        # [60000, 64, 7, 7] -> [60000, 128, 7, 7]
        x = F.relu(self.batch_norm3(self.conv3(x)))
        # [60000, 128, 7, 7] -> [60000, 128, 3, 3]
        x = self.pool(x)

        x = x.view(-1, 128 * 3 * 3)
        # [60000, 128 * 3 * 3] -> [60000, 512]
        x = F.relu(self.fc1(x))
        # [60000, 512] -> [60000, 10]
        x = self.fc2(x)
        return x


# net = FMNIST_CNN()
# print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in net.parameters())))
