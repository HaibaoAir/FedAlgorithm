import torch
import torch.nn as nn
from torchvision.models import resnet50


class Cifar100_ResNet50(nn.Module):
    def __init__(self, num_classes=100):
        super(Cifar100_ResNet50, self).__init__()
        self.model = resnet50()

        # 修改首层卷积以适配 CIFAR-100 输入 (32x32)
        self.model.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.model.maxpool = nn.Identity()  # CIFAR-100 不需要首层 maxpool

        # 修改最后分类层，原来是 Linear(2048, 1000)
        self.model.fc = nn.Linear(in_features=2048, out_features=num_classes)

    def forward(self, x):
        return self.model(x)
