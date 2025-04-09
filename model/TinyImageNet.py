import torch.nn as nn
from torchvision.models import resnet18


class TinyImageNet_ResNet18(nn.Module):
    def __init__(self, num_classes=200):
        super(TinyImageNet_ResNet18, self).__init__()
        self.model = resnet18()

        # 修改输入适配 64x64
        self.model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.model.maxpool = nn.Identity()

        # 修改输出类别
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)
