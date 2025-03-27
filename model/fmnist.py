import torch
import torch.nn as nn
import torch.nn.functional as F

class FMNIST_CNN(nn.Module):
    def __init__(self):
        super(FMNIST_CNN, self).__init__()
        # 卷积层：图片规模不变，通道数翻倍
        self.conv1 = nn.Conv2d(in_channels=1, # channel部分
                               out_channels=32, 
                               kernel_size=5, # size部分：(width + 2 * padding - kernel) / stride + 1
                               stride=1, 
                               padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, 
                               out_channels=64, 
                               kernel_size=5, 
                               stride=1, 
                               padding=2)
        # 池化层：通道数不变，图片规模减半
        self.pool = nn.MaxPool2d(kernel_size=2,
                                  stride=2,
                                  padding=0)
        # 线性层
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, 
                                 out_features=512)
        
        self.fc2 = nn.Linear(in_features=512, 
                                 out_features=10)
        # Dropout层
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        # [60000, 1, 28, 28] -> [60000, 32, 28, 28]
        x = F.relu(self.conv1(x))
        # [60000, 32, 28, 28] -> [60000, 32, 14, 14]
        x = self.pool(x)
        # [60000, 32, 14, 14] -> [60000, 64, 14, 14]
        x = F.relu(self.conv2(x))
        # [60000, 64, 14, 14] -> [60000, 64, 7, 7]
        x = self.pool(x)
        
        x = x.view(-1, 64 * 7 * 7)
        # [60000, 64 * 7 * 7] -> [60000, 512]
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        # [60000, 512] -> [60000, 10]
        x = self.dropout(x)
        x = self.fc2(x)
        return x
        
# class FMNIST_CNN(nn.Module):
#     def __init__(self):
#         super(FMNIST_CNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # 1x28x28 -> 32x28x28
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 32x28x28 -> 64x28x28
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # 64x28x28 -> 128x28x28
#         self.pool = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14

#         self.fc1 = nn.Linear(128 * 3 * 3, 512)  # 展平为一维后，连接全连接层
#         self.fc2 = nn.Linear(512, 10)  # 输出10个类（Fashion MNIST有10个类别）

#         self.batch_norm1 = nn.BatchNorm2d(32)
#         self.batch_norm2 = nn.BatchNorm2d(64)
#         self.batch_norm3 = nn.BatchNorm2d(128)

#     def forward(self, x):
#         x = self.pool(torch.relu(self.batch_norm1(self.conv1(x))))  # Conv1 + BatchNorm + ReLU + MaxPool
#         # print(x.shape)
#         x = self.pool(torch.relu(self.batch_norm2(self.conv2(x))))  # Conv2 + BatchNorm + ReLU + MaxPool
#         # print(x.shape)
#         x = self.pool(torch.relu(self.batch_norm3(self.conv3(x))))  # Conv3 + BatchNorm + ReLU + MaxPool
#         # print(x.shape)
#         x = x.view(-1, 128 * 3 * 3)  # 展平
#         x = torch.relu(self.fc1(x))  # 全连接层1 + ReLU
#         x = self.fc2(x)  # 输出层
#         return x
    
net = FMNIST_CNN()
print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in net.parameters())))