import torch
import torch.nn as nn
import torch.nn.functional as F

class MNIST_MLP(nn.Module):
    def __init__(self):
        super(MNIST_MLP, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = x.view(-1, 784) # view是重构，permute是换位置，reshape是啥
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x
        
class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
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
    
net_1 = MNIST_CNN()
net_2 = MNIST_MLP()
print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in net_1.parameters())))
print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in net_2.parameters())))