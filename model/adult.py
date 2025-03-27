import torch
import torch.nn as nn
import torch.nn.functional as F

class Adult_MLP(nn.Module):
    def __init__(self, input_dim):
        super(Adult_MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)   # 第一层：输入到隐藏层
        self.fc2 = nn.Linear(64, 32)          # 第一层：输入到隐藏层
        self.fc3 = nn.Linear(32, 1)           # 输出层
        self.relu = nn.ReLU()                 # ReLU 激活函数

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x