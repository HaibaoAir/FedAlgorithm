import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from model.mnist import MNIST_MLP
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# 超参数
num_task = 3
tasks = [0, 1, 2]
num_epoch = 10 # 整除验证频率
device = torch.device('cuda')
intensity = 1e7
eval_freq = 2
save_path = '../logs/ewc_version_4'

optimal_params_list = []
fisher_information_list = []


class Local_Dataset(Dataset):
    def __init__(self,
                 local_data,
                 local_label,
                 transform,
                 ):
        self.local_data = local_data
        self.local_label = local_label
        self.transform = transform
            
    def __getitem__(self, idx):
        data = self.transform(self.local_data[idx]) # 加载器只会感应到最近一层的dataset的transform
        label = self.local_label[idx]
        return data, label
    
    def __len__(self):
        return self.local_label.shape[0]


def load_data():
    transform = transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.ToTensor(), # [0, 255] -> [0, 1]
                                transforms.Normalize((0.13065973,),(0.3015038,))])

    mnist_train = torchvision.datasets.MNIST(root='../data', download=True, train=True)
    mnist_test = torchvision.datasets.MNIST(root='../data', download=True, train=False)
    
    train_images = mnist_train.data.numpy() # [60000, 28, 28]
    train_labels = mnist_train.targets.numpy()
    test_images = mnist_test.data.numpy()
    test_labels = mnist_test.targets.numpy()
    
    tasks_train_dataset = []
    tasks_test_dataset = []
    for _ in range(num_task):
        permute_idx = np.random.permutation(train_images.shape[1] * train_images.shape[2])
        
        train_images_reshaped = train_images.reshape(train_images.shape[0], -1)
        shuffled_train_images = train_images_reshaped[:, permute_idx]
        shuffled_train_images = shuffled_train_images.reshape(train_images.shape)
        
        test_images_reshaped = test_images.reshape(test_images.shape[0], -1)
        shuffled_test_images = test_images_reshaped[:, permute_idx]
        shuffled_test_images = shuffled_test_images.reshape(test_images.shape)
        
        tasks_train_dataset.append(Local_Dataset(shuffled_train_images, train_labels, transform))
        tasks_test_dataset.append(Local_Dataset(shuffled_test_images, test_labels, transform))
    
    return tasks_train_dataset, tasks_test_dataset

# def load_data():
#     mnist_train = torchvision.datasets.MNIST("../data", train=True, download=True, transform=transforms.ToTensor())
#     mnist_test = torchvision.datasets.MNIST("../data", train=False, download=True, transform=transforms.ToTensor())

#     f_mnist_train = torchvision.datasets.FashionMNIST("../data", train=True, download=True, transform=transforms.ToTensor())
#     f_mnist_test = torchvision.datasets.FashionMNIST("../data", train=False, download=True, transform=transforms.ToTensor())

#     tasks_train_dataset = [mnist_train, f_mnist_train]
#     tasks_test_dataset = [mnist_test, f_mnist_test]
    
#     return tasks_train_dataset, tasks_test_dataset

class LinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim, act='relu', use_bn=False):
        super(LinearLayer, self).__init__()
        self.use_bn = use_bn
        self.lin = nn.Linear(input_dim, output_dim)
        self.act = nn.ReLU() if act == 'relu' else act
        if use_bn:
            self.bn = nn.BatchNorm1d(output_dim)
    def forward(self, x):
        if self.use_bn:
            return self.bn(self.act(self.lin(x)))
        return self.act(self.lin(x))

class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.shape[0], -1)
    
class BaseModel(nn.Module):
    
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super(BaseModel, self).__init__()
        self.f1 = Flatten()
        self.lin1 = LinearLayer(num_inputs, num_hidden, use_bn=True)
        self.lin2 = LinearLayer(num_hidden, num_hidden, use_bn=True)
        self.lin3 = nn.Linear(num_hidden, num_outputs)
        
    def forward(self, x):
        return self.lin3(self.lin2(self.lin1(self.f1(x))))


def EWC_loss(pred, label, model):
    
    criterion = torch.nn.CrossEntropyLoss()
    base_loss = criterion(pred, label)
    
    WEC_list = []
    for kappa in range(len(optimal_params_list)):
        WEC = 0
        for name, current_param in model.named_parameters(): 
            optimal_param = optimal_params_list[kappa][name]
            fisher = fisher_information_list[kappa][name]
            WEC += (fisher * (current_param - optimal_param) ** 2).sum()
        WEC_list.append(WEC)
        
    loss = base_loss + (intensity / 2) * sum(WEC_list)
    return loss


def main():    
    # 加载数据
    tasks_train_dataset, tasks_test_dataset = load_data()
    
    # 定义网络-共用一个参数
    model = BaseModel(28 * 28, 100, 10)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    
    # 正式训练
    hist_tasks = []
    for k in tasks:
        
        hist_tasks.append(k)
        
        # 加载数据集
        train_dataloader = DataLoader(tasks_train_dataset[k], batch_size=100, shuffle=True)

        # 训练模型
        acc_matrix = torch.zeros(int(num_epoch / eval_freq), len(hist_tasks))
        for epoch in tqdm(range(num_epoch)):
            total_loss = 0
            for batch in train_dataloader:
                data, label = batch
                data = data.to(device)
                label = label.to(device)
                pred = model(data)
                loss = EWC_loss(pred, label, model)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            total_loss /= len(train_dataloader)
            
            if epoch % eval_freq == 0:
                with torch.no_grad():
                    acc_list = []
                    for kappa in hist_tasks:
                        test_dataloader = DataLoader(tasks_test_dataset[kappa], batch_size=100, shuffle=False)
                        correct = 0
                        total = 0
                        for batch in test_dataloader:
                            data, label = batch
                            data = data.to(device)
                            label = label.to(device)
                            pred = model(data)
                            pred = torch.argmax(pred, dim=1)
                            correct += (pred == label).sum().item()
                            total += label.shape[0]
                        acc = correct / total
                        acc_list.append(acc)
                    acc_list = torch.tensor(acc_list)
                    acc_matrix[int(epoch / eval_freq)] = acc_list
                
        acc_matrix = acc_matrix.transpose(0, 1)
        
        color = ['C0', 'C2', 'C3', 'C4', 'C5']
        marker = ['^', 's', 'o', 'v', 'D', 's', '+', 'p', ',']
        linestyle = [':', '-', ':', ':', ':']
        for kappa in range(len(hist_tasks)):
            plt.plot(acc_matrix[kappa], color=color[kappa], marker=marker[kappa], markersize=4, linestyle=linestyle[kappa], label='task {}'.format(hist_tasks[kappa]))
        plt.ylim(0, 1)
        plt.legend()
        plt.savefig(save_path + '_task_{}.jpg'.format(k))
        plt.close()
        
        print('acc_list:{}'.format(acc_matrix[:, -1]))

        
        # 浅拷贝
        # params = model.state_dict()
        # 深拷贝
        # 又因为历史任务参数永久冻结，放到EWC_loss中时反向传播不再更新其参数，因而是detach
        params = {name: param.clone().detach() for name, param in model.named_parameters()}
        optimal_params_list.append(params)

        
        # 计算Fisher信息
        log_probs_list = []
        for batch in train_dataloader:
            ## 计算似然函数
            data, label = batch
            data = data.to(device)
            label = label.to(device)
            pred = model(data)
            log_probs = torch.log_softmax(pred, dim=1)
            log_probs = pred[range(len(label)), label]
            log_probs_list.append(log_probs)
        log_probs_avg = torch.cat(log_probs_list).mean()
        # 求导
        model.zero_grad()
        log_probs_avg.backward()
        ## 平方
        fisher_information = {}
        for name, params in model.named_parameters():
            fisher_information[name] = params.grad.clone().detach() ** 2
            
        fisher_information_list.append(fisher_information)
main()
        
    
            
