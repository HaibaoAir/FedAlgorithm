import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, Subset, ConcatDataset, DataLoader # 三件套
from torchvision import transforms
import matplotlib.pyplot as plt
import random
from copy import deepcopy

import sys
sys.path.append('../..')
from model.mnist import MNIST_Linear, MNIST_CNN
from model.cifar import Cifar10_CNN

seed = 10
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

class Local_Dataset(Dataset):
    def __init__(self,
                 local_data,
                 local_label,
                 trans):
        self.local_data = local_data
        self.local_label = local_label
        self.trans = trans
        
    def __getitem__(self, index):
        data = self.trans(self.local_data[index])
        label = self.local_label[index]
        return data, label
    
    def __len__(self):
        return self.local_label.shape[0]


class Client(object):
    def __init__(self, 
                 dataset_name,
                 dataset,
                 dev,
                 net_name,
                 learning_rate):
        self.dataset_name = dataset_name
        self.datasource_list = dataset # 数据源，是列表
        self.dev = dev
        self.data = None
        self.global_parameters = None # 下发的全局模型
        self.local_parameters = None # 上传的本地模型

        # 定义net
        self.net = None
        if self.dataset_name == 'mnist':
            if net_name == 'linear':
                self.net = MNIST_Linear()
            elif net_name == 'cnn':
                self.net = MNIST_CNN()
            else:
                raise NotImplementedError('{}'.format(net_name))
        elif self.dataset_name == 'cifar10':
            if net_name == 'cnn':
                self.net = Cifar10_CNN()
            else:
                raise NotImplementedError('{}'.format(net_name))
        else:
            raise NotImplementedError('{}'.format(net_name))
        self.net.to(self.dev)
        
        # 定义loss function
        self.criterion = F.cross_entropy # 交叉熵：softmax + NLLLoss 参考知乎
        
        # 定义optimizer
        self.optim = torch.optim.SGD(self.net.parameters(), learning_rate)
        
    def discard_data(self, theta):
        size = int(len(self.data) * (1 - theta))
        idcs = [idc for idc in range(len(self.data))]
        discard = random.sample(idcs, size)
        for item in discard:
            idcs.remove(item)
        self.data = Subset(self.data, idcs)
    
    def collect_data(self, t, increment):
        tau = t % 10
        idcs = [idc for idc in range(len(self.datasource_list[tau]))]
        used = random.sample(idcs, increment)
        # print('left:{}, increment:{}'.format(len(idcs),increment))
        for item in used:
            idcs.remove(item)
        newdata = Subset(self.datasource_list[tau], used)
        self.data = ConcatDataset([self.data, newdata]) if self.data != None else newdata
        self.datasource_list = Subset(self.datasource_list[tau], idcs)
        
    def local_update(self,
                     t,
                     k,
                     num_epoch, 
                     batch_size, 
                     global_parameters,
                     theta,
                     datasize):

        self.global_parameters = global_parameters

        # 收集数据
        increment = int(datasize - theta * len(self.data)) if self.data != None else int(datasize)
        self.discard_data(theta) if self.data != None else None
        self.collect_data(t, increment)
        
        # 训练
        # 计算Omega_Part2
        grad_list = []
        loss_list = []
        self.net.load_state_dict(self.global_parameters, strict=True)
        dataloader = DataLoader(self.data, batch_size=batch_size, shuffle=True)
        for epoch in range(num_epoch):
            for batch in dataloader:
                data, label = batch
                data = data.to(self.dev)
                label = label.to(self.dev)
                pred = self.net(data)
                loss = self.criterion(pred, label)
                loss_list.append(loss.cpu().detach())
                self.optim.zero_grad()
                loss.backward()
                tmp_grad_list = []
                for param_group in self.optim.param_groups:
                    params = param_group['params']
                    for param in params:
                        grad = param.grad.reshape(-1)
                        tmp_grad_list.append(grad)
                tmp_grad = torch.cat(tmp_grad_list)
                grad_list.append(tmp_grad)
                self.optim.step()
        new_grad = sum(grad_list) / len(grad_list)
        new_loss = sum(loss_list) / len(loss_list)

        self.local_parameters = deepcopy(self.net.state_dict())
        return self.local_parameters, new_grad, new_loss
            

class Client_Group(object):
    def __init__(self,
                 dev,
                 num_client,
                 dataset_name,
                 is_iid,
                 alpha,
                 net_name,
                 learning_rate):
        self.dev = dev
        self.num_client = num_client
        self.clients = []
        self.scales = []
        self.dataset_name = dataset_name
        self.is_iid = is_iid
        self.alpha = alpha
        self.net_name = net_name
        self.learning_rate = learning_rate
        self.test_dataloader = None
        self.dataset_allocation()
    
    
    def load_mnist(self):
        # 下载：[60000, 28, 28], tensor + tensor
        train_dataset = torchvision.datasets.MNIST(root='../../data', download=True, train=True)
        test_dataset = torchvision.datasets.MNIST(root='../../data', download=True, train=False)
        train_data = train_dataset.data
        test_data = test_dataset.data
        train_label = train_dataset.targets
        test_label = test_dataset.targets
        # 预处理：先划分到各客户端再预处理，因此分开
        trans = transforms.Compose([transforms.ToPILImage(),
                                    transforms.ToTensor(), # [0, 255] -> [0, 1]
                                    transforms.Normalize((0.13065973,),(0.3015038,))])  # [0, 1] -> [-1, 1]
        return train_data, train_label, test_data, test_label, trans
    
     
    def load_cifar10(self):
        # 下载：[50000, 32, 32, 3], tensor + list
        train_dataset = torchvision.datasets.CIFAR10(root='../../data', download=True, train=True)
        test_dataset = torchvision.datasets.CIFAR10(root='../../data', download=True, train=False)
        train_data = torch.tensor(train_dataset.data).permute(0, 3, 1, 2)
        train_label = torch.tensor(train_dataset.targets)
        test_data = torch.tensor(test_dataset.data).permute(0, 3, 1, 2)
        test_label = torch.tensor(test_dataset.targets)
        trans = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize(32),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2033, 0.1994, 0.2010))])
        return train_data, train_label, test_data, test_label, trans
    
    # 划分独立同分布数据    
    def iid_split(self, train_label, num_client):
        all_idcs = [i for i in range(train_label.shape[0])]
        num_local_dataset = int(train_label.shape[0] / num_client)
        client_idcs = []
        for i in range(num_client):
            client_idcs.append(np.random.choice(all_idcs, num_local_dataset))
            all_idcs = list(set(all_idcs) - set(client_idcs[-1]))
        return client_idcs
    
    def iid_split_1(self, train_label, num_client):
        idcs = [idc for idc in range(train_label.shape[0])]
        num_dataset = int(train_label.shape[0] / num_client)
        client_idcs = []
        for k in range(num_client):
            used = random.sample(idcs, num_dataset)
            client_idcs.append(used)
            for idc in used:
                idcs.remove(idc)
        return client_idcs
    
    def iid_split_2(self, train_label, num_client):
        client_idcs = [[] for k in range(num_client)]
        for target in range(10):
            target_idcs = np.where(train_label == target)[0]
            num_local_target_idcs = int(len(target_idcs) / num_client)
            for k in range(num_client):
                client_idcs[k].append(np.random.choice(target_idcs, num_local_target_idcs))
                target_idcs = list(set(target_idcs) - set(client_idcs[k][-1]))
        return client_idcs
    
    # 划分非独立同分布数据
    def dirichlet_split(self, train_label, alpha, num_client):
        train_label = np.array(train_label)
        num_class = train_label.max() + 1
        # (K, N) class label distribution matrix X, record how much each client occupies in each class
        label_distribution = np.random.dirichlet([alpha] * num_client, num_class) 
        # Record the sample subscript corresponding to each K category
        class_idcs = [np.argwhere(train_label==y).flatten() for y in range(num_class)]
        # Record the index of N clients corresponding to the sample set respectively
        client_idcs = [[] for _ in range(num_client)] 
        for c, fracs in zip(class_idcs, label_distribution):
            # np.split divides the samples of class k into N subsets according to the proportion
            # for i, idcs is to traverse the index of the sample set corresponding to the i-th client
            for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
                client_idcs[i] += [idcs]
        client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
        return client_idcs        
    
    # 下载、预处理、训练集划分和封装、测试集封装
    def dataset_allocation(self):
        if self.dataset_name == 'mnist':
            train_data, train_label, test_data, test_label, trans = self.load_mnist()
        elif self.dataset_name == 'cifar10':
            train_data, train_label, test_data, test_label, trans = self.load_cifar10()
        else:
            raise NotImplementedError('{}'.format(self.dataset_name))

        if self.is_iid == 1:
            client_idcs = self.iid_split(
                train_label=train_label,
                num_client=self.num_client
            )
        elif self.is_iid == 0:
            client_idcs = self.dirichlet_split(
                train_label=train_label,
                alpha=self.alpha,
                num_client=self.num_client
            )
        else:
            client_idcs = self.iid_split_2( # 是个矩阵
                train_label=train_label,
                num_client=self.num_client
            )
        
        # # 刻画结果
        # plt.figure(figsize=(12, 8))
        # plt.hist([train_label[idc]for idc in client_idcs], stacked=True,
        #         bins=np.arange(min(train_label)-0.5, max(train_label) + 1.5, 1),
        #         label=["Client {}".format(i) for i in range(self.num_client)],
        #         rwidth=0.5)
        # plt.xticks(np.arange(10))
        # plt.xlabel("Label type")
        # plt.ylabel("Number of samples")
        # plt.legend(loc="upper right")
        # plt.title("Display Label Distribution on Different Clients")
        # plt.show()
        
        if self.is_iid == 0 or self.is_iid == 1:
            for idc in client_idcs:
                local_data = train_data[idc]
                local_label = train_label[idc]
                local_dataset = Local_Dataset(local_data, local_label, trans)
                client = Client(self.dataset_name, local_dataset, self.dev, self.net_name, self.learning_rate)
                self.clients.append(client)
                
        else:
            for k in range(self.num_client):
                local_dataset_list = []
                for target in range(10):
                    local_data = train_data[client_idcs[k][target]]
                    local_label = train_label[client_idcs[k][target]]
                    local_dataset = Local_Dataset(local_data, local_label, trans)
                    local_dataset_list.append(local_dataset)
                client = Client(self.dataset_name, local_dataset_list, self.dev, self.net_name, self.learning_rate)
                self.clients.append(client)
                    
        self.test_dataset = Local_Dataset(test_data, test_label, trans)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=100, shuffle=False)