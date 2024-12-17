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
        self.datasource = dataset # 数据源
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
        size = int(len(self.data) * theta)
        idcs = [idc for idc in range(len(self.data))]
        discard = random.sample(idcs, size)
        for item in discard:
            idcs.remove(item)
        self.data = Subset(self.data, idcs)
    
    def collect_data(self, increment):
        idcs = [idc for idc in range(len(self.datasource))]
        used = random.sample(idcs, increment)
        # print('left:{}, increment:{}'.format(len(idcs),increment))
        for item in used:
            idcs.remove(item)
        newdata = Subset(self.datasource, used)
        self.data = ConcatDataset([self.data, newdata]) if self.data != None else newdata
        self.datasource = Subset(self.datasource, idcs)
        
    def local_update(self,
                     t,
                     k,
                     num_epoch, 
                     batch_size, 
                     global_parameters,
                     theta,
                     datasize):

        self.global_parameters = global_parameters
        
        # rho = 0
        # beta = 0
        # mu = 0
        # old_loss = 0
        # if t != 0:
        #     # 计算本地参数的损失值和梯度值
        #     local_loss_list = []
        #     local_grad_list = []
        #     self.net.load_state_dict(self.local_parameters, strict=True)
        #     dataloader = DataLoader(self.data, batch_size=batch_size, shuffle=True)
        #     for epoch in range(num_epoch):
        #         self.optim.zero_grad()
        #         for batch in dataloader:
        #             data, label = batch
        #             data = data.to(self.dev)
        #             label = label.to(self.dev)
        #             pred = self.net(data)
        #             loss = self.criterion(pred, label)
        #             loss.backward()
        #             local_loss_list.append(loss)
                
        #         for param_group in self.optim.param_groups:
        #             params = param_group['params']
        #             for param in params:
        #                 grad = param.grad.reshape(-1)
        #                 local_grad_list.append(grad)

        #     local_loss = torch.mean(torch.tensor(local_loss_list))
        #     local_grad = torch.cat(local_grad_list)
        #     # print('client {}, round {}, 2'.format(k, t))
        #     # print(id(self.local_parameters))
        #     # print(self.local_parameters['fc2.bias'])
        #     # exit(0)
            
        #     # 计算全局参数的损失值和梯度
        #     global_loss_list = []
        #     global_grad_list = []
        #     self.net.load_state_dict(self.global_parameters, strict=True)
        #     dataloader = DataLoader(self.data, batch_size=batch_size, shuffle=True)
        #     for epoch in range(num_epoch):
        #         self.optim.zero_grad()
        #         for batch in dataloader:
        #             data, label = batch
        #             data = data.to(self.dev)
        #             label = label.to(self.dev)
        #             pred = self.net(data)
        #             loss = self.criterion(pred, label)
        #             loss.backward()
        #             global_loss_list.append(loss)
                
        #         for param_group in self.optim.param_groups:
        #             params = param_group['params']
        #             for param in params:
        #                 grad = param.grad.reshape(-1)
        #                 global_grad_list.append(grad)
                            
        #     global_loss = torch.mean(torch.tensor(global_loss_list))
        #     global_grad = torch.cat(global_grad_list)
        #     print(global_grad.shape)
        #     # print('client {}, round {}, 3'.format(k, t))
        #     # print(id(self.local_parameters))
        #     # print(self.local_parameters['fc2.bias'])
            
        #     loss_diff = local_loss - global_loss
        #     grad_diff = torch.sqrt(torch.sum(torch.square(local_grad - global_grad)))
            
        #     # 计算两个参数的差
        #     assert self.global_parameters.keys() == self.local_parameters.keys()
        #     param_diff = 0
        #     for key in self.local_parameters.keys():
        #         item1 = self.local_parameters[key]
        #         item2 = self.global_parameters[key]
        #         param_diff += torch.sum(torch.square(item1 - item2))
        #     param_diff = torch.sqrt(param_diff)
            
        #     # 估计rho
        #     rho = loss_diff / param_diff
            
        #     # 估计beta
        #     beta = grad_diff / param_diff
            
        #     # 估计mu
        #     global_grad_sum = torch.sum(torch.square(global_grad))
        #     mu = 0.5 * global_grad_sum / global_loss
            
        #     print('loss_diff = {}, grad_diff = {}, param_diff = {}, global_grad_sum = {}'.format(loss_diff, grad_diff, param_diff, global_grad_sum))
        #     print('rho = {}, beta = {}, mu = {}'.format(rho, beta, mu))
            
            # # 计算Omega_Part1
            # old_loss_list = []
            # self.net.load_state_dict(self.global_parameters, strict=True)
            # dataloader = DataLoader(self.data, batch_size=batch_size, shuffle=True)
            # for epoch in range(num_epoch):
            #     for batch in dataloader:
            #         data, label = batch
            #         data = data.to(self.dev)
            #         label = label.to(self.dev)
            #         pred = self.net(data)
            #         loss = self.criterion(pred, label)
            #         old_loss_list.append(loss)
            # old_loss = torch.mean(torch.tensor(old_loss_list))

        # 收集数据
        increment = int(datasize - theta * len(self.data)) if self.data != None else int(datasize)
        self.discard_data(theta) if self.data != None else None
        self.collect_data(increment)
        
        # 训练
        # 计算Omega_Part2
        grad_list = []
        self.net.load_state_dict(self.global_parameters, strict=True)
        dataloader = DataLoader(self.data, batch_size=batch_size, shuffle=True)
        for epoch in range(num_epoch):
            for batch in dataloader:
                data, label = batch
                data = data.to(self.dev)
                label = label.to(self.dev)
                pred = self.net(data)
                loss = self.criterion(pred, label)
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

        self.local_parameters = deepcopy(self.net.state_dict())
        return self.local_parameters, new_grad
            

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

        if self.is_iid:
            client_idcs = self.iid_split(
                train_label=train_label,
                num_client=self.num_client
            )
        else:
            client_idcs = self.dirichlet_split(
                train_label=train_label,
                alpha=self.alpha,
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
        
        for idc in client_idcs:
            local_data = train_data[idc]
            local_label = train_label[idc]
            local_dataset = Local_Dataset(local_data, local_label, trans)
            client = Client(self.dataset_name, local_dataset, self.dev, self.net_name, self.learning_rate)
            self.clients.append(client)
            self.scales.append(len(idc))
        
        self.test_dataset = Local_Dataset(test_data, test_label, trans)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=100, shuffle=False)