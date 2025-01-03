from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, Subset, ConcatDataset, DataLoader # 三件套
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
    
    
class Shuffle_Dataset(Dataset):
    def __init__(self,
                 local_data,
                 local_label,
                 ):
        self.local_data = local_data
        self.local_label = local_label
        
    def __getitem__(self, index):
        data = self.local_data[index]
        label = self.local_label[index]+10
        return data, label
    
    def __len__(self):
        return len(self.local_label)


class Client(object):
    def __init__(self, 
                 dataset_name,
                 dataset,
                 dev,
                 net_name,
                 learning_rate):
        self.dataset_name = dataset_name
        self.datasource_list = dataset # 数据源，是列表
        # 三个一组
        # self.new_datasource_list = []
        # self.new_datasource_list.append(ConcatDataset([self.datasource_list[0], self.datasource_list[3], self.datasource_list[6]]))
        # self.new_datasource_list.append(ConcatDataset([self.datasource_list[1], self.datasource_list[4], self.datasource_list[7]]))
        # self.new_datasource_list.append(ConcatDataset([self.datasource_list[2], self.datasource_list[5], self.datasource_list[8], self.datasource_list[9]]))
        # self.datasource_list = self.new_datasource_list
        # self.cycle = 3
        
        self.dev = dev
        self.data = None
        self.global_parameters = None # 下发的全局模型
        self.local_parameters = None # 上传的本地模型

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
        self.criterion = F.cross_entropy # 交叉熵：softmax + NLLLoss 参考知乎
        self.optim = torch.optim.SGD(self.net.parameters(), learning_rate)
        
    def poison(self, sigma):
        idcs = [idc for idc in range(len(self.data))]
        poison = random.sample(idcs, int(len(self.data) * sigma))
        for item in poison:
            idcs.remove(item)
        health_data = Subset(self.data, idcs)
        poison_data = Subset(self.data, poison)
        data_list = []
        label_list = []
        for item in poison_data:
            data = item[0]
            label = item[1]
            data_list.append(data)
            label_list.append((label+1) % 10)
        poison_data = Shuffle_Dataset(data_list, label_list)
        self.data = ConcatDataset([health_data, poison_data])
        
    def init_data(self, t, k, datasize):
        independent_size = datasize // 10
        for tau in range(10):
            idcs = [idc for idc in range(len(self.datasource_list[tau]))]
            used = random.sample(idcs, independent_size)
            # print('left:{}, increment:{}'.format(len(idcs),increment))
            for item in used:
                idcs.remove(item)
            newdata = Subset(self.datasource_list[tau], used)
            self.data = ConcatDataset([self.data, newdata]) if self.data != None else newdata
            self.datasource_list[tau] = Subset(self.datasource_list[tau], idcs)
        
    def discard_data(self, theta):
        size = int(len(self.data) * (1 - theta))
        idcs = [idc for idc in range(len(self.data))]
        discard = random.sample(idcs, size)
        for item in discard:
            idcs.remove(item)
        self.data = Subset(self.data, idcs)
    
    def collect_data(self, t, k, increment):
        independent_size = increment // 10
        for tau in range(10):
            idcs = [idc for idc in range(len(self.datasource_list[tau]))]
            used = random.sample(idcs, independent_size)
            # print('left:{}, increment:{}'.format(len(idcs),increment))
            for item in used:
                idcs.remove(item)
            newdata = Subset(self.datasource_list[tau], used)
            self.data = ConcatDataset([self.data, newdata]) if self.data != None else newdata
            self.datasource_list[tau] = Subset(self.datasource_list[tau], idcs)
        
    def local_update(self,
                     idx,
                     t,
                     k,
                     sigma,
                     num_epoch,
                     batch_size,
                     global_parameters,
                     theta,
                     increment_next,
                     datasize):

        self.global_parameters = global_parameters
        sigma = sigma / 20
        
        # 收集数据
        if t == 0:
            datasize = int(datasize)
            self.init_data(t, k, datasize)
            self.poison(0.6) # 本身的有毒性
            # print('init_data_1:{}, init_data_2:{}, increment_next:{}'.format(datasize, len(self.data), increment_next))
        else:
            self.poison(0.6) # 后续的有毒性
            # print('origin', len(self.data))
            self.discard_data(theta)
            # print('decay', len(self.data))
            # increment = int(datasize - theta * len(self.data)) 串联了呀
            increment = int(datasize - len(self.data))
            self.collect_data(t, k, increment)
            # print('accumu', len(self.data)) 
            # print('res_data_1:{}, res_data_2:{}, theta:{}, increment:{}, increment_next:{}'.format(datasize, len(self.data), theta, increment, increment_next))
        
        
        if t == 0:
            self.count_matrix = []
            
        width = 0.4
        if k == 0:
            # 数据分布
            count_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
            for item in self.data:
                label = item[1].item()
                if label < 10:
                    count_dict[label] += 1
                else:
                    count_dict[10] += 1
            print(count_dict)
            self.count_matrix.append(list(count_dict.values()))
            material = np.array(self.count_matrix)
            for target in range(11):
                if target == 0:
                    plt.bar(np.array(range(material.shape[0])), material[:, target], width = width, label='{}'.format(target)) 
                elif target < 10:
                    plt.bar(np.array(range(material.shape[0])), material[:, target], width = width, bottom=np.sum(material[:, tar] for tar in range(target)), label='{}'.format(target))
                else:
                    plt.bar(np.array(range(material.shape[0])), material[:, target], width = width, bottom=np.sum(material[:, tar] for tar in range(10)), color='white', edgecolor='black', hatch='/', label='{}'.format('poison'))
            plt.xlabel('Round')
            plt.ylabel('Data Distribution')
            plt.legend()
            plt.savefig('../../logs/fedstream/data_distribution_{}.png'.format(idx), dpi=200)
            plt.close()
             
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
                label %= 10
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
    
    
    def iid_split_2(self, train_label, num_client):
        client_idcs = [[] for k in range(num_client)]
        for target in range(10):
            target_idcs = np.where(train_label == target)[0]
            num_local_target_idcs = int(len(target_idcs) / num_client)
            for k in range(num_client):
                client_idcs[k].append(np.random.choice(target_idcs, num_local_target_idcs))
                target_idcs = list(set(target_idcs) - set(client_idcs[k][-1]))
        return client_idcs
       
    
    # 下载、预处理、训练集划分和封装、测试集封装
    def dataset_allocation(self):
        if self.dataset_name == 'mnist':
            train_data, train_label, test_data, test_label, trans = self.load_mnist()
        elif self.dataset_name == 'cifar10':
            train_data, train_label, test_data, test_label, trans = self.load_cifar10()
        else:
            raise NotImplementedError('{}'.format(self.dataset_name))

        # 划分数据集
        client_idcs = self.iid_split_2( # 是个矩阵
            train_label=train_label,
            num_client=self.num_client
        )
        
        for k in range(self.num_client):
            local_dataset_list = []
            for target in range(10):
                local_data = train_data[client_idcs[k][target]]
                local_label = train_label[client_idcs[k][target]]
                local_dataset = Local_Dataset(local_data, local_label, trans)
                local_dataset_list.append(local_dataset)
            client = Client(self.dataset_name, local_dataset_list, self.dev, self.net_name, self.learning_rate)
            self.clients.append(client)
        
        # 划分测试集
        self.test_data_list = []
        for target in range(10):
            idcs = np.where(test_label == target)[0]
            tmp_data = test_data[idcs]
            tmp_label = test_label[idcs]
            tmp_dataset = Local_Dataset(tmp_data, tmp_label, trans)
            self.test_data_list.append(tmp_dataset)
        # 3个一组
        # self.new_test_data_list = []
        # self.new_test_data_list.append(ConcatDataset([self.test_data_list[0], self.test_data_list[3], self.test_data_list[6]]))
        # self.new_test_data_list.append(ConcatDataset([self.test_data_list[1], self.test_data_list[4], self.test_data_list[7]]))
        # self.new_test_data_list.append(ConcatDataset([self.test_data_list[2], self.test_data_list[5], self.test_data_list[8], self.test_data_list[9]]))
        # self.test_data_list = self.new_test_data_list