import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, Subset, ConcatDataset, DataLoader
from matplotlib import pyplot as plt
import random
import math
from copy import deepcopy
import time

sys.path.append("../")
from model.mnist import MNIST_LR, MNIST_MLP, MNIST_CNN
from model.fmnist import FMNIST_LR, FMNIST_MLP, FMNIST_CNN
from model.cifar10 import Cifar10_CNN
from model.SVHN import SVHN_CNN
from model.cifar100 import Cifar100_ResNet18
from model.TinyImageNet import TinyImageNet_ResNet18

seed = 10
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


class Local_Dataset(Dataset):
    def __init__(self, local_data, local_label, trans):
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
    def __init__(self, dataset_list, args):
        self.datasource_list = dataset_list  # 数据源，是每个类的分类
        self.dataset_name = args.dataset_name
        self.num_class = args.num_class
        self.init_num_class = args.init_num_class
        self.dirichlet = args.dirichlet
        self.own_class = None
        self.data = None

        self.dev = torch.device(args.device)
        self.num_epoch = args.num_epoch
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.dev = torch.device(args.device)

        self.mu = args.mu

        self.net = None
        self.net_name = args.net_name
        if self.dataset_name == "mnist":
            if self.net_name == "lr":
                self.net = MNIST_LR()
            elif self.net_name == "mlp":
                self.net = MNIST_MLP()
            elif self.net_name == "cnn":
                self.net = MNIST_CNN()
            else:
                raise NotImplementedError("{}".format(self.net_name))
        elif self.dataset_name == "fmnist":
            if self.net_name == "lr":
                self.net = FMNIST_LR()
            elif self.net_name == "mlp":
                self.net = FMNIST_MLP()
            elif self.net_name == "cnn":
                self.net = FMNIST_CNN()
            else:
                raise NotImplementedError("{}".format(self.net_name))
        elif self.dataset_name == "cifar10":
            if self.net_name == "cnn":
                self.net = Cifar10_CNN()
            else:
                raise NotImplementedError("{}".format(self.net_name))
        elif self.dataset_name == "svhn":
            if self.net_name == "cnn":
                self.net = SVHN_CNN()
            else:
                raise NotImplementedError("{}".format(self.net_name))
        elif self.dataset_name == "cifar100":
            if self.net_name == "cnn":
                self.net = Cifar100_ResNet18()
            else:
                raise NotImplementedError("{}".format(self.net_name))
        elif self.dataset_name == "tinyimagenet":
            if self.net_name == "cnn":
                self.net = TinyImageNet_ResNet18()
            else:
                raise NotImplementedError("{}".format(self.net_name))
        else:
            raise NotImplementedError("{}".format(self.net_name))
        self.net.to(self.dev)
        self.criterion = F.cross_entropy
        # self.optim = torch.optim.SGD(self.net.parameters(), self.learning_rate)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=1e-3, weight_decay=1e-4)

        self.mu = args.mu

    def init_data(self, t, k, datasize):
        self.own_class = np.random.choice(
            range(self.init_num_class),
            math.ceil(self.init_num_class * self.dirichlet),
            replace=False,
        )
        volume = datasize // self.own_class.shape[0]
        print(self.own_class.shape[0], volume, self.own_class)
        for tau in self.own_class:
            idcs = [idc for idc in range(len(self.datasource_list[tau]))]
            used = idcs[:volume]
            idcs = idcs[volume:] + used
            newdata = Subset(self.datasource_list[tau], used)
            self.data = ConcatDataset([self.data, newdata]) if self.data else newdata
            self.datasource_list[tau] = Subset(self.datasource_list[tau], idcs)

    def discard_data(self, theta):
        start_idx = int(len(self.data) * (1 - theta))
        idcs = list(range(start_idx, len(self.data)))
        self.data = Subset(self.data, idcs)

    def collect_data(self, t, k, increment):
        # t=0阶段用的是初始类别
        # 从t=1开始收集6，t=4收集9
        if t + self.init_num_class <= self.num_class:
            if random.random() < self.dirichlet:
                self.own_class = np.append(self.own_class, t + self.init_num_class - 1)
            volume = increment // self.own_class.shape[0]
            print(self.own_class.shape[0], volume, self.own_class)
            for tau in self.own_class:
                idcs = [idc for idc in range(len(self.datasource_list[tau]))]
                used = idcs[:volume]
                idcs = idcs[volume:] + used
                newdata = Subset(self.datasource_list[tau], used)
                self.data = ConcatDataset([self.data, newdata])
                self.datasource_list[tau] = Subset(self.datasource_list[tau], idcs)
        else:
            volume = increment // self.own_class.shape[0]
            print(self.own_class.shape[0], volume, self.own_class)
            for tau in self.own_class:
                idcs = [idc for idc in range(len(self.datasource_list[tau]))]
                used = idcs[:volume]
                idcs = idcs[volume:] + used
                newdata = Subset(self.datasource_list[tau], used)
                self.data = ConcatDataset([self.data, newdata])
                self.datasource_list[tau] = Subset(self.datasource_list[tau], idcs)

    def local_update_avg(
        self,
        t,
        k,
        theta,
        datasize,
        global_parameters,
    ):

        # 收集数据
        ## 初始化：类多，量大，照顾到后面有的不更新
        if t == 0:
            datasize = int(datasize)
            self.init_data(t, k, datasize)  # t=0用0-7
        ## 每轮增一类，用完就全量
        else:
            a = time.time()
            # print('origin', len(self.data))
            self.discard_data(theta)
            # print('decay', len(self.data))
            # increment = int(datasize - theta * len(self.data)) 串联了呀
            b = time.time()
            increment = int(datasize - len(self.data))  # t=1收集好了8
            self.collect_data(t, k, increment)
            # print('accumu', len(self.data))
            # print('res_data_1:{}, res_data_2:{}, theta:{}, increment:{}, increment_next:{}'.format(datasize, len(self.data), theta, increment, increment_next))
            c = time.time()
            print(
                "client {}: discard {}, collect {}, sum {}".format(
                    k, b - a, c - b, c - a
                )
            )
        if t == 0:
            self.count_matrix = []

        # 训练
        loss_list = []
        self.net.load_state_dict(global_parameters, strict=True)
        dataloader = DataLoader(self.data, batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.num_epoch):
            for batch in dataloader:
                data, label = batch
                data = data.to(self.dev)
                label = label.to(self.dev)
                pred = self.net(data)
                loss = self.criterion(pred, label)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                loss_list.append(loss.item())
        new_loss = sum(loss_list) / len(loss_list)

        return deepcopy(self.net.state_dict()), new_loss

    def local_update_prox(
        self,
        t,
        k,
        theta,
        datasize,
        global_parameters,
    ):

        # 收集数据
        ## 初始化：类多，量大，照顾到后面有的不更新
        if t == 0:
            datasize = int(datasize)
            self.init_data(t, k, datasize)  # t=0用0-7
        ## 每轮增一类，用完就全量
        else:
            # print('origin', len(self.data))
            self.discard_data(theta)
            # print('decay', len(self.data))
            # increment = int(datasize - theta * len(self.data)) 串联了呀
            increment = int(datasize - len(self.data))  # t=1收集好了8
            self.collect_data(t, k, increment)
            # print('accumu', len(self.data))
            # print('res_data_1:{}, res_data_2:{}, theta:{}, increment:{}, increment_next:{}'.format(datasize, len(self.data), theta, increment, increment_next))

        if t == 0:
            self.count_matrix = []

        # 训练
        loss_list = []
        self.net.load_state_dict(global_parameters, strict=True)
        dataloader = DataLoader(self.data, batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.num_epoch):
            for batch in dataloader:
                data, label = batch
                data = data.to(self.dev)
                label = label.to(self.dev)
                pred = self.net(data)
                loss = self.criterion(pred, label)

                proximal_term = 0.0
                for name, param in self.net.named_parameters():
                    global_param = global_parameters[name].to(param.device)
                    proximal_term += (param - global_param.detach()).pow(2).sum()

                loss += (self.mu / 2) * proximal_term

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                loss_list.append(loss.item())
        new_loss = sum(loss_list) / len(loss_list)

        return deepcopy(self.net.state_dict()), new_loss

    def local_update_dyn(
        self,
        t,
        k,
        theta,
        datasize,
        global_parameters,
        prev_grads,
        feddyn_alpha,
    ):
        # 收集数据
        ## 初始化：类多，量大，照顾到后面有的不更新
        if t == 0:
            datasize = int(datasize)
            self.init_data(t, k, datasize)  # t=0用0-7
        ## 每轮增一类，用完就全量
        else:
            # print('origin', len(self.data))
            self.discard_data(theta)
            # print('decay', len(self.data))
            # increment = int(datasize - theta * len(self.data)) 串联了呀
            increment = int(datasize - len(self.data))  # t=1收集好了8
            self.collect_data(t, k, increment)
            # print('accumu', len(self.data))
            # print('res_data_1:{}, res_data_2:{}, theta:{}, increment:{}, increment_next:{}'.format(datasize, len(self.data), theta, increment, increment_next))

        # 训练
        par_flat = torch.cat(
            [
                global_parameters[name].detach().to(param.device).view(-1)
                for name, param in self.net.named_parameters()
            ]
        )

        loss_list = []
        self.net.load_state_dict(global_parameters, strict=True)
        dataloader = DataLoader(self.data, batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.num_epoch):
            for batch in dataloader:
                data, label = batch
                data = data.to(self.dev)
                label = label.to(self.dev)
                pred = self.net(data)
                loss = self.criterion(pred, label)

                # 三个变量 curr_params, prev_grads, par_flat
                curr_params = torch.cat(
                    [param.reshape(-1) for param in self.net.parameters()]
                )
                lin_penalty = torch.sum(curr_params * prev_grads)

                norm_penalty = (feddyn_alpha / 2.0) * torch.linalg.norm(
                    curr_params - par_flat, 2
                ) ** 2

                loss = loss - lin_penalty + norm_penalty

                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    parameters=self.net.parameters(), max_norm=10
                )
                self.optim.step()
                loss_list.append(loss.item())

        new_loss = sum(loss_list) / len(loss_list)
        cur_flat = torch.cat(
            [param.detach().reshape(-1) for param in self.net.parameters()]
        )
        prev_grads -= feddyn_alpha * (cur_flat - par_flat)  # ht
        return deepcopy(self.net.state_dict()), new_loss, prev_grads


class Client_Group(object):
    def __init__(self, args):
        self.args = args
        self.dev = torch.device(args.device)
        self.num_client = args.num_client
        self.clients = []
        self.scales = []
        self.dataset_name = args.dataset_name
        self.num_class = args.num_class
        self.init_num_class = args.init_num_class
        self.dirichlet = args.dirichlet
        self.net_name = args.net_name
        self.learning_rate = args.learning_rate
        self.dataset_allocation()

    def load_mnist(self):
        # 下载原始数据集：[60000, 28, 28], tensor + tensor
        train_dataset = torchvision.datasets.MNIST(
            root="../data", download=True, train=True
        )
        test_dataset = torchvision.datasets.MNIST(
            root="../data", download=True, train=False
        )

        # 提取数据和标签
        train_data = train_dataset.data
        test_data = test_dataset.data
        train_label = train_dataset.targets
        test_label = test_dataset.targets

        # 数据增强 + 标准化
        train_trans = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize((0.13065973,), (0.3015038,)),
            ]
        )

        test_trans = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize((0.13065973,), (0.3015038,)),
            ]
        )

        return (
            train_data,
            train_label,
            train_trans,
            test_data,
            test_label,
            test_trans,
        )

    def load_fmnist(self):
        # 下载原始数据集：[60000, 28, 28], tensor + tensor
        train_dataset = torchvision.datasets.FashionMNIST(
            root="../data", download=True, train=True
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root="../data", download=True, train=False
        )

        # 提取数据和标签
        train_data = train_dataset.data
        test_data = test_dataset.data
        train_label = train_dataset.targets
        test_label = test_dataset.targets

        # 数据增强 + 标准化
        train_trans = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5,), (0.5,)
                ),  # Fashion MNIST是灰度图像，通道数为1
            ]
        )

        test_trans = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        return (
            train_data,
            train_label,
            train_trans,
            test_data,
            test_label,
            test_trans,
        )

    def load_cifar10(self):

        # 下载原始数据集：[50000, 32, 32, 3], tensor + list
        train_dataset = torchvision.datasets.CIFAR10(
            root="../data", train=True, download=True
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root="../data", train=False, download=True
        )

        # 提取数据和标签
        train_data = torch.tensor(train_dataset.data).permute(0, 3, 1, 2)
        test_data = torch.tensor(test_dataset.data).permute(0, 3, 1, 2)
        train_label = torch.tensor(train_dataset.targets)
        test_label = torch.tensor(test_dataset.targets)

        # 数据增强 + 预处理
        train_trans = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
                ),
            ]
        )

        test_trans = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
                ),
            ]
        )

        return (
            train_data,
            train_label,
            train_trans,
            test_data,
            test_label,
            test_trans,
        )

    def load_SVHN(self):
        # 下载原始数据集：[73257, 3, 32, 32], tensor + list
        train_dataset = torchvision.datasets.SVHN(
            root="../data", split="train", download=True
        )
        test_dataset = torchvision.datasets.SVHN(
            root="../data", split="test", download=True
        )

        # 提取数据和标签
        train_data = torch.tensor(train_dataset.data)
        train_label = torch.tensor(train_dataset.labels)
        test_data = torch.tensor(test_dataset.data)
        test_label = torch.tensor(test_dataset.labels)

        # SVHN 中 label=10 表示数字 0，需要手动处理一下
        train_label[train_label == 10] = 0
        test_label[test_label == 10] = 0

        # 数据增强 + 标准化
        train_trans = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4377, 0.4438, 0.4728), (0.198, 0.201, 0.197)),
            ]
        )

        test_trans = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize((0.4377, 0.4438, 0.4728), (0.198, 0.201, 0.197)),
            ]
        )

        return (
            train_data,
            train_label,
            train_trans,
            test_data,
            test_label,
            test_trans,
        )

    def load_cifar100(self):

        # 下载原始数据集：[50000, 32, 32, 3], tensor + list
        train_dataset = torchvision.datasets.CIFAR100(
            root="../data", train=True, download=True
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root="../data", train=False, download=True
        )

        # 提取数据和标签
        train_data = torch.tensor(train_dataset.data).permute(0, 3, 1, 2)
        test_data = torch.tensor(test_dataset.data).permute(0, 3, 1, 2)
        train_label = torch.tensor(train_dataset.targets)
        test_label = torch.tensor(test_dataset.targets)

        # 数据增强 + 正则标准化
        train_trans = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
                ),  # 官方推荐 mean/std
            ]
        )

        test_trans = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
                ),
            ]
        )

        return (
            train_data,
            train_label,
            train_trans,
            test_data,
            test_label,
            test_trans,
        )

    def load_tinyimagenet(self):

        # 下载原始数据集：[100000, 3, 64, 64], tensor + list
        data_root = "../data/tiny-imagenet-200"
        train_dataset = torchvision.datasets.ImageFolder(
            root=f"{data_root}/train", transform=None
        )
        test_dataset = torchvision.datasets.ImageFolder(
            root=f"{data_root}/val", transform=None
        )

        # 提取数据和标签
        def extract_data(dataset):
            data_list = []
            label_list = []
            for img, label in dataset:
                data_list.append(transforms.ToTensor()(img))  # [C, 64, 64]
                label_list.append(label)
            data_tensor = torch.stack(data_list)
            label_tensor = torch.tensor(label_list)
            return data_tensor, label_tensor

        train_data, train_label = extract_data(train_dataset)
        test_data, test_label = extract_data(test_dataset)

        # 图像增强 + 归一化（官方 mean/std 不提供，可用 ImageNet）
        train_trans = transforms.Compose(
            [
                transforms.RandomResizedCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # ImageNet 标准
                ),
            ]
        )

        test_trans = transforms.Compose(
            [
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        return (
            train_data,
            train_label,
            train_trans,
            test_data,
            test_label,
            test_trans,
        )

    def split_data(self, train_label, num_client, num_class):
        client_idcs = [[] for k in range(num_client)]
        for target in range(num_class):
            target_idcs = np.where(train_label == target)[0]
            num_local_target_idcs = int(len(target_idcs) / num_client)
            for k in range(num_client):
                client_idcs[k].append(
                    np.random.choice(target_idcs, num_local_target_idcs, replace=False)
                )
                target_idcs = list(set(target_idcs) - set(client_idcs[k][-1]))
        return client_idcs

    # 下载、预处理、训练集划分和封装、测试集封装
    def dataset_allocation(self):
        if self.dataset_name == "mnist":
            train_data, train_label, train_trans, test_data, test_label, test_trans = (
                self.load_mnist()
            )
        elif self.dataset_name == "fmnist":
            train_data, train_label, train_trans, test_data, test_label, test_trans = (
                self.load_fmnist()
            )
        elif self.dataset_name == "cifar10":
            train_data, train_label, train_trans, test_data, test_label, test_trans = (
                self.load_cifar10()
            )
        elif self.dataset_name == "svhn":
            train_data, train_label, train_trans, test_data, test_label, test_trans = (
                self.load_SVHN()
            )
        elif self.dataset_name == "cifar100":
            train_data, train_label, train_trans, test_data, test_label, test_trans = (
                self.load_cifar100()
            )
        elif self.dataset_name == "tinyimagenet":
            train_data, train_label, train_trans, test_data, test_label, test_trans = (
                self.load_tinyimagenet()
            )
        else:
            raise NotImplementedError("{}".format(self.dataset_name))

        # 划分训练集
        client_idcs = self.split_data(  # 是个矩阵
            train_label=train_label,
            num_client=self.num_client,
            num_class=self.num_class,
        )

        for k in range(self.num_client):
            local_dataset_list = []
            for target in range(self.num_class):
                local_data = train_data[client_idcs[k][target]]
                local_label = train_label[client_idcs[k][target]]
                local_dataset = Local_Dataset(local_data, local_label, train_trans)
                local_dataset_list.append(local_dataset)
            client = Client(local_dataset_list, self.args)
            self.clients.append(client)

        # 划分测试集
        self.test_data_list = []
        for target in range(self.num_class):
            idcs = np.where(test_label == target)[0]
            global_data = test_data[idcs]
            global_label = test_label[idcs]
            global_dataset = Local_Dataset(global_data, global_label, test_trans)
            self.test_data_list.append(global_dataset)
