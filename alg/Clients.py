from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, Subset, ConcatDataset, DataLoader  # 三件套
import random
from copy import deepcopy

import math

import sys

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


class Shuffle_Dataset(Dataset):
    def __init__(
        self,
        local_data,
        local_label,
    ):
        self.local_data = local_data
        self.local_label = local_label

    def __getitem__(self, index):
        data = self.local_data[index]
        label = self.local_label[index] + 10
        return data, label

    def __len__(self):
        return len(self.local_label)


# 背景：
# 一个是默认学习新任务，想办法不忘记旧任务，要用过去所有任务做测试集
# 一个是默认不会更新数据集，想办法让他更新，要跟上潮流，用实时数据集做测试集

# 背景是不是该改一下，theta解释为持续学习数据回放的程度。理论推导就应该改成回放数据量与模型表现的关系+回放的存储成本

# theta+delta

# mislabel+用全类测试集 确实体现不出分布的变化，但是确实能体现出theta顾量还是顾新鲜度的权衡[底层逻辑是theta大，保留的多，mis+全类下太大错的比例多，影响；太小就可惜了那么多正确的全类]
# 类增量增[1234]+防止遗忘+用全类测试集 确实和不更新的比好，因为类全嘛。但是theta肯定越大越好。此外类用完了怎么办[增强]
# 类增量增+防止遗忘+mislabel

# 最应该的是类增量增[1234，要高达60%]+实时测试集[极端实时；全类，全类主次之分，测试集类增，测试集类增且主次之分；不行，baseline会震荡]，这样theta也不能够发挥作用[theta的本质就是保留一堆旧数据，极端实时数据集任何旧的没屁用，但是不那么极端的呢，不那么极端的，在新数据收集差不多的情况下，全保留的话模型肯定稍微偏旧的一些新的差点，不保留则相反，总体来说呢整体一定要特别偏爱新数据才能抗衡。也就是有但不能太多]
## 小点的模型，还有观察一下数据
# 时间敏感度有点可惜，怎么用上呢[其实不用也行，只是理论看变化，不用实验]


# 测试集是分布数量固定的，就在那里的，人们就是等训练结束后拿过来用。而不是随去随用，这是多任务持续学习场景。
# 验证theta的最优性的时候，测试集全类全量，iid[123..9, 123..9]的形式添加数据验证，但是有问题，每次加全部训练，theta肯定越大越好
# 和基线比的时候，测试集全类全量，non-iid[1,12,123,1234]的形式添加数据验证
# non-iid是什么情况
class Client(object):
    def __init__(
        self,
        dataset_name,
        dataset_list,
        num_class,
        num_init_class,
        dirichlet,
        dev,
        net_name,
        learning_rate,
    ):
        self.dataset_name = dataset_name
        self.datasource_list = dataset_list  # 数据源，是每个类的分类
        self.num_class = num_class
        self.num_init_class = num_init_class
        self.dirichlet = dirichlet

        self.dev = dev
        self.data = None
        self.global_parameters = None  # 下发的全局模型
        self.local_parameters = None  # 上传的本地模型

        self.net = None
        if self.dataset_name == "mnist":
            if net_name == "lr":
                self.net = MNIST_LR()
            elif net_name == "mlp":
                self.net = MNIST_MLP()
            elif net_name == "cnn":
                self.net = MNIST_CNN()
            else:
                raise NotImplementedError("{}".format(net_name))
        elif self.dataset_name == "fmnist":
            if net_name == "lr":
                self.net = FMNIST_LR()
            elif net_name == "mlp":
                self.net = FMNIST_MLP()
            elif net_name == "cnn":
                self.net = FMNIST_CNN()
            else:
                raise NotImplementedError("{}".format(net_name))
        elif self.dataset_name == "cifar10":
            if net_name == "cnn":
                self.net = Cifar10_CNN()
            else:
                raise NotImplementedError("{}".format(net_name))
        elif self.dataset_name == "svhn":
            if net_name == "cnn":
                self.net = SVHN_CNN()
            else:
                raise NotImplementedError("{}".format(net_name))
        elif self.dataset_name == "cifar100":
            if net_name == "cnn":
                self.net = Cifar100_ResNet18()
            else:
                raise NotImplementedError("{}".format(net_name))
        elif self.dataset_name == "tinyimagenet":
            if net_name == "cnn":
                self.net = TinyImageNet_ResNet18()
            else:
                raise NotImplementedError("{}".format(net_name))
        else:
            raise NotImplementedError("{}".format(net_name))
        self.net.to(self.dev)
        self.criterion = F.cross_entropy  # 交叉熵：softmax + NLLLoss 参考知乎
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
            label_list.append((label + 1) % 10)
        poison_data = Shuffle_Dataset(data_list, label_list)
        self.data = ConcatDataset([health_data, poison_data])

    def init_data(self, t, k, datasize):
        true_num_class = math.ceil(self.num_init_class * self.dirichlet)
        independent_size = datasize // true_num_class
        independent_class = np.random.choice(
            range(self.num_init_class), true_num_class, replace=False
        )
        print(true_num_class, independent_size, independent_class)
        for tau in independent_class:
            idcs = [idc for idc in range(len(self.datasource_list[tau]))]
            used = random.sample(idcs, independent_size)
            # print('left:{}, increment:{}'.format(len(idcs),increment))
            for item in used:
                idcs.remove(item)
            newdata = Subset(self.datasource_list[tau], used)
            self.data = (
                ConcatDataset([self.data, newdata]) if self.data != None else newdata
            )
            self.datasource_list[tau] = Subset(self.datasource_list[tau], idcs)

    def discard_data(self, theta):
        size = int(len(self.data) * (1 - theta))
        idcs = [idc for idc in range(len(self.data))]
        discard = random.sample(idcs, size)
        for item in discard:
            idcs.remove(item)
        self.data = Subset(self.data, idcs)

    def collect_data(self, t, k, increment):
        # t=0阶段用的是初始类别
        # 从t=1开始用新类
        if t + self.num_init_class < self.num_class:
            true_num_class = math.ceil((t + self.num_init_class) * self.dirichlet)
            independent_size = increment // true_num_class
            independent_class = np.random.choice(
                range(t + self.num_init_class), true_num_class, replace=False
            )
            for tau in independent_class:
                idcs = [idc for idc in range(len(self.datasource_list[tau]))]
                used = random.sample(idcs, independent_size)
                # print('left:{}, increment:{}'.format(len(idcs),increment))
                for item in used:
                    idcs.remove(item)
                newdata = Subset(self.datasource_list[tau], used)
                self.data = (
                    ConcatDataset([self.data, newdata])
                    if self.data != None
                    else newdata
                )
                self.datasource_list[tau] = Subset(self.datasource_list[tau], idcs)
        else:
            true_num_class = math.ceil((self.num_class) * self.dirichlet)
            independent_size = increment // true_num_class
            independent_class = np.random.choice(
                range(self.num_class), true_num_class, replace=False
            )
            for tau in independent_class:
                idcs = [idc for idc in range(len(self.datasource_list[tau]))]
                used = random.sample(idcs, independent_size)
                # print('left:{}, increment:{}'.format(len(idcs),increment))
                for item in used:
                    idcs.remove(item)
                newdata = Subset(self.datasource_list[tau], used)
                self.data = (
                    ConcatDataset([self.data, newdata])
                    if self.data != None
                    else newdata
                )
                self.datasource_list[tau] = Subset(self.datasource_list[tau], idcs)

    def local_update(
        self,
        idx,
        t,
        k,
        sigma,
        num_epoch,
        batch_size,
        global_parameters,
        theta,
        increment_next,
        datasize,
        poison_sigma,
    ):

        # 收集数据
        ## 初始化：类多，量大，照顾到后面有的不更新
        if t == 0:
            datasize = int(datasize)
            self.init_data(t, k, datasize)  # t=0用0-7
            # self.poison(poison_sigma) # 本身的有毒性
            # print('init_data_1:{}, init_data_2:{}, increment_next:{}'.format(datasize, len(self.data), increment_next))
        ## 每轮增一类，用完就全量
        else:
            # self.poison(poison_sigma) # 后续的有毒性
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

        width = 0.4
        if k == 0:
            # 数据分布
            count_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
            for item in self.data:
                label = item[1].item()
                count_dict[label] += 1
            print(count_dict)
            self.count_matrix.append(list(count_dict.values()))
            material = np.array(self.count_matrix)
            a = np.sum(material[:, :-1], axis=1)
            b = material[:, -1]
            plt.bar(
                np.array(range(material.shape[0])),
                a,
                width=width,
                color="C0",
                edgecolor="black",
                label="{}".format("valid_data"),
            )
            plt.bar(
                np.array(range(material.shape[0])),
                b,
                width=width,
                bottom=a,
                color="white",
                edgecolor="black",
                hatch="/",
                label="{}".format("stale_data"),
            )
            plt.xlabel("Round")
            plt.ylabel("Data Distribution")
            plt.legend()
            plt.savefig(
                "../logs/fedstream/data_distribution_{}.png".format(idx), dpi=200
            )
            plt.close()

        # 训练
        loss_list = []
        self.net.load_state_dict(global_parameters, strict=True)
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
                self.optim.step()
        new_loss = sum(loss_list) / len(loss_list)

        self.local_parameters = deepcopy(self.net.state_dict())
        return self.local_parameters, new_loss

    def local_update_prox(
        self,
        idx,
        t,
        k,
        sigma,
        num_epoch,
        batch_size,
        global_parameters,
        theta,
        increment_next,
        datasize,
        poison_sigma,
        init_num_class,
        mu,
    ):

        # 收集数据
        ## 初始化：类多，量大，照顾到后面有的不更新
        if t == 0:
            datasize = int(datasize)
            self.init_data(t, k, datasize, init_num_class)  # t=0用0-7
        else:
            self.discard_data(theta)
            increment = int(datasize - len(self.data))  # t=1收集好了8
            self.collect_data(t, k, increment, init_num_class)

        if t == 0:
            self.count_matrix = []
        width = 0.4
        if k == 0:
            # 数据分布
            count_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
            for item in self.data:
                label = item[1].item()
                count_dict[label] += 1
            print(count_dict)
            self.count_matrix.append(list(count_dict.values()))
            material = np.array(self.count_matrix)
            a = np.sum(material[:, :-1], axis=1)
            b = material[:, -1]
            plt.bar(
                np.array(range(material.shape[0])),
                a,
                width=width,
                color="C0",
                edgecolor="black",
                label="{}".format("valid_data"),
            )
            plt.bar(
                np.array(range(material.shape[0])),
                b,
                width=width,
                bottom=a,
                color="white",
                edgecolor="black",
                hatch="/",
                label="{}".format("stale_data"),
            )
            plt.xlabel("Round")
            plt.ylabel("Data Distribution")
            plt.legend()
            plt.savefig(
                "../logs/fedstream/data_distribution_{}.png".format(idx), dpi=200
            )
            plt.close()

        # 训练
        loss_list = []
        self.net.load_state_dict(global_parameters, strict=True)
        dataloader = DataLoader(self.data, batch_size=batch_size, shuffle=True)
        for epoch in range(num_epoch):
            for batch in dataloader:
                data, label = batch
                data = data.to(self.dev)
                label = label.to(self.dev)
                pred = self.net(data)
                loss = self.criterion(pred, label)
                loss_list.append(loss.cpu().detach())

                prox_term = 0
                for w, w_g in zip(self.net.parameters(), global_parameters):
                    # w_g应该是固定的，避免对它的梯度更新。
                    # detach在修改loss的时候要注意
                    # clone在服务器与客户端之间传递要注意
                    prox_term += ((w - w_g.detach()) ** 2).sum()
                loss += (mu / 2) * prox_term

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
        new_loss = sum(loss_list) / len(loss_list)

        self.local_parameters = deepcopy(self.net.state_dict())
        return self.local_parameters, new_loss


class Client_Group(object):
    def __init__(
        self,
        dev,
        num_client,
        dataset_name,
        num_class,
        num_init_class,
        dirichlet,
        net_name,
        learning_rate,
    ):
        self.dev = dev
        self.num_client = num_client
        self.clients = []
        self.scales = []
        self.dataset_name = dataset_name
        self.num_class = num_class
        self.num_init_class = num_init_class
        self.dirichlet = dirichlet
        self.net_name = net_name
        self.learning_rate = learning_rate
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
                    np.random.choice(target_idcs, num_local_target_idcs)
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
            client = Client(
                self.dataset_name,
                local_dataset_list,
                self.num_class,
                self.num_init_class,
                self.dirichlet,
                self.dev,
                self.net_name,
                self.learning_rate,
            )
            self.clients.append(client)

        # 划分测试集
        self.test_data_list = []
        for target in range(self.num_class):
            idcs = np.where(test_label == target)[0]
            global_data = test_data[idcs]
            global_label = test_label[idcs]
            global_dataset = Local_Dataset(global_data, global_label, test_trans)
            self.test_data_list.append(global_dataset)
