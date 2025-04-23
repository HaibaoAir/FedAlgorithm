import os
import sys

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Real
from tqdm import tqdm
import time

sys.path.append("../")
from Clients import Client_Group
from model.mnist import MNIST_LR
from model.fmnist import FMNIST_LR
from model.SVHN import SVHN_CNN
from model.cifar10 import Cifar10_CNN
from model.cifar100 import Cifar100_ResNet50
from model.TinyImageNet import TinyImageNet_ResNet18


class Server(object):
    def __init__(self, args):
        self.args = args
        # 服务器持有测试集和测试用网络
        self.num_client = args.num_client
        self.num_sample = args.num_sample
        self.dataset_name = args.dataset_name
        self.num_class = args.num_class
        self.dev = torch.device(args.device)

        self.net_name = args.net_name
        self.learning_rate = args.learning_rate
        self.init_data_net()

        # 再加上通讯轮数
        self.num_round = args.num_round
        self.eval_freq = args.eval_freq

    def init_data_net(self):
        self.client_group = Client_Group(self.args)
        self.test_data_list = self.client_group.test_data_list

        # 定义net
        self.net = None
        if self.dataset_name == "mnist":
            if self.net_name == "lr":
                self.net = MNIST_LR()
            else:
                raise NotImplementedError("{}".format(self.net_name))
        elif self.dataset_name == "fmnist":
            if self.net_name == "lr":
                self.net = FMNIST_LR()
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
            if self.net_name == "resnet":
                self.net = Cifar100_ResNet50()
            else:
                raise NotImplementedError("{}".format(self.net_name))
        elif self.dataset_name == "tinyimagenet":
            if self.net_name == "resnet":
                self.net = TinyImageNet_ResNet18()
            else:
                raise NotImplementedError("{}".format(self.net_name))
        else:
            raise NotImplementedError("{}".format(self.net_name))

        self.net.to(self.dev)

    def run(self, theta, data_matrix):
        # 初始化数据和网络
        self.init_data_net()
        global_parameter = {}
        for key, var in self.net.state_dict().items():
            global_parameter[key] = var.clone()
        # 计算聚合权重
        rate_matrix = np.stack(
            [data_matrix[:, t] / sum(data_matrix[:, t]) for t in range(self.num_round)]
        )
        rate_matrix = torch.from_numpy(rate_matrix).T
        # 记录loss与acc
        global_loss_list = []
        accuracy_list = []
        # 训练
        for t in tqdm(range(self.num_round)):
            a = time.time()
            next_global_parameter = {}
            global_loss = 0
            for k in range(self.num_client):
                result = self.client_group.clients[k].local_update_avg(
                    t,
                    k,
                    theta,
                    data_matrix[k][t],
                    global_parameter,
                )
                local_parameter = result[0]
                for item in local_parameter.items():
                    if item[0] not in next_global_parameter.keys():
                        next_global_parameter[item[0]] = (
                            rate_matrix[k][t] * item[1].clone()
                        )
                    else:
                        next_global_parameter[item[0]] += (
                            rate_matrix[k][t] * item[1].clone()
                        )
                # print(
                #     local_parameter["conv_block.0.weight"].data_ptr()
                #     == next_global_parameter["conv_block.0.weight"].data_ptr()
                # )
                # exit(0)
                # 不会污染第一个本地模型

                local_loss = result[1]
                global_loss += rate_matrix[k][t] * local_loss

            # 求global_parameters和global_loss_list
            global_parameter = next_global_parameter
            global_loss_list.append(global_loss)
            b = time.time()
            print("round time = {}".format(b - a))
            # 验证
            if t % self.eval_freq == 0 or t == self.num_round - 1:
                correct = 0
                total = 0
                self.net.load_state_dict(global_parameter)
                with torch.no_grad():
                    # 固定哦
                    test_dataloader = DataLoader(
                        ConcatDataset(self.test_data_list),
                        batch_size=1024,
                        shuffle=False,
                    )
                    for batch in test_dataloader:
                        data, label = batch
                        # print(label)
                        data = data.to(self.dev)
                        label = label.to(self.dev)
                        pred = self.net(data)  # [batch_size， 10]，输出的是概率
                        pred = torch.argmax(pred, dim=1)
                        correct += (pred == label).sum().item()
                        total += label.shape[0]
                acc = correct / total
                accuracy_list.append(acc)

        return global_loss_list, accuracy_list
