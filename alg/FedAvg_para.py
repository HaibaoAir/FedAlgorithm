### 这是简化后的服务器部分（FedAvg_Server_Static.py）

import os
import sys
import random
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm
import time
from multiprocessing import Process, Manager

sys.path.append("../")
from Clients_Static import Client, run_client_avg_static  # 用新的Clients_Static
from model.mnist import MNIST_LR
from model.fmnist import FMNIST_LR
from model.SVHN import SVHN_CNN
from model.cifar10 import Cifar10_CNN
from model.cifar100 import Cifar100_ResNet50
from model.TinyImageNet import TinyImageNet_ResNet18


class Server(object):
    def __init__(self, args):
        self.args = args
        self.num_client = args.num_client
        self.dataset_name = args.dataset_name
        self.num_class = args.num_class
        self.dev = torch.device(args.device)

        self.net_name = args.net_name
        self.learning_rate = args.learning_rate

        self.init_data_net()

        self.num_round = args.num_round
        self.eval_freq = args.eval_freq

    def init_data_net(self):
        from Clients_Static import Client_Group

        self.client_group = Client_Group(self.args)
        self.test_data_list = self.client_group.test_data_list

        self.net = None
        if self.dataset_name == "mnist":
            self.net = MNIST_LR()
        elif self.dataset_name == "fmnist":
            self.net = FMNIST_LR()
        elif self.dataset_name == "cifar10":
            self.net = Cifar10_CNN()
        elif self.dataset_name == "svhn":
            self.net = SVHN_CNN()
        elif self.dataset_name == "cifar100":
            self.net = Cifar100_ResNet50()
        elif self.dataset_name == "tinyimagenet":
            self.net = TinyImageNet_ResNet18()
        else:
            raise NotImplementedError(self.dataset_name)

        self.net.to(self.dev)

    def run(self):
        global_parameter = {k: v.cpu() for k, v in self.net.state_dict().items()}

        global_loss_list = []
        accuracy_list = []

        for t in tqdm(range(self.num_round)):
            manager = Manager()
            return_dict = manager.dict()
            processes = []

            for k in range(self.num_client):
                client = self.client_group.build_client(k)
                p = Process(
                    target=run_client_avg_static,
                    args=(client, t, k, global_parameter, return_dict),
                )
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            next_global_parameter = {
                k: torch.zeros_like(v, dtype=torch.float32)
                for k, v in global_parameter.items()
            }
            global_loss = 0

            for k in range(self.num_client):
                local_parameter, local_loss = return_dict[k]
                for key in next_global_parameter:
                    next_global_parameter[key] += (
                        1 / self.num_client
                    ) * local_parameter[key]
                global_loss += (1 / self.num_client) * local_loss

            global_parameter = next_global_parameter
            global_loss_list.append(global_loss.item())

            if t % self.eval_freq == 0 or t == self.num_round - 1:
                self.net.load_state_dict(global_parameter)
                acc = self.evaluate()
                accuracy_list.append(acc)

        return global_loss_list, accuracy_list

    def evaluate(self):
        correct = 0
        total = 0
        self.net.eval()
        with torch.no_grad():
            test_loader = DataLoader(
                ConcatDataset(self.test_data_list), batch_size=1024, shuffle=False
            )
            for batch in test_loader:
                data, label = batch
                data = data.to(self.dev)
                label = label.to(self.dev)
                pred = self.net(data)
                pred = torch.argmax(pred, dim=1)
                correct += (pred == label).sum().item()
                total += label.shape[0]
        return correct / total
