import sys
sys.path.append('../')
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

from util.Client import Client_Group
from model.adult import Adult_MLP
from model.mnist import MNIST_MLP, MNIST_CNN
from model.fmnist import FMNIST_CNN


class Server(object):
    def __init__(self, args):
        # 客户端初始化
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset_name = args['dataset']
        self.is_iid = args['is_iid']
        self.alpha = args['alpha']
        self.num_client = args['num_client']
        self.num_sample = args['num_sample']
        self.client_group = Client_Group(self.dev,
                                         self.num_client,
                                         self.dataset_name,
                                         self.is_iid,
                                         self.alpha)
        self.scale_list = self.client_group.scales
        self.weight_list = [self.scale_list[idx] / sum(self.scale_list) for idx in range(self.num_client)]
        self.test_dataloader = self.client_group.test_dataloader
        # print(self.thetas)
        
        # 定义网络
        self.net = None
        if self.dataset_name == 'adult':
            if args['model'] == 'mlp':
                self.net = Adult_MLP(101)
                self.net.to(self.dev)
                self.criterion = nn.BCELoss() # 二分类交叉熵损失函数
                self.optimizer = torch.optim.Adam(self.net.parameters(), args['learning_rate'])
            else:
                raise NotImplementedError('{}'.format(args['model']))            
            
        elif self.dataset_name == 'mnist':
            if args['model'] == 'mlp': 
                self.net = MNIST_MLP()
                self.net.to(self.dev)
                self.criterion = F.cross_entropy
                self.optimizer = torch.optim.SGD(self.net.parameters(), args['learning_rate'])
            elif args['model'] == 'cnn':
                self.net = MNIST_CNN()
                self.net.to(self.dev)
                self.criterion = F.cross_entropy
                self.optimizer = torch.optim.SGD(self.net.parameters(), args['learning_rate'])
            else:
                raise NotImplementedError('{}'.format(args['model']))
        elif self.dataset_name == 'fmnist':
            if args['model'] == 'cnn':
                self.net = FMNIST_CNN()
                self.net.to(self.dev)
                self.criterion = F.cross_entropy
                self.optimizer = torch.optim.SGD(self.net.parameters(), args['learning_rate'])
            else:
                raise NotImplementedError('{}'.format(args['model']))
            
        else:
            raise NotImplementedError('{}'.format(self.dataset_name))
        
        
        self.num_round = args['num_round']
        self.num_epoch = args['num_epoch']
        self.batch_size = args['batch_size']
        self.eval_freq = args['eval_freq']
        
    def run(self):
        pass