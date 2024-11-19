import sys
sys.path.append('../..')
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

from clients import Client_Group
from model.mnist import MNIST_Linear, MNIST_CNN
from model.cifar import Cifar10_CNN

args = {
    'num_client': 3,
    'num_sample': 10,
    'dataset': 'mnist',
    'is_iid': 0,
    'alpha': 0.5,
    'model': 'cnn',
    'learning_rate': 0.01,
    'num_round': 5,
    'num_epoch': 1,
    'batch_size': 32,
    'eval_freq': 1,
}

class Server(object):
    def __init__(self, params):
        # 客户端初始化
        self.dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
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
        self.scales = self.client_group.scales
        self.thetas = [item / sum(self.scales) for item in self.scales]
        self.test_dataloader = self.client_group.test_dataloader
        print(self.thetas)
        
        # 定义net
        self.net = None
        if self.dataset_name == 'mnist':
            if args['model'] == 'linear': 
                self.net = MNIST_Linear()
            elif args['model'] == 'cnn':
                self.net = MNIST_CNN()
            else:
                raise NotImplementedError('{}'.format(args['model']))
        if self.dataset_name == 'cifar10':
            if args['model'] == 'cnn':
                self.net = Cifar10_CNN()
            else:
                raise NotImplementedError('{}'.format(args['model']))
        self.net.to(self.dev)
        # 定义loss function           
        self.criterion = F.cross_entropy # 交叉熵：softmax + NLLLoss 参考知乎 
        
        # 定义optimizer
        lr = args['learning_rate']
        self.optim = torch.optim.SGD(self.net.parameters(), lr)
        
        self.num_round = args['num_round']
        self.num_epoch = args['num_epoch']
        self.batch_size = args['batch_size']
        self.eval_freq = args['eval_freq']
        
    def run(self):
        self.global_parameter = {}
        for key, var in self.net.state_dict().items():
            self.global_parameter[key] = var.clone()
        
        accuracy_list = []
            
        for round in tqdm(range(self.num_round)):
            lst = []
            next_global_parameter = {}
            for idc in range(self.num_client):
                local_parameter = self.client_group.clients[idc].local_update(self.num_epoch,
                                                                              self.batch_size,
                                                                              self.net,
                                                                              self.criterion,
                                                                              self.optim,
                                                                              self.global_parameter,
                                                                              idc)
                lst.append(local_parameter)
                for item in local_parameter.items():
                    if item[0] not in next_global_parameter.keys():
                        next_global_parameter[item[0]] = self.thetas[idc] * item[1]
                    else:
                        next_global_parameter[item[0]] += self.thetas[idc] * item[1]
            self.global_parameter = next_global_parameter
            print('server:', self.global_parameter['fc2.bias'])
            print('client 0:', lst[0]['fc2.bias'])
            print('client 1:', lst[1]['fc2.bias'])
            
            if self.num_round % self.eval_freq == 0:    
                correct = 0
                total = 0
                with torch.no_grad():
                    for batch in self.test_dataloader:
                        data, label = batch
                        data = data.to(self.dev)
                        label = label.to(self.dev)
                        pred = self.net(data) # [batch_size， 10]，输出的是概率
                        pred = torch.argmax(pred, dim=1)
                        correct += (pred == label).sum().item()
                        total += label.shape[0]
                acc = correct / total
                accuracy_list.append(acc)
        
        with open('../../logs/fedavg/accuracy.txt', 'a') as file:
            file.write('{}\n'.format(time.asctime()))
            for accuracy in accuracy_list:
                file.write('{:^7.5f} '.format(accuracy))
            file.write('\n')
        
        plt.title('accuracy')
        plt.plot(np.arange(0, self.num_round, self.eval_freq), accuracy_list)
        plt.savefig('../../logs/fedavg/accuracy.jpg')
        
        
server = Server(args)   
server.run()      