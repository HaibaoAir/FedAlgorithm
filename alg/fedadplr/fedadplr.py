import os
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
    'num_client': 10,
    'num_sample': 10,
    'dataset': 'mnist',
    'is_iid': 1,
    'alpha': 1.0,
    'model': 'cnn',
    'learning_rate': 0.01,
    'min_learning_rate': 1e-6,
    'num_round': 20,
    'num_epoch': 1,
    'batch_size': 32,
    'eval_freq': 1,
    'eps': 1e-2,
    'max_iteration': 50,
    'weight': 1e-4
}

class Server(object):
    def __init__(self, args):
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
                                         self.alpha,)
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
        self.net = self.net.to(self.dev)
        
        # 定义loss function           
        self.criterion = F.cross_entropy # 交叉熵：softmax + NLLLoss 参考知乎 
        
        # 定义optimizer
        self.lr = args['learning_rate']
        self.min_lr = args['min_learning_rate']
        opt_params_list = []
        for name, params in self.net.named_parameters():
            opt_params = {
                'params': params,
                'params_name': name,
                'lr': self.lr
            }
            opt_params_list.append(opt_params)
        self.optim = torch.optim.SGD(opt_params_list)
        # print('len of optim.param_groups:', len(self.optim.param_groups))
        # print(self.optim.param_groups[0])

        self.num_round = args['num_round']
        self.num_epoch = args['num_epoch']
        self.batch_size = args['batch_size']
        self.eps = args['eps']
        self.max_iteration = args['max_iteration']
        self.weight = args['weight']
        
    def run(self):
        torch.manual_seed(2021)
        
        self.global_parameter_list = []
        for round in range(self.num_round + 1):
            for layer in self.net.modules():
                if type(layer) == nn.Conv2d:
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.constant_(layer.bias, 0)
                elif type(layer) == nn.Linear:
                    nn.init.normal_(layer.weight, 0, 0.01)
                    nn.init.constant_(layer.bias, 0)
            
            global_parameter = {}
            for key, var in self.net.state_dict().items():
                global_parameter[key] = var.clone()
            self.global_parameter_list.append(global_parameter)
        
        eps_list = []
        for round in range(self.num_round):
            eps = 0
            for key in self.global_parameter_list[round]:
                eps += torch.sum(torch.abs(self.global_parameter_list[round][key].reshape(-1) - self.global_parameter_list[round + 1][key].reshape(-1)))
            eps_list.append(eps)
        print(eps_list[0])
            
        indicator = 0
        for iteration in range(self.max_iteration):
            next_global_parameter_list = [self.global_parameter_list[0]]         
            next_lr_matrix = []
            for round in range(self.num_round):
                print('iteration: {}, round: {}'.format(iteration, round))
                next_global_parameter = {}
                next_lr_list = []
                for idc in range(self.num_client):
                    local_parameter, lr = self.client_group.clients[idc].pretrain_local_update(self.num_epoch,
                                                                                            self.batch_size,
                                                                                            self.net,
                                                                                            self.criterion,
                                                                                            self.optim,
                                                                                            self.global_parameter_list,
                                                                                            self.min_lr,
                                                                                            self.weight,
                                                                                            round)

                    next_lr_list.append(lr)
                    for item in local_parameter.items():
                        if item[0] not in next_global_parameter.keys():
                            next_global_parameter[item[0]] = self.thetas[idc] * item[1]
                        else:
                            next_global_parameter[item[0]] += self.thetas[idc] * item[1]
                
                next_lr_matrix.append(next_lr_list)
                next_global_parameter_list.append(next_global_parameter)
            
            eps_list = []
            for round in range(1, self.num_round + 1):
                eps = 0
                for key in self.global_parameter_list[round]:
                    eps += torch.sum(torch.abs(self.global_parameter_list[round][key].reshape(-1) - next_global_parameter_list[round][key].reshape(-1)))
                eps_list.append(eps)
                    
            for round in range(1, self.num_round + 1):
                if eps_list[round] > self.eps:
                    self.global_parameter_list = next_global_parameter_list
                    indicator = 1
                    print('not reach the accuracy')
                    break
            
            if indicator == 0:
                print('!!!reach the accuracy!!!')
                break
            
            with open('../../logs/fedadplr/lr.txt', 'a') as file:
                file.write('{}\n'.format(time.asctime()))
                file.write('iteration {}: '.format(iteration))
                for lr in [next_lr_matrix[i][:1] for i in range(self.num_round)]:
                    print(lr)
                    if isinstance(lr[0], float):
                        file.write('{:^7.6f} '.format(lr[0]))
                    else:
                        file.write('{:^7.6f} '.format(lr[0].item()))
                file.write('\n')
                
            with open('../../logs/fedadplr/eps.txt', 'a') as file:
                file.write('{}\n'.format(time.asctime()))
                file.write('iteration {}: '.format(iteration))
                for eps in eps_list:
                    file.write('{:^7.2f} '.format(eps.item()))
                file.write('\n')
                
            self.eval(iteration, self.global_parameter_list)
            
    
    
    def eval(self, iteration, global_parameter_list):
        accuracy_list = []
        for global_parameter in global_parameter_list:
            self.net.load_state_dict(global_parameter, strict=True) 
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
                accuracy = correct / total
                accuracy_list.append(accuracy)
        
        plt.title('iteration'.format(iteration))
        plt.plot(np.arange(0, self.num_round + 1), accuracy_list)
        plt.savefig('../../logs/fedadplr/{}.jpg'.format(iteration))

    
server = Server(args)   
server.run()   


    # def run(self):
    #     loss_list = []
    #     acc_list = []
    #     # 全局迭代
    #     for round in tqdm(range(self.num_round)):
    #         # 采样
    #         idcs = random.sample(range(self.num_client), self.num_sample)
    #         # 本地更新
    #         next_global_parameter = {}
    #         for idc in idcs:
    #             local_parameter = self.client_group.clients[idc].local_update(self.num_epoch,
    #                                                                           self.batch_size,
    #                                                                           self.net,
    #                                                                           self.criterion,
    #                                                                           self.optim,
    #                                                                           self.global_parameter)
    #             for item in local_parameter.items():
    #                 if item[0] not in next_global_parameter.keys():
    #                     next_global_parameter[item[0]] = self.thetas[idc] * item[1]
    #                 else:
    #                     next_global_parameter[item[0]] += self.thetas[idc] * item[1]
    #         self.global_parameter = next_global_parameter
            
    #         if self.num_round % args['eval_freq'] == 0:    
    #             correct = 0
    #             total = 0
    #             with torch.no_grad():
    #                 for batch in self.test_dataloader:
    #                     data, label = batch
    #                     pred = self.net(data) # [batch_size， 10]，输出的是概率
    #                     pred = torch.argmax(pred, dim=1)
    #                     correct += (pred == label).sum().item()
    #                     total += label.shape[0]
    #             acc = correct / total
    #             acc_list.append(acc)
        
    #             plt.plot(np.arange(0, self.num_round, args['eval_freq']), acc_list)
    #             plt.show()
        
    #     print(acc_list)   