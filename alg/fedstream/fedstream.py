import sys
import time
sys.path.append('../..')
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy

from clients import Client_Group
from model.mnist import MNIST_Linear, MNIST_CNN
from model.cifar import Cifar10_CNN

from sko.PSO import PSO

args = {
    'num_client': 10,
    'num_sample': 10,
    'dataset': 'mnist',
    'is_iid': 0,
    'alpha': 1.0,
    'model': 'cnn',
    'learning_rate': 0.01,
    'num_round': 10,
    'num_epoch': 1,
    'batch_size': 32,
    'eval_freq': 1,
}

seed = 10
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

class Server(object):
    def __init__(self, args):
        # 初始化客户端
        # 初始化网络+参数
        # 初始化损失函数
        # 初始化优化器
        self.init_params(args)
        
        # 训练超参数
        self.num_round = args['num_round']
        self.num_epoch = args['num_epoch']
        self.batch_size = args['batch_size']
        self.eval_freq = args['eval_freq']
        
        # 背景超参数
        self.epsilon_list = [1] * self.num_client #[K] 0-1之间
        self.avg_delta_list = [1] * self.num_client # [K]
        self.alpha_list = [0.5] * self.num_client # 收集数据的价格
        self.beta_list = [0] * self.num_client # 训练数据的价格
        self.theta = 0.5 # 数据丢弃率
        self.eps = 1
        self.fix_max_iter = 50
        
        self.p = 0.5
    
    def init_params(self, args):
        # 初始化客户端
        self.dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.dataset_name = args['dataset']
        self.is_iid = args['is_iid']
        self.alpha = args['alpha']
        self.num_client = args['num_client']
        self.num_sample = args['num_sample']
        self.net_name = args['model']
        self.learning_rate = args['learning_rate']
        self.eta = self.learning_rate
        self.client_group = Client_Group(self.dev,
                                         self.num_client,
                                         self.dataset_name,
                                         self.is_iid,
                                         self.alpha,
                                         self.net_name,
                                         self.learning_rate,
                                         )
        self.scales = self.client_group.scales
        self.rate = [item / sum(self.scales) for item in self.scales]
        self.test_dataloader = self.client_group.test_dataloader
        
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
            
      
    def online_estimate_R(self,
                          t,
                          phi,
                          data_list,
                          ):
        
        def func(reward):
            delta_sum = 0
            epsilon_sum = 0
            new_data_list = []
            new_data_sum = 0
            for k in range(self.num_client):
                part_1 = 1 / (2 * self.alpha_list[k])
                part_2 = self.epsilon_list[k] * reward / (self.avg_delta_list[k] * phi)
                part_3 = self.beta_list[k] * (1 - pow(self.theta, self.num_round - t)) / (1 - self.theta)
                increment = part_1 * (part_2 - part_3)
                new_data = self.theta * data_list[k].cpu() + increment
                new_data_list.append(new_data)
                
                new_data_sum += new_data
                delta_sum += new_data_list[k] * (self.avg_delta_list[k] ** 2)
                epsilon_sum += 1 / (self.epsilon_list[k] ** 2)

            self.kappa_1 = 1
            self.kappa_2 = 1
            self.kappa_3 = 1e6
            self.gamma_1 = 1
            self.gamma_2 = 1e-4 # 权衡因子
            # Omega不影响
            item_1 = pow(self.kappa_1, self.num_round - 2 - t)
            item_2 = self.kappa_2 * delta_sum / new_data_sum
            item_3 = self.kappa_3 * epsilon_sum / (new_data_sum ** 2)
            item_4 = reward
            res_1_origin = item_1 * (item_2 + item_3)
            res_1 = self.gamma_1 * item_1 * (item_2 + item_3)
            res_2 = self.gamma_2 * item_4
            res = res_1 + res_2
            print('item2:{},item3:{},res_1_origin:{}'.format(item_2, item_3,res_1_origin))
            print('res_1:{}, res_2:{}'.format(res_1, res_2))
            return res

        pso = PSO(func=func,
                   dim=1,
                   pop=20,
                   max_iter=100,
                   lb=[1e3],
                   ub=[1e6],
                   eps=0.9)
        pso.run()
        reward = pso.gbest_x
        print(len(pso.gbest_y_hist))

        increment_list = []
        new_data_list = []
        for k in range(self.num_client):
            part_1 = 1 / (2 * self.alpha_list[k])
            part_2 = self.epsilon_list[k] * reward / (self.avg_delta_list[k] * phi)
            part_3 = self.beta_list[k] * (1 - pow(self.theta, self.num_round - t)) / (1 - self.theta)
            increment = part_1 * (part_2 - part_3)
            increment_list.append(increment)
            new_data = self.theta * data_list[k].cpu() + increment
            new_data_list.append(new_data)
        return reward, increment_list, new_data_list


    def online_estimate_phi(self, 
                            t,
                            data_list):
        # 初始化phi
        phi = random.random() * 60
        
        ls_phi = []
        ls_reward = []
        ls_increment = []
        # 计算新的phi
        for _ in range(self.fix_max_iter):
            reward, increment_list, new_data_list = self.online_estimate_R(t, phi, data_list)
            new_phi = sum([self.epsilon_list[k] * increment_list[k] / self.avg_delta_list[k] for k in range(self.num_client)])
            ls_phi.append(new_phi)
            ls_reward.append(reward)
            print('reward:{}'.format(reward))
            print('increment:{}'.format(increment_list))
            print('new_phi:{}'.format(new_phi))
        
            if new_phi < 0:
                print('destroy!')
                exit(0)
                
            # 判断收敛
            if abs(new_phi - phi) > self.eps:
                phi = new_phi
                # ls_phi.append(phi)
                # ls_reward.append(reward)
                # ls_increment.append(increment_list[0])
            
            else:
                # print(len(ls_phi))
                # print(ls_phi, '***********')
                # print(ls_reward, '**********')
                # print(ls_increment)
                # plt.plot(ls_phi)
                # plt.plot(ls_reward)
                # plt.plot(ls_increment)
                # plt.savefig('3.png')
                # print('can end!')
                return new_phi, increment_list, new_data_list
        print('-------------------------------')
        print(ls_phi)
        ax1 = plt.subplot(121)
        ax1.plot(ls_phi)
        ax2 = plt.subplot(122)
        ax2.plot(ls_reward)
        plt.savefig('1.png')
        # print(ls_phi, '***********')
        # print(ls_reward, '**********')
        # print(ls_increment)
        # plt.plot(ls_phi)
        # plt.plot(ls_reward)
        # plt.plot(ls_increment)
        # plt.savefig('3.png')
        # print('cannot end!')


    def online_train(self):
        # 初始化数据
        new_data_list = [random.randint(50, 100) for _ in range(self.num_client)]
        
        self.global_parameter = {}
        for key, var in self.net.state_dict().items():
            self.global_parameter[key] = var.clone()
        
        # 训练
        for t in tqdm(range(self.num_round)):
            
            # 使用新收集的数据
            data_list = torch.tensor(new_data_list).to(self.dev)
            data_sum = sum(data_list).to(self.dev)
            
            # 在开始定好R和下一轮的数据，但是不用，因为有一个数据慢慢收集的过程
            phi, new_increment_list, new_data_list = self.online_estimate_phi(t, data_list) # 1, [K]
            print('increment: {}'.format(new_increment_list))
            
            # 开始训练
            next_global_parameter = {}
            local_grad_list = []
            global_grad = 0
            for k in range(self.num_client):
                item = self.client_group.clients[k].local_update(t,
                                                                k,
                                                                self.num_epoch,
                                                                self.batch_size,
                                                                self.global_parameter,
                                                                self.theta,
                                                                data_list[k])
                
                rate = data_list[k] / data_sum
                local_parameter = item[0]
                for item in local_parameter.items():
                    if item[0] not in next_global_parameter.keys():
                        next_global_parameter[item[0]] = rate * item[1]
                    else:
                        next_global_parameter[item[0]] += rate * item[1]
                        
                local_grad = item[1]
                local_grad_list.append(local_grad)
                global_grad += rate * local_grad
                
            # 求global_parameters
            self.global_parameter = next_global_parameter
            
            # 求delta，更新历史平均delta
            for k in range(self.num_client):
                delta = rate * torch.sqrt(torch.sum(torch.square(local_grad_list[k] - global_grad))) # 大错，用delta代替Upsilon
                self.avg_delta_list[k] = (1 - self.p) * self.avg_delta_list[k] + self.p * delta.cpu()
        
server = Server(args)
server.online_train()

# R(T) = 0怎么保证：最后一轮R强制为0
# 初始化你搞的好一点
# 注意不收敛的问题