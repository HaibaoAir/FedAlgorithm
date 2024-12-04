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
    'num_client': 3,
    'num_sample': 3,
    'dataset': 'mnist',
    'is_iid': 0,
    'alpha': 1.0,
    'model': 'cnn',
    'learning_rate': 0.01,
    'num_round': 5,
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
        self.net.to(self.dev)   
             
        # 训练超参数
        self.num_round = args['num_round']
        self.num_epoch = args['num_epoch']
        self.batch_size = args['batch_size']
        self.eval_freq = args['eval_freq']

        # 初始化data_matrix[K,T]
        self.data_origin_init = [random.randint(50, 100) for _ in range(self.num_client)]
        self.data_matrix_init = []
        for k in range(self.num_client):
            data_list = [self.data_origin_init[k]]
            for t in range(1, self.num_round):
                data = random.randint(50, 100)
                data_list.append(data)
            self.data_matrix_init.append(data_list)
        
        # 初始化phi_list[T]
        self.phi_list_init = []
        for t in range(self.num_round):
            phi = random.random() * 60
            self.phi_list_init.append(phi)
        
        # 背景超参数
        self.epsilon_list = np.array([1] * self.num_client) #[K] 0-1之间
        self.delta_list = np.array([1] * self.num_client) # [K]
        self.alpha_list = [1e-3] * self.num_client # 收集数据的价格
        self.beta_list = [1e-7] * self.num_client # 训练数据的价格 如果稍大变负数就收敛不了，如果是0就没有不动点的意义
        self.theta = 0.5 # 数据留存率
        self.fix_eps = 1
        self.fix_max_iter = 10000
        
        self.kappa_1 = 1
        self.kappa_2 = 1
        self.kappa_3 = 1e6 # 重点1
        self.gamma = 0.5
        
        self.lb = 1
        self.ub = 100 # 10e6 # 重点2
        self.pop = 1000 # 探索空间划定得越大，需要的粒子就多点，就能找到精确最优解！
        self.pso_eps = 1e-3
        self.pso_max_iter = 500
              
    
    def estimate_D(self, phi_list, reward, ind=0):
        # 初始化数据矩阵
        data_matrix = self.data_matrix_init
        
        for idc in range(self.fix_max_iter):
            # 计算增量矩阵[K,T]
            increment_matrix = []
            for k in range(self.num_client):
                increment_list = []
                for t in range(0, self.num_round - 1):
                    item = 0
                    for tau in range(t + 1, self.num_round):
                        item1 = pow(self.theta, tau - t - 1)
                        item2 = self.epsilon_list[k] * reward[0] / (self.delta_list[k] * phi_list[tau])
                        item3 = 2 * self.beta_list[k] * data_matrix[k][tau]
                        item += item1 * (item2 - item3)
                    increment = 1 / (2 * self.alpha_list[k]) * item
                    # if increment <= 0:
                    #     print('dual')
                    increment = max(0, increment) # 好哇好
                    increment_list.append(increment)
                increment_list.append(0)
                increment_matrix.append(increment_list)
                
            if ind == 1:
                print(np.array(increment_matrix))
            
            # 新的数据矩阵[K, T]
            next_data_matrix = []
            for k in range(self.num_client):
                next_data_list = [np.array(self.data_origin_init[k])]
                for t in range(1, self.num_round):
                    next_data = self.theta * next_data_list[t - 1] + increment_matrix[k][t - 1]
                    next_data_list.append(next_data)
                next_data_matrix.append(next_data_list)
            # 判断收敛
            flag = 0
            for k in range(self.num_client):
                for t in range(1, self.num_round):
                    if abs(next_data_matrix[k][t] - data_matrix[k][t]) > self.fix_eps:
                        flag = 1
                        break
                if flag == 1:
                    break
            if flag == 1:
                data_matrix = next_data_matrix
            else:
                # print('triumph1, count = {}'.format(idc))
                return np.array(next_data_matrix)
        print('failure1')
        return np.array(next_data_matrix)
    
    def estimate_R(self, phi_list):
        
        def func(reward):
            # 计算单列和
            data_matrix = self.estimate_D(phi_list, reward) 
            data_list = [] # [T]
            for t in range(self.num_round):
                data = 0
                for k in range(self.num_client):
                    data += data_matrix[k][t]
                data_list.append(data)
            
            res = 0
            for t in range(self.num_round):
                delta_sum = 0
                epsilon_sum = 0
                for k in range(self.num_client):
                    delta_sum += data_matrix[k][t] * (self.delta_list[k] ** 2)
                    epsilon_sum += 1 / (self.epsilon_list[k] ** 2)
                    
                # Omega不影响
                item_1 = pow(self.kappa_1, self.num_round - 1 - t) * self.kappa_2 * delta_sum / data_list[t]
                item_2 = pow(self.kappa_1, self.num_round - 1 - t) * self.kappa_3 * epsilon_sum / (data_list[t] ** 2)
                item = (1 - self.gamma) * (item_1 + item_2) + self.gamma * reward
                res += item
            return res
        
        pso =  PSO(func=func,
                   dim=1,
                   pop=self.pop,
                   max_iter=self.pso_max_iter,
                   lb=[self.lb],
                   ub=[self.ub],
                   eps=self.pso_eps)
        pso.run()
        return pso.gbest_x, pso.gbest_y_hist
    
    
    def estimate_phi(self):
        
        def func1(phi_list, reward):
            # 计算单列和
            data_matrix = self.estimate_D(phi_list, reward) 
            data_list = [] # [T]
            for t in range(self.num_round):
                data = 0
                for k in range(self.num_client):
                    data += data_matrix[k][t]
                data_list.append(data)
            
            res = 0
            res_2 = 0
            for t in range(self.num_round):
                delta_sum = 0
                epsilon_sum = 0
                for k in range(self.num_client):
                    delta_sum += data_matrix[k][t] * (self.delta_list[k] ** 2)
                    epsilon_sum += 1 / (self.epsilon_list[k] ** 2)
                    
                # Omega不影响
                item_1 = pow(self.kappa_1, self.num_round - 1 - t) * self.kappa_2 * delta_sum / data_list[t]
                item_2 = pow(self.kappa_1, self.num_round - 1 - t) * self.kappa_3 * epsilon_sum / (data_list[t] ** 2)
                item = (1 - self.gamma) * (item_1 + item_2) + self.gamma * reward
                res_2 += item_2
                res += item
            return data_matrix, res_2, res
            
        def func2(phi_list, reward, theta):
            # 初始化数据矩阵
            data_matrix = self.data_matrix_init
            
            for idc in range(self.fix_max_iter):
                # 计算增量矩阵[K,T]
                increment_matrix = []
                for k in range(self.num_client):
                    increment_list = []
                    for t in range(0, self.num_round - 1):
                        item = 0
                        for tau in range(t + 1, self.num_round):
                            item1 = pow(theta, tau - t - 1)
                            item2 = self.epsilon_list[k] * reward[0] / (self.delta_list[k] * phi_list[tau])
                            item3 = 2 * self.beta_list[k] * data_matrix[k][tau]
                            item += item1 * (item2 - item3)
                        increment = 1 / (2 * self.alpha_list[k]) * item
                        # if increment <= 0:
                        #     print('dual')
                        increment = max(0, increment) # 好哇好
                        increment_list.append(increment)
                    increment_list.append(0)
                    increment_matrix.append(increment_list)
                
                # 新的数据矩阵[K, T]
                next_data_matrix = []
                for k in range(self.num_client):
                    next_data_list = [np.array(self.data_origin_init[k])]
                    for t in range(1, self.num_round):
                        next_data = theta * next_data_list[t - 1] + increment_matrix[k][t - 1]
                        next_data_list.append(next_data)
                    next_data_matrix.append(next_data_list)
                # 判断收敛
                flag = 0
                for k in range(self.num_client):
                    for t in range(1, self.num_round):
                        if abs(next_data_matrix[k][t] - data_matrix[k][t]) > self.fix_eps:
                            flag = 1
                            break
                    if flag == 1:
                        break
                if flag == 1:
                    data_matrix = next_data_matrix
                else:
                    # 观察一下
                    obs_matrix = []
                    for k in range(self.num_client):
                        obs_list = [1]
                        for t in range(1, self.num_round):
                            obs = obs_list[t-1] * (1 - increment_matrix[k][t-1] / next_data_matrix[k][t]) + 1
                            obs_list.append(obs)
                        obs_matrix.append(obs_list)
                    
                    return np.array(next_data_matrix), np.array(obs_matrix)
            print('failure1')
            return np.array(next_data_matrix)
        
        # 初始化phi_list
        phi_list = self.phi_list_init
        
        # 计算新的phi_list[T]
        phi_hist = []
        for idc in range(self.fix_max_iter):
            phi_hist.append(phi_list[1])
            reward, res = self.estimate_R(phi_list)
            data_matrix = self.estimate_D(phi_list, reward, 1)
            print('******************************************************')
            print('{}, {}, {}, {}'.format(idc, phi_list, reward, res))
            print('{}'.format(data_matrix))
            # print('--------------------------------------')
            # print('{}'.format(func(phi_list, reward)[0]))
            # # print('--------------------------------------')
            # # print('{}'.format(func(phi_list, reward)[1]))
            # 先epsilon/delta，再施加到data_matrix
            # print(data_matrix.shape)
            next_phi_list = np.sum(data_matrix * ((self.epsilon_list / self.delta_list).reshape(self.num_client, 1)), axis=0)
            # 判断收敛
            if np.max(np.abs(next_phi_list - phi_list)) > self.fix_eps:
                phi_list = next_phi_list
                x1 = np.arange(self.lb, self.ub, 1)
                y1 = [np.sum(self.estimate_D(phi_list, np.array([i]))) for i in x1]
                y2 = []
                y3 = []
                for i in x1:
                    _, a, b = func1(phi_list, np.array([i]))
                    y2.append(a)
                    y3.append(b)
                plt.subplot(2,3,1)
                plt.plot(x1, y1)
                plt.subplot(2,3,2)
                plt.plot(x1, y2)
                plt.subplot(2,3,3)
                plt.plot(x1, y3)
                plt.savefig('../../logs/fedavg/2.png')  
                
            else:
                x2 = np.arange(0, 1, 0.1)
                y4 = []
                for i in x2:
                    _, c = func2(phi_list, reward, i)
                    plt.subplot(2,3,5)
                    plt.plot(c[1])
                    plt.savefig('../../logs/fedavg/2.png')    
                    
                plt.subplot(2,3,6)
                plt.plot(phi_hist)
                plt.savefig('../../logs/fedavg/2.png')
                print('triumph2')
                return next_phi_list
            
        print('failure2')
        return next_phi_list


    def online_train(self):
        
        # 正式训练前定好一切
        phi_list = self.estimate_phi() # [T]
        reward, res = self.estimate_R(phi_list) # [K, T]
        data_matrix = self.estimate_D(phi_list, reward)
        # numpy 切片格式 [:, :]
        data_sum_list = [sum(data_matrix[:, t]) for t in range(self.num_round)]
        
        # 初始化数据
        # 临时记录，求平均后是self.delta_list
        delta_matrix = [[] for k in range(self.num_client)]
        
        self.global_parameter = {}
        for key, var in self.net.state_dict().items():
            self.global_parameter[key] = var.clone()
            
        accuracy_list = []
        
        # 训练
        for t in tqdm(range(self.num_round)):
            
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
                                                                data_matrix[k][t])
                
                rate = data_matrix[k][t] / data_sum_list[t]
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
            
            # 求delta
            for k in range(self.num_client):
                delta = rate * torch.sqrt(torch.sum(torch.square(local_grad_list[k] - global_grad))) # 大错，用delta代替Upsilon
                delta_matrix[k].append(delta)
            
            # 验证
            if t % self.eval_freq == 0:    
                correct = 0
                total = 0
                self.net.load_state_dict(self.global_parameter)
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
                
            plt.subplot(2,3,4)
            plt.plot(accuracy_list)
            plt.savefig('../../logs/fedavg/2.png')
        
        with open('../../logs/fedavg/accuracy.txt', 'a') as file:
            file.write('{}\n'.format(time.asctime()))
            for accuracy in accuracy_list:
                file.write('{:^7.5f} '.format(accuracy))
            file.write('\n')
                
        
server = Server(args)
# server.estimate_D([1,1,1,1,1], [100])
server.online_train()

# R(T) = 0怎么保证：最后一轮R强制为0
# 初始化你搞的好一点
# 注意不收敛的问题