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

from alg.fedstream.clients_old import Client_Group
from model.mnist import MNIST_Linear, MNIST_CNN
from model.cifar10 import Cifar10_CNN

from sko.PSO import PSO

args = {
    'num_client': 3,
    'num_sample': 3,
    'dataset': 'mnist',
    'is_iid': 0,
    'a': 1.0,
    'model': 'cnn',
    'learning_rate': 0.01,
    'num_round': 5,
    'num_epoch': 1,
    'batch_size': 32,
    'eval_freq': 1,
    'save_path': '../../logs/fedavg/app_2.png',
    
    'delta': 1,
    'psi': 1,
    'sigma': 1,
    'alpha': 1e-2,
    'beta': 5e-6,
    
    'kappa_1': 1,
    'kappa_2': 1,
    'kappa_3': 1e-2,
    'kappa_4': 1e-2,
    'gamma': 1e-3,
    
    'lb': 1, # 1
    'ub': 100,
    'pop': 1000, # 2
    'pso_eps': 1e-2,
    'pso_max_iter': 500,
    
    'fix_eps_1': 1e-2,
    'fix_eps_2': 3, # 3
    'fix_max_iter': 1000,
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
        self.a = args['a']
        self.num_client = args['num_client']
        self.num_sample = args['num_sample']
        self.net_name = args['model']
        self.learning_rate = args['learning_rate']
        self.eta = self.learning_rate
        self.client_group = Client_Group(self.dev,
                                         self.num_client,
                                         self.dataset_name,
                                         self.is_iid,
                                         self.a,
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
        self.delta_list = np.array([args['delta']] * self.num_client) # [K]
        self.psi = args['psi']
        self.sigma_list = np.array([args['sigma']] * self.num_client) # [K]
        self.alpha_list = [args['alpha']] * self.num_client # 收集数据的价格
        self.beta_list = [args['beta']] * self.num_client # 训练数据的价格 如果稍大变负数就收敛不了，如果是0就没有不动点的意义
        self.theta = None
        
        self.fix_eps_1 = args['fix_eps_1']
        self.fix_eps_2 = args['fix_eps_2']
        self.fix_max_iter = args['fix_max_iter']
        
        self.kappa_1 = args['kappa_1']
        self.kappa_2 = args['kappa_2']
        self.kappa_3 = args['kappa_3']
        self.kappa_4 = args['kappa_4']
        self.gamma = args['gamma']
        
        self.lb = args['lb'] # 反正不看过程只看结果，原来1-100得50
        self.ub = args['ub']
        self.pop = args['pop']
        self.pso_eps = args['pso_eps']
        self.pso_max_iter = args['pso_max_iter']
        
        self.save_path = args['save_path']
    
    # ---------------------------------------------------#    
    # 补充实验，确定theta，phi-reward-Delta一起不动点 vs 确定theta, reward最优，phi-Delta不动点#
    def estimate_direct_D(self, phi_list, reward):
        # 初始化数据矩阵
        data_matrix = self.data_matrix_init
        reward = reward[0]
        
        for idc in range(self.fix_max_iter):
            # 计算增量矩阵[K,T]
            increment_matrix = []
            for k in range(self.num_client):
                increment_list = []
                for t in range(0, self.num_round - 1):
                    item = 0
                    for tau in range(t + 1, self.num_round):
                        item1 = pow(self.theta, tau - t - 1)
                        item2 = reward / (self.delta_list[k] * phi_list[tau])
                        item3 = 2 * self.beta_list[k] * data_matrix[k][tau]
                        item += item1 * (item2 - item3)
                    increment = 1 / (2 * self.alpha_list[k]) * item
                    if increment <= 0:
                        print('dual')
                    increment = max(0, increment) # 好哇好
                    increment_list.append(increment)
                increment_list.append(0)
                increment_matrix.append(increment_list)
            
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
                    if abs(next_data_matrix[k][t] - data_matrix[k][t]) > self.fix_eps_1:
                        flag = 1
                        break
                if flag == 1:
                    break
            if flag == 1:
                data_matrix = next_data_matrix
            else:
                # print('triumph1, count = {}'.format(idc))
                stale_matrix = [[1] * self.num_round] * self.num_client
                for k in range(self.num_client):
                    for t in range(1, self.num_round):
                        stale_matrix[k][t] = stale_matrix[k][t-1] * self.theta * next_data_matrix[k][t-1] / next_data_matrix[k][t] + 1
                return np.array(increment_matrix), np.array(next_data_matrix), np.array(stale_matrix)
        print('failure1')
        return np.array(next_data_matrix)


    def estimate_direct_phi(self, reward):
                
        # 初始化phi_list
        phi_list = self.phi_list_init
        
        # 计算新的phi_list[T]
        for idc in range(self.fix_max_iter):
            increment_matrix, data_matrix, stale_matrix= self.estimate_direct_D(phi_list, reward)
            # print('******************************************************')
            # print('{}, {}, {}, {}'.format(idc, phi_list, self.reward, self.theta))
            # print('{}'.format(data_matrix))
            
            next_phi_list = np.sum(data_matrix * ((1 / self.delta_list).reshape(self.num_client, 1)), axis=0)
            # 判断收敛
            max_diff = np.max(np.abs(next_phi_list - phi_list))
            print('max_diff_phi:{}'.format(max_diff))
            
            if max_diff > self.fix_eps_2:
                phi_list = next_phi_list 
            else:
                print('triumph2')
                return next_phi_list, increment_matrix, data_matrix, stale_matrix
            
        print('failure2')
        return next_phi_list
    
    
    def estimate_direct_reward(self):
        
        def func(reward):
            # 计算单列和
            var = self.estimate_direct_phi(reward) 
            phi_list = var[0]
            increment_matrix = var[1]
            data_matrix = var[2]
            stale_matrix = var[3]
            
            data_list = [] # [T]
            for t in range(self.num_round):
                data = 0
                for k in range(self.num_client):
                    data += data_matrix[k][t]
                data_list.append(data)
            
            res = 0
            for t in range(self.num_round):
                delta_sum = 0
                stale_sum = 0
                for k in range(self.num_client):
                    delta_sum += data_matrix[k][t] * (self.delta_list[k] ** 2)
                    stale_sum += data_matrix[k][t] * stale_matrix[k][t] * (self.sigma_list[k] ** 2)
                    
                # Omega不影响
                item_1 = pow(self.kappa_1, self.num_round - 1 - t) * self.kappa_2 * self.num_client * (self.psi ** 2) / data_list[t]
                item_2 = pow(self.kappa_1, self.num_round - 1 - t) * self.kappa_3 * stale_sum / data_list[t]
                item_3 = pow(self.kappa_1, self.num_round - 1 - t) * self.kappa_4 * delta_sum / data_list[t]
                item = (1 - self.gamma) * (item_1 + item_2 + item_3) + self.gamma * reward
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
    # ---------------------------------------------------#


    def online_train(self):    
        styles = {
            0.2: '^-',
            0.4: 'D-',
            0.6: 'o-',
            0.8: 'x-',
        }
        theta_list = []
        reward_list = []
        res_list = []
        for item in range(2, 10, 2):
            self.theta = item / 10
            reward, res = self.estimate_direct_reward()
            theta_list.append(self.theta)
            reward_list.append(reward)
            res_list.append(res[0])
            print(theta_list)
            print(reward_list)
            print(res_list)
            plt.subplot(2,2,1)
            plt.xlabel('theta', fontdict={'family':'Times New Roman', 'size':16, 'weight':'bold'})
            plt.ylabel('reward', fontdict={'family':'Times New Roman', 'size':16, 'weight':'bold'})
            # plt.bar(theta_list, reward_list, width=0.1)
            plt.plot(theta_list, reward_list, 'o--')
            plt.subplot(2,2,2)
            plt.xlabel('theta', fontdict={'family':'Times New Roman', 'size':16, 'weight':'bold'})
            plt.ylabel('res', fontdict={'family':'Times New Roman', 'size':16, 'weight':'bold'})
            # plt.bar(theta_list, res_list, width=0.1)
            plt.plot(theta_list, res_list, 'o--')
            plt.legend(prop={'family':'Times New Roman', 'weight':'bold'})
            plt.tight_layout()
            plt.savefig(self.save_path, dpi=200)
        plt.close()
        exit(0)
        
        # 正式训练前定好一切
        phi_list = self.estimate_phi() # [T]
        theta, res = self.estimate_theta(phi_list) # [K, T]
        _, data_matrix, _ = self.estimate_D(phi_list, theta)
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
                                                                theta,
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
            plt.savefig(self.save_path)
        
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