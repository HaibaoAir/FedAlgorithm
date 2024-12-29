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
from model.cifar import Cifar10_CNN

from sko.PSO import PSO

args = {
    'num_client': 5,
    'num_sample': 5,
    'dataset': 'mnist',
    'is_iid': 0,
    'a': 1.0,
    'model': 'cnn',
    'learning_rate': 0.01,
    'num_round': 20,
    'num_epoch': 1,
    'batch_size': 32,
    'eval_freq': 1,
    'save_path': '../../logs/fedstream/5_client.png',
    'save_path_1': '../../logs/fedstream/5_client_delta.png',
    'save_path_2': '../../logs/fedstream/5_client_data.png',
    'save_path_3': '../../logs/fedstream/5_client_staleness.png',
    'save_path_4': '../../logs/fedstream/5_client_totalsize.png',
    
    'delta': 1,
    'psi': 1,
    'sigma': 1,
    'alpha': 1e-3,
    'beta': 5e-6,
    
    'kappa_1': 1,
    'kappa_2': 1,
    'kappa_3': 1e-2,
    'kappa_4': 1e-2,
    'gamma': 1e-4,
    
    # # 一阶段
    # 'reward_lb': 1,
    # 'reward_ub': 100,
    # 'theta_lb': 0,
    # 'theta_ub': 1,
    # 'pop': 1000, 
    # 'pso_eps': 1e-5,
    # 'pso_max_iter': 500,
    
    # 'fix_eps_1': 1e-2,
    # 'fix_eps_2': 3, # 3
    # 'fix_max_iter': 1000,
    
    # 二阶段
    'reward_lb': 60,
    'reward_ub': 80,
    'theta_lb': 0.1,
    'theta_ub': 0.5,
    'pop': 3000, 
    'pso_eps': 1e-5,
    'pso_max_iter': 500,
    
    'fix_eps_1': 1e-2,
    'fix_eps_2': 5,
    'fix_max_iter': 10000,
    
    'reward': 72,
    'theta': 0.38,
    'rand_lb': 0,
    'rand_ub': 200,
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
        self.save_path = args['save_path']
        self.save_path_1 = args['save_path_1']
        self.save_path_2 = args['save_path_2']
        self.save_path_3 = args['save_path_3']
        self.save_path_4 = args['save_path_4']

        # 初始化data_matrix[K,T]
        # self.data_origin_init = [random.randint(500, 1000) for _ in range(self.num_client)]
        back = [955, 850, 568, 985, 883, 876, 576, 686, 572, 878, 694, 706, 745, 629, 762, 940, 559, 684, 761, 951, 543, 659, 881, 836, 696, 891, 676, 941, 932, 971]
        self.data_origin_init = back[:self.num_client]
        self.data_matrix_init = []
        for k in range(self.num_client):
            data_list = [self.data_origin_init[k]]
            for t in range(1, self.num_round):
                data = random.randint(500, 1000)
                data_list.append(data)
            self.data_matrix_init.append(data_list)
        
        # 初始化phi_list[T]
        self.phi_list_init = []
        for t in range(self.num_round):
            phi = random.random() * 60
            self.phi_list_init.append(phi)
            
        # 初始化第一个客户端的increment_list_random[T]，正常是100，所以在0-200之间波动
        self.increment_list_random = [random.uniform(args['rand_lb'], args['rand_ub']) for _ in range(self.num_round)]
        # back = [6.799899316733904, 40.30057339050413, 41.85100294581272, 76.09632213222253, 125.71510780424123, 53.84139887153938, 76.0021775733326, 108.52100136559162, 155.9337560715368, 136.25913372335364, 143.60220860586222, 109.88048291136543, 162.99749924847512, 156.8106585247205, 39.60702745383995, 140.75469347248824, 89.86066020693198, 66.43020335566217, 188.78970137221742, 167.06313851460135, 32.64559497744435, 121.54468787466448, 190.62672457670234, 123.18911146053316, 57.34961767312166, 112.10578022743817, 77.84928213130794, 90.67444456668605, 105.8185228043597, 123.04188725577502]
        # self.increment_list_random = back[:self.num_round]
        
        # 背景超参数
        self.delta_list = np.array([args['delta']] * self.num_client) # [K]
        self.psi = args['psi']
        self.sigma_list = np.array([args['sigma']] * self.num_client) # [K]
        self.alpha_list = [args['alpha']] * self.num_client # 收集数据的价格
        self.beta_list = [args['beta']] * self.num_client # 训练数据的价格 如果稍大变负数就收敛不了，如果是0就没有不动点的意义
        self.reward = args['reward']
        self.theta = args['theta']
                   
        self.kappa_1 = args['kappa_1']
        self.kappa_2 = args['kappa_2']
        self.kappa_3 = args['kappa_3']
        self.kappa_4 = args['kappa_4']
        self.gamma = args['gamma']
        
        self.reward_lb = args['reward_lb'] # 反正不看过程只看结果，原来1-100得50
        self.reward_ub = args['reward_ub']
        self.theta_lb = args['theta_lb']
        self.theta_ub = args['theta_ub']
        self.pop = args['pop']
        self.pso_eps = args['pso_eps']
        self.pso_max_iter = args['pso_max_iter']
        
        self.fix_eps_1 = args['fix_eps_1']
        self.fix_eps_2 = args['fix_eps_2']
        self.fix_max_iter = args['fix_max_iter']
              
    
    def estimate_direct_D(self, phi_list, reward, theta):
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
                        item2 = reward / (self.delta_list[k] * phi_list[tau])
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
                        stale_matrix[k][t] = stale_matrix[k][t-1] * theta * next_data_matrix[k][t-1] / next_data_matrix[k][t] + 1
                return np.array(increment_matrix), np.array(next_data_matrix), np.array(stale_matrix)
        print('failure1')
        return np.array(next_data_matrix)
    
    
    def estimate_direct_phi(self):
        
        # 初始化phi_list
        phi_list = self.phi_list_init
        
        # 计算新的phi_list[T]
        for idc in range(self.fix_max_iter):
            increment_matrix, data_matrix, _= self.estimate_direct_D(phi_list, self.reward, self.theta)
            # print('******************************************************')
            # print('{}, {}, {}, {}'.format(idc, phi_list, self.reward, self.theta))
            # print('{}'.format(data_matrix))
            
            next_phi_list = np.sum(data_matrix * ((1 / self.delta_list).reshape(self.num_client, 1)), axis=0)
            # 判断收敛
            max_diff = np.max(np.abs(next_phi_list - phi_list))
            
            if max_diff > self.fix_eps_2:
                phi_list = next_phi_list 
            else:
                print('triumph2')
                return next_phi_list
            
        print('direct failure2')
        exit(0)
        return next_phi_list
    
    
    def estimate_zero_D(self, phi_list, reward, theta):
        # 初始化数据矩阵
        data_matrix = self.data_matrix_init
        
        for idc in range(self.fix_max_iter):
            # 计算增量矩阵[K,T]
            increment_matrix = []
            # 第一个客户端是0
            increment_list = [0] * self.num_round
            increment_matrix.append(increment_list)
            # 剩余客户端还是正常
            for k in range(1, self.num_client):
                increment_list = []
                for t in range(0, self.num_round - 1):
                    item = 0
                    for tau in range(t + 1, self.num_round):
                        item1 = pow(theta, tau - t - 1)
                        item2 = reward / (self.delta_list[k] * phi_list[tau])
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
                        stale_matrix[k][t] = stale_matrix[k][t-1] * theta * next_data_matrix[k][t-1] / next_data_matrix[k][t] + 1
                return np.array(increment_matrix), np.array(next_data_matrix), np.array(stale_matrix)
        
        print('failure1')
        return np.array(next_data_matrix)
    
    
    def estimate_zero_phi(self):
        
        # 初始化phi_list
        phi_list = self.phi_list_init
        
        # 计算新的phi_list[T]
        for idc in range(self.fix_max_iter):
            increment_matrix, data_matrix, _= self.estimate_zero_D(phi_list, self.reward, self.theta)
            # print('******************************************************')
            # print('{}, {}, {}, {}'.format(idc, phi_list, self.reward, self.theta))
            # print('{}'.format(data_matrix))
            
            next_phi_list = np.sum(data_matrix * ((1 / self.delta_list).reshape(self.num_client, 1)), axis=0)
            # 判断收敛
            max_diff = np.max(np.abs(next_phi_list - phi_list))
            # print('max_diff_phi:{}'.format(max_diff))
            
            if max_diff > self.fix_eps_2:
                phi_list = next_phi_list 
            else:
                print('triumph2')
                return next_phi_list
            
        print('zero failure2')
        return next_phi_list
    

    def estimate_random_D(self, phi_list, reward, theta):
        # 初始化数据矩阵
        data_matrix = self.data_matrix_init
        
        for idc in range(self.fix_max_iter):
            # 计算增量矩阵[K,T]
            increment_matrix = []
            # 第一个客户端是随机数
            increment_matrix.append(self.increment_list_random)
            # 剩余客户端还是正常
            for k in range(1, self.num_client):
                increment_list = []
                for t in range(0, self.num_round - 1):
                    item = 0
                    for tau in range(t + 1, self.num_round):
                        item1 = pow(theta, tau - t - 1)
                        item2 = reward / (self.delta_list[k] * phi_list[tau])
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
                    if abs(next_data_matrix[k][t] - data_matrix[k][t]) > self.fix_eps_1:
                        flag = 1
                        break
                if flag == 1:
                    break
            if flag == 1:
                data_matrix = next_data_matrix
                # print(np.array(increment_matrix))
            else:
                # print('triumph1, count = {}'.format(idc))
                stale_matrix = [[1] * self.num_round] * self.num_client
                for k in range(self.num_client):
                    for t in range(1, self.num_round):
                        stale_matrix[k][t] = stale_matrix[k][t-1] * theta * next_data_matrix[k][t-1] / next_data_matrix[k][t] + 1
                return np.array(increment_matrix), np.array(next_data_matrix), np.array(stale_matrix)
        
        print('failure1')
        return np.array(next_data_matrix)
    
    
    def estimate_random_phi(self):
        
        # 初始化phi_list
        phi_list = self.phi_list_init
        
        # 计算新的phi_list[T]
        for idc in range(self.fix_max_iter):
            increment_matrix, data_matrix, _= self.estimate_random_D(phi_list, self.reward, self.theta)
            # print('******************************************************')
            # print('{}, {}, {}, {}'.format(idc, phi_list, self.reward, self.theta))
            # print('{}'.format(data_matrix))
            
            next_phi_list = np.sum(data_matrix * ((1 / self.delta_list).reshape(self.num_client, 1)), axis=0)
            # 判断收敛
            max_diff = np.max(np.abs(next_phi_list - phi_list))
            # print('max_diff_phi:{}'.format(max_diff))
            
            if max_diff > self.fix_eps_2:
                phi_list = next_phi_list 
            else:
                print('triumph2')
                return next_phi_list
            
        print('random failure2')
        return next_phi_list
    
    
    def calculate_utility(self, phi_list, increment_matrix, data_matrix):
        
        utility_list = []
        for k in range(self.num_client):
            res = 0
            for t in range(self.num_round):
                item_1 = data_matrix[k][t] * self.reward / (phi_list[t] * self.delta_list[k])
                item_2 = self.alpha_list[k] * (increment_matrix[k][t] ** 2)
                item_3 = self.beta_list[k] * (data_matrix[k][t] ** 2)
                item = item_1 - item_2 - item_3
                res += item
            utility_list.append(res)
        return utility_list


    def online_train(self):
        
        # totalsize图
        width = 0.8
        flag = 0
        
        num_list = []
        increment_list_1 = []
        increment_list_2 = []
        increment_list_3 = []
        enum = [3, 6, 9, 12, 15]
        rand = [120, 70, 60, 55, 48]
        for idx in range(len(enum)):
            args['num_client'] = enum[idx]
            args['rand_lb'] = 0
            args['rand_ub'] = 2 * rand[idx]
            self.__init__(args)
            
            num_list.append(enum[idx])
            
            # case 1
            phi_list = self.estimate_zero_phi()
            increment_matrix, data_matrix, _ = self.estimate_zero_D(phi_list, self.reward, self.theta)
            increment_list_2.append(sum(increment_matrix[0]))
            plt.bar(np.array(num_list)-width, increment_list_2, width=width, hatch='/', color='C1', label=r'$\Delta = 0$')
            for a, b in zip(np.array(num_list)-width, increment_list_2):
                plt.text(a, b, 0, ha='center', va='bottom', fontsize=10)

            # case 2
            phi_list = self.estimate_direct_phi() # [T]
            increment_matrix, data_matrix, staleness_matrix = self.estimate_direct_D(phi_list, self.reward, self.theta)
            increment_list_1.append(sum(increment_matrix[0]))
            plt.bar(np.array(num_list), increment_list_1, width=width, hatch='-', color='C0', label=r'$\Delta = optimal$')
            
            # case 3
            phi_list = self.estimate_random_phi()
            increment_matrix, data_matrix, _ = self.estimate_random_D(phi_list, self.reward, self.theta)
            increment_list_3.append(sum(increment_matrix[0]))
            plt.bar(np.array(num_list)+width, increment_list_3, width=width, hatch='\\', color='C2', label=r'$\Delta = random$')
            
            plt.xticks(num_list)
            plt.ylabel(r'Total Increment $\sum \Delta(t)$')
            plt.xlabel(r'Number of Clients $N$')
            plt.legend(frameon=False)
            plt.savefig(self.save_path_4, dpi=200)
            plt.close()
            
    def online_train_1(self):
        
        # 绘制utility图
        width = 0.8
        flag = 0
        
        num_list = []
        utility_list_1 = []
        utility_list_2 = []
        utility_list_3 = []
        enum = [3, 6, 9, 12, 15]
        rand = [120, 70, 60, 55, 48]
        for idx in range(len(enum)):
            args['num_client'] = enum[idx]
            args['rand_lb'] = 0
            args['rand_ub'] = 2 * rand[idx]
            self.__init__(args)
            
            num_list.append(enum[idx])

            # case 1
            phi_list = self.estimate_zero_phi()
            increment_matrix, data_matrix, _ = self.estimate_zero_D(phi_list, self.reward, self.theta)
            utility_list_2.append(max(self.calculate_utility(phi_list, increment_matrix, data_matrix)[0], 0))
            print('data_matrix[0]:{}'.format(data_matrix[0]))
            plt.bar(np.array(num_list) - width, utility_list_2, width, color='C1', hatch='/', label=r'$\Delta = 0$')

            # case 2
            phi_list = self.estimate_direct_phi() # [T]
            increment_matrix, data_matrix, staleness_matrix = self.estimate_direct_D(phi_list, self.reward, self.theta)
            utility_list_1.append(max(self.calculate_utility(phi_list, increment_matrix, data_matrix)[0], 0))
            plt.bar(np.array(num_list), utility_list_1, width, color='C0', hatch='-', label=r'$\Delta = optimal$')
        
            # case 3
            phi_list = self.estimate_random_phi()
            increment_matrix, data_matrix, _ = self.estimate_random_D(phi_list, self.reward, self.theta)
            utility_list_3.append(max(self.calculate_utility(phi_list, increment_matrix, data_matrix)[0], 0))
            plt.bar(np.array(num_list) + width, utility_list_3, width, color='C2', hatch='\\', label=r'$\Delta = random$')
            
            plt.xticks(num_list)
            plt.xlabel(r'Number of Clients $N$')
            plt.ylabel(r'Utility of Client $U_k$')
            if flag == 0:
                flag = 1
                plt.legend()
            plt.tight_layout()
            plt.savefig(self.save_path, dpi=200)
     
    
    def online_train_2(self):
        
        # 揭示原理图
        num_list = []
        increment_list = []
        datasize_list = []
        staleness_list = []
        enum = [3, 6, 9, 12, 15]
        rand = [120, 70, 60, 55, 48]
        for num in enum:
            args['num_client'] = num
            self.__init__(args)
            
            num_list.append(num)
            phi_list = self.estimate_direct_phi() # [T]
            increment_matrix, data_matrix, staleness_matrix = self.estimate_direct_D(phi_list, self.reward, self.theta)
            increment_list.append(np.mean(increment_matrix, axis=0))
            datasize_list.append(np.mean(data_matrix, axis=0))
            staleness_list.append(np.mean(staleness_matrix, axis=0))
            
            for n in range(len(num_list)):
                plt.plot(increment_list[n], label='{} client'.format(num_list[n]))
                plt.ylabel(r'Average Increment')
                plt.xlabel(r'Round $T$')
                print('num {} increment:{}'.format(num_list[n], increment_list[n][10]))
            plt.legend(frameon=False)
            plt.savefig(self.save_path_1, dpi=200)
            plt.close()
            
            for n in range(len(num_list)):
                plt.plot(datasize_list[n], label='{} client'.format(num_list[n]))
                plt.ylabel(r'Average Increment')
                plt.xlabel(r'Round $T$')
            plt.legend(frameon=False)
            plt.savefig(self.save_path_2, dpi=200)
            plt.close()
            
            for n in range(len(num_list)):
                plt.plot(staleness_list[n], label='{} client'.format(num_list[n]))
                plt.ylabel(r'Average Increment')
                plt.xlabel(r'Round $T$')
            plt.legend(frameon=False)
            plt.savefig(self.save_path_3, dpi=200)
            plt.close()
            
                        
        
server = Server(args)
server.online_train_1()

# R(T) = 0怎么保证：最后一轮R强制为0
# 初始化你搞的好一点
# 注意不收敛的问题