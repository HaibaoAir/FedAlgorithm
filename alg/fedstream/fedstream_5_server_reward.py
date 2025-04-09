import json
import os
import sys
import time
sys.path.append('../..')
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader # 三件套
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy

from alg.fedstream.clients import Client_Group
from model.mnist import MNIST_Linear, MNIST_CNN
from model.cifar10 import Cifar10_CNN

from sko.PSO import PSO

args = {
    'num_client': 5,
    'num_sample': 5,
    'dataset': 'mnist',
    'is_iid': 2,
    'a': 1.0,
    'model': 'cnn',
    'learning_rate': 0.01,
    'num_round': 20,
    'num_epoch': 1,
    'batch_size': 32,
    'eval_freq': 1,
    'save_path': '../../logs/fedstream/5_server.png',
    'save_path_1': '../../logs/fedstream/5_server_1.png',
    'save_path_1_delta': '../../logs/fedstream/5_server_1_delta.png',
    'save_path_1_data': '../../logs/fedstream/5_server_1_data.png',
    'save_path_1_stale': '../../logs/fedstream/5_server_1_stale.png',
    'save_path_2': '../../logs/fedstream/5_server_2.png',
    'save_path_2_9': '../../logs/fedstream/5_server_2_9.png',
    'save_path_2_19': '../../logs/fedstream/5_server_2_19.png',
    'save_path_3': '../../logs/fedstream/5_server_3.png',
    'save_path_3_loss': '../../logs/fedstream/5_server_3_loss.png',
    'save_path_3_acc': '../../logs/fedstream/5_server_3_acc.png',
    'save_path_4': '../../logs/fedstream/5_server_4.png',
    'pre_estimate_path_1': '../../logs/fedstream/pre_estimate_5_1.npy',
    'pre_estimate_path_2': '../../logs/fedstream/pre_estimate_5_2.npy',
    
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
    # self.reward_lb = 1
    # self.reward_ub = 100
    # self.theta_lb = 0
    # self.theta_ub = 1
    # self.pop = 1000
    # self.pso_eps = 1e-5
    # self.pso_max_iter = 500
    
    # 二阶段
    'reward_lb': 1,
    'reward_ub': 100,
    'theta_lb': 0.1,
    'theta_ub': 1,
    'pop': 3000,
    'pso_eps': 1e-5,
    'pso_max_iter': 500,
    
    'fix_eps_1': 1e-2,
    'fix_eps_2': 5,
    'fix_max_iter': 100000,
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
        self.init_data_net()
        
        # 训练超参数
        self.num_round = args['num_round']
        self.num_epoch = args['num_epoch']
        self.batch_size = args['batch_size']
        self.eval_freq = args['eval_freq']
        self.save_path = args['save_path']
        self.save_path_1 = args['save_path_1']
        self.save_path_1_delta = args['save_path_1_delta']
        self.save_path_1_data = args['save_path_1_data']
        self.save_path_1_stale = args['save_path_1_stale']
        self.save_path_2 = args['save_path_2']
        self.save_path_2_9 = args['save_path_2_9']
        self.save_path_2_19 = args['save_path_2_19']
        self.save_path_3 = args['save_path_3']
        self.save_path_3_loss = args['save_path_3_loss']
        self.save_path_3_acc = args['save_path_3_acc']
        self.save_path_4 = args['save_path_4']
        self.pre_estimate_path_1 = args['pre_estimate_path_1']
        self.pre_estimate_path_2 = args['pre_estimate_path_2']

        # 初始化data_matrix[K,T]
        self.data_origin_init = [random.randint(500, 1000) for _ in range(self.num_client)]
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
            
        # 背景超参数
        self.delta_list = np.array([args['delta']] * self.num_client) # [K]
        self.psi = args['psi']
        self.sigma_list = np.array([args['sigma']] * self.num_client) # [K]
        self.alpha_list = [args['alpha']] * self.num_client # 收集数据的价格
        self.beta_list = [args['beta']] * self.num_client # 训练数据的价格 如果稍大变负数就收敛不了，如果是0就没有不动点的意义
                  
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
           
        
    def init_data_net(self):
        self.client_group = Client_Group(self.dev,
                                         self.num_client,
                                         self.dataset_name,
                                         self.is_iid,
                                         self.a,
                                         self.net_name,
                                         self.learning_rate,
                                         )
        self.test_data_list = self.client_group.test_data_list
        # for item in self.test_data_list[2]:
        #     print(item[-1])
        # exit(0)
            
 
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
              
    
    def estimate_D(self, phi_list, reward, theta):
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
    
    
    def estimate_reward_theta(self, phi_list):
        
        def func(var):
            reward, theta = var
            
            # 计算单列和
            _, data_matrix, stale_matrix = self.estimate_D(phi_list, reward, theta) 
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
                   dim=2,
                   pop=self.pop,
                   max_iter=self.pso_max_iter,
                   lb=[self.reward_lb, self.theta_lb],
                   ub=[self.reward_ub, self.theta_ub],
                   eps=self.pso_eps)
        pso.run()
        return pso.gbest_x, pso.gbest_y_hist
    
    
    def estimate_phi(self):
        
        # 初始化phi_list
        phi_list = self.phi_list_init
        phi_hist = []
        reward_hist = []
        theta_hist = []
        max_diff_hist = []
        
        # 计算新的phi_list[T]
        for idc in range(self.fix_max_iter):
            var, res = self.estimate_reward_theta(phi_list)
            reward, theta = var
            increment_matrix, data_matrix, _= self.estimate_D(phi_list, reward, theta)
            print('******************************************************')
            print('{}, {}, {}, {}, {}'.format(idc, phi_list, reward, theta, res))
            print('{}'.format(data_matrix))
            
            next_phi_list = np.sum(data_matrix * ((1 / self.delta_list).reshape(self.num_client, 1)), axis=0)
            # 判断收敛
            max_diff = np.max(np.abs(next_phi_list - phi_list))
            print('max_diff_phi:{}'.format(max_diff))
            
            if max_diff > self.fix_eps_2:
                phi_list = next_phi_list 
                
                # if idc == 0 or idc == 1:
                #     continue
                
                phi_hist.append(phi_list[-1])
                reward_hist.append(reward)
                theta_hist.append(theta)
                max_diff_hist.append(max_diff)
                
                fig = plt.figure()
                ax1 = fig.add_subplot(2,1,1)
                ax1.set_xlabel('iterations')
                ax1.set_ylabel('phi')
                ax1.plot(phi_hist, 'k-')
                
                ax2 = fig.add_subplot(2,1,2)
                ax2.set_xlabel('iterations')
                ax2.set_ylabel('reward')
                ax2.spines['left'].set_edgecolor('C0')
                ax2.yaxis.label.set_color('C0')
                ax2.tick_params(axis='y', colors='C0')
                line_2 = ax2.plot(reward_hist, color='C0', linestyle='-', label='reward')
                
                ax3 = ax2.twinx()
                ax3.set_ylabel('theta')
                ax3.spines['right'].set_edgecolor('red')
                ax3.yaxis.label.set_color('red')
                ax3.tick_params(axis='y', colors='red')
                line_3 = ax3.plot(theta_hist, 'r--', label='theta')
                
                lines = line_2 + line_3
                labs = [label.get_label() for label in lines]
                ax3.legend(lines,labs, frameon=False, loc=4)
                
                fig.tight_layout()
                fig.savefig(self.save_path, dpi=200)
                plt.close()
                
            else:           
                max_diff_hist.append(max_diff)    
                print('triumph2')
                return next_phi_list
            
        print('failure2')
        return next_phi_list
    
    
    # 给定R和theta，估计phi和delta ---------------------------------------------------
    def estimate_direct_phi(self, reward, theta):
        
        # 初始化phi_list
        phi_list = self.phi_list_init
        
        # 计算新的phi_list[T]
        for idc in range(self.fix_max_iter):
            increment_matrix, data_matrix, stale_matrix = self.estimate_D(phi_list, reward, theta)
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
                return next_phi_list, increment_matrix, data_matrix, stale_matrix
            
        print('failure2')
        return next_phi_list


    def estimate_direct_func(self, reward, data_matrix, stale_matrix):        
        # 计算单列和
        data_list = [] # [T]
        for t in range(self.num_round):
            data = 0
            for k in range(self.num_client):
                data += data_matrix[k][t]
            data_list.append(data)
        
        res1 = 0
        res2 = 0
        res3 = 0
        res4 = 0
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
            res1 += item_1
            res2 += item_2
            res3 += item_3
            res4 += self.gamma * reward
        return res, res1, res2, res3, res4


    def online_train(self):
        # 正式训练前定好一切
        if os.path.exists(self.pre_estimate_path_1) == False:
            phi_list = self.estimate_phi() # [T]
            result = self.estimate_reward_theta(phi_list) # [K, T]
            np.save(self.pre_estimate_path_1, phi_list)
            np.save(self.pre_estimate_path_2, result)
        
        phi_list = np.load(self.pre_estimate_path_1)
        result = np.load(self.pre_estimate_path_2)
        reward = result[0][0]
        theta = result[0][1]
        res = result[1][1]
        
        delta_list = []
        res_list = []
        los_list_9 = []
        acc_list_9 = []
        los_list_19 = []
        acc_list_19 = []
        res1_list = []
        res2_list = []
        res3_list = []
        res4_list = []
        delta_reward_list = []
        delta_theta_list = []
        hist_increment_matrix = []
        hist_data_matrix = []
        hist_stale_matrix = []
        hist_loss_matrix = []
        hist_acc_matrix = []
        flag = 0
        styles = ['^-', 'D-', 'o-', 'x-', 'v-']
        enum = [[-70, 0], [-35, 0], [0, 0], [35, 0], [70, 0]]
        for idx in range(len(enum)):
            delta_com = enum[idx]
            new_reward = reward + delta_com[0]
            new_theta = theta + delta_com[1]
            print(new_reward)
            print(new_theta)
            delta_list.append('{}'.format(delta_com))
            delta_reward_list.append(delta_com[0])
            delta_theta_list.append(delta_com[1])

            var = self.estimate_direct_phi(new_reward, new_theta)
            phi_list = var[0]
            increment_matrix = var[1]
            data_matrix = var[2]
            stale_matrix = var[3]
            hist_increment_matrix.append(np.mean(increment_matrix, axis=0))
            hist_data_matrix.append(np.mean(data_matrix, axis=0))
            hist_stale_matrix.append(np.mean(stale_matrix, axis=0))
            
            res, res1, res2, res3, res4 = self.estimate_direct_func(new_reward, data_matrix, stale_matrix)
            res_list.append(res)
            res1_list.append(res1)
            res2_list.append(res2)
            res3_list.append(res3)
            res4_list.append(res4)
                                
            # # 画图1
            # for n in range(idx+1):
            #     if delta_reward_list[n] < 0:
            #         label = r'$(R^*{}, \theta^*$)'.format(delta_reward_list[n])
            #     elif delta_reward_list[n] == 0:
            #         label = r'$(R^*, \theta^*$)'
            #     else:
            #         label = r'$(R^*+{}, \theta^*$)'.format(delta_reward_list[n])
            #     plt.ylim(-10, 200)
            #     plt.yticks(fontproperties = 'Times New Roman', size = 14)
            #     plt.xticks(range(0, self.num_round, 2), fontproperties = 'Times New Roman', size = 14)
            #     plt.plot(hist_increment_matrix[n], styles[n], label=label)
            #     plt.ylabel(r'Increment $\Delta$', fontdict={'family':'Times New Roman', 'size':16, 'weight':'bold'})
            #     plt.xlabel(r'Round $T$', fontdict={'family':'Times New Roman', 'size':16, 'weight':'bold'})
            #     plt.legend(frameon=False)
            #     plt.savefig(self.save_path_1_delta, dpi=200)
            # plt.close()

            # for n in range(idx+1):
            #     if delta_reward_list[n] < 0:
            #         label = r'$(R^*{}, \theta^*$)'.format(delta_reward_list[n])
            #     elif delta_reward_list[n] == 0:
            #         label = r'$(R^*, \theta^*$)'
            #     else:
            #         label = r'$(R^*+{}, \theta^*$)'.format(delta_reward_list[n])
            #     plt.ylim(-10, 720)
            #     plt.yticks(fontproperties = 'Times New Roman', size = 14)
            #     plt.xticks(range(0, self.num_round, 2), fontproperties = 'Times New Roman', size = 14)
            #     plt.plot(hist_data_matrix[n], styles[n], label=label)
            #     plt.ylabel(r'Datasize $D$', fontdict={'family':'Times New Roman', 'size':16, 'weight':'bold'})
            #     plt.xlabel(r'Round $T$', fontdict={'family':'Times New Roman', 'size':16, 'weight':'bold'})
            #     plt.legend(loc='upper right', frameon=False)
            #     plt.savefig(self.save_path_1_data, dpi=200)
            # plt.close()
            
            # for n in range(idx+1):
            #     if delta_reward_list[n] < 0:
            #         label = r'$(R^*{}, \theta^*$)'.format(delta_reward_list[n])
            #     elif delta_reward_list[n] == 0:
            #         label = r'$(R^*, \theta^*$)'
            #     else:
            #         label = r'$(R^*+{}, \theta^*$)'.format(delta_reward_list[n])
            #     plt.ylim(0, 10)
            #     plt.yticks(fontproperties = 'Times New Roman', size = 14)
            #     plt.xticks(range(0, self.num_round, 2), fontproperties = 'Times New Roman', size = 14)
            #     plt.plot(hist_stale_matrix[n], styles[n], label=label)
            #     plt.ylabel(r'Staleness $S$', fontdict={'family':'Times New Roman', 'size':16, 'weight':'bold'})
            #     plt.xlabel(r'Round $T$', fontdict={'family':'Times New Roman', 'size':16, 'weight':'bold'})
            #     plt.legend(frameon=False)
            #     plt.savefig(self.save_path_1_stale, dpi=200)
            # plt.close()

            # 初始化数据
            self.init_data_net()
            # numpy 切片格式 [:, :]
            data_sum_list = [sum(data_matrix[:, t]) for t in range(self.num_round)]
            # 临时记录，求平均后是self.delta_list
            delta_matrix = [[] for k in range(self.num_client)]
            
            self.global_parameter = {}
            for key, var in self.net.state_dict().items():
                self.global_parameter[key] = var.clone()
            
            global_loss_list = []
            accuracy_list = []
            
            # 训练
            for t in tqdm(range(self.num_round)):
                
                # 开始训练
                next_global_parameter = {}
                local_grad_list = []
                local_loss_list = []
                global_grad = 0
                global_loss = 0
                for k in range(self.num_client):
                    result = self.client_group.clients[k].local_update(t,
                                                                    k,
                                                                    self.num_epoch,
                                                                    self.batch_size,
                                                                    self.global_parameter,
                                                                    new_theta, # 妈的是new的不是旧的
                                                                    increment_matrix[k][t],
                                                                    data_matrix[k][t])
                    
                    rate = data_matrix[k][t] / data_sum_list[t]
                    local_parameter = result[0]
                    for item in local_parameter.items():
                        if item[0] not in next_global_parameter.keys():
                            next_global_parameter[item[0]] = rate * item[1]
                        else:
                            next_global_parameter[item[0]] += rate * item[1]
                            
                    local_grad = result[1]
                    local_grad_list.append(local_grad)
                    global_grad += rate * local_grad
                    
                    local_loss = result[2]
                    local_loss_list.append(local_loss)
                    global_loss += rate * local_loss
                    
                # 求global_parameters
                self.global_parameter = next_global_parameter
                
                # 求delta
                for k in range(self.num_client):
                    delta = rate * torch.sqrt(torch.sum(torch.square(local_grad_list[k] - global_grad))) # 大错，用delta代替Upsilon
                    delta_matrix[k].append(delta)
                    
                # 加入global_loss_list
                global_loss_list.append(global_loss)
                
                # 验证
                if t % self.eval_freq == 0:
                    correct = 0
                    total = 0
                    self.net.load_state_dict(self.global_parameter)
                    with torch.no_grad():
                        test_dataloader = DataLoader(ConcatDataset(self.test_data_list), batch_size=100, shuffle=False)
                        for batch in test_dataloader:
                            data, label = batch
                            data = data.to(self.dev)
                            label = label.to(self.dev)
                            pred = self.net(data) # [batch_size， 10]，输出的是概率
                            pred = torch.argmax(pred, dim=1)
                            correct += (pred == label).sum().item()
                            total += label.shape[0]
                    acc = correct / total
                    accuracy_list.append(acc)
                
                # # 画图2                    
                # if t == 9:
                #     los_list_9.append(global_loss_list[-1])
                #     acc_list_9.append(accuracy_list[-1])   
                    
                #     width = 0.8
                #     x = range(len(delta_reward_list))
                #     labels = [r'$(R^* - 70, \theta^*)$', r'$(R^* - 35, \theta^*)$', r'$(R^*, \theta^*)$', r'$(R^* + 35, \theta^*)$', r'$(R^* + 70, \theta^*)$']
                #     new_gamma = 5e-3
                #     print('los_list_9', los_list_9)
                #     cost_1_list = (1 - new_gamma) * np.array(los_list_9)
                #     cost_2_list = new_gamma * (np.array(delta_reward_list) + reward)
                #     # plt.bar(x - 1.5 * width, cost_list, color='C0', width=width, label='cost')
                #     plt.bar(x, cost_1_list, color='C0', width=width, label='loss')
                #     plt.bar(x, cost_2_list, color='C3', width=width, label='reward', bottom=cost_1_list)
                #     # plt.bar(x + 1.5 * width, acc_list, color='C3', width=width, label='acc')
                #     plt.xticks(x, labels[:len(x)])
                #     plt.ylabel(r'Realistic Cost of Server $U$')
                #     plt.legend(frameon=False)
                #     # if flag == 0:
                #     #     plt.legend()
                #     #     flag = 1
                #     plt.savefig(self.save_path_2_9, dpi=200)
                #     plt.close()
                    
                # if t == 19:
                #     los_list_19.append(global_loss_list[-1])
                #     acc_list_19.append(accuracy_list[-1])   
                    
                #     width = 0.8
                #     x = range(len(delta_reward_list))
                #     labels = [r'$(R^* - 70, \theta^*)$', r'$(R^* - 35, \theta^*)$', r'$(R^*, \theta^*)$', r'$(R^* + 35, \theta^*)$', r'$(R^* + 70, \theta^*)$']
                #     new_gamma = 5e-3
                #     print('los_list_19', los_list_19)
                #     cost_1_list = (1 - new_gamma) * np.array(los_list_19)
                #     cost_2_list = new_gamma * (np.array(delta_reward_list) + reward)
                #     # plt.bar(x - 1.5 * width, cost_list, color='C0', width=width, label='cost')
                #     plt.bar(x, cost_1_list, color='C0', width=width, label='loss')
                #     plt.bar(x, cost_2_list, color='C3', width=width, label='reward', bottom=cost_1_list)
                #     # plt.bar(x + 1.5 * width, acc_list, color='C3', width=width, label='acc')
                #     plt.xticks(x, labels[:len(x)])
                #     plt.ylabel(r'Realistic Cost of Server $U$')
                #     plt.legend(frameon=False)
                #     # if flag == 0:
                #     #     plt.legend()
                #     #     flag = 1
                #     plt.savefig(self.save_path_2_19, dpi=200)
                #     plt.close()
            
            # 画图3
            hist_loss_matrix.append(global_loss_list)
            hist_acc_matrix.append(accuracy_list)
            for n in range(idx+1):
                if delta_reward_list[n] < 0:
                    label = r'$(R^*{}, \theta^*$)'.format(delta_reward_list[n])
                elif delta_reward_list[n] == 0:
                    label = r'$(R^*, \theta^*$)'
                else:
                    label = r'$(R^*+{}, \theta^*$)'.format(delta_reward_list[n])   
                plt.xticks(range(0, self.num_round, 2), fontproperties = 'Times New Roman', size = 14)
                plt.plot(hist_loss_matrix[n], styles[n], color='C{}'.format(n), label=label)
                plt.ylabel('Loss', fontproperties = 'Times New Roman', size = 14)
                plt.xlabel('Round', fontproperties = 'Times New Roman', size = 14)
                plt.legend()
                plt.savefig(self.save_path_3_loss, dpi=200)
            plt.close()
            
            for n in range(idx+1):
                if delta_reward_list[n] < 0:
                    label = r'$(R^*{}, \theta^*$)'.format(delta_reward_list[n])
                elif delta_reward_list[n] == 0:
                    label = r'$(R^*, \theta^*$)'
                else:
                    label = r'$(R^*+{}, \theta^*$)'.format(delta_reward_list[n])   
                plt.xticks(range(0, self.num_round, 2), fontproperties = 'Times New Roman', size = 14)
                plt.plot(hist_acc_matrix[n], styles[n], color='C{}'.format(n), label=label)
                plt.ylabel('Accuracy', fontproperties = 'Times New Roman', size = 14)
                plt.xlabel('Round', fontproperties = 'Times New Roman', size = 14)
                plt.legend()
                plt.savefig(self.save_path_3_acc, dpi=200)
            plt.close()
                         
        
server = Server(args)
server.online_train()

# R(T) = 0怎么保证：最后一轮R强制为0
# 初始化你搞的好一点
# 注意不收敛的问题