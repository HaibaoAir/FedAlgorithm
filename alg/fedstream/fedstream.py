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
        self.delta_list = np.array([1] * self.num_client) # [K]
        self.sigma_list = np.array([1] * self.num_client)
        self.psi = 1
        self.alpha_list = [1e-3] * self.num_client # 收集数据的价格
        self.beta_list = [5e-6] * self.num_client # 训练数据的价格 如果稍大变负数就收敛不了，如果是0就没有不动点的意义
        self.fix_eps_1 = 0.1
        self.fix_eps_2 = 5
        self.fix_max_iter = 1000
        
        self.kappa_1 = 1
        self.kappa_2 = 1
        self.kappa_3 = 1e-2
        self.kappa_4 = 1
        self.gamma = 1e-4
        
        # 一阶段
        self.reward_lb = 1
        self.reward_ub = 100
        self.theta_lb = 0
        self.theta_ub = 1
        
        # # 二阶段
        # self.reward_lb = 50
        # self.reward_ub = 60
        # self.theta_lb = 0.4
        # self.theta_ub = 0.5
        self.pop = 500 # 探索空间划定得越大，需要的粒子就多点，就能找到精确最优解！
        self.pso_eps = 1e-10
        self.pso_max_iter = 500
              
    
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
        
        def func(phi_list, reward, theta):
            # 计算单列和
            increment_matrix, data_matrix, stale_matrix = self.estimate_D(phi_list, reward, theta) 
            data_list = [] # [T]
            for t in range(self.num_round):
                data = 0
                for k in range(self.num_client):
                    data += data_matrix[k][t]
                data_list.append(data)
            
            res_1 = 0
            res_2 = 0
            res_3 = 0
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
                res_1 += item_1
                res_2 += item_2
                res_3 += item_3
            return res_1, res_2, res_3, res, np.sum(increment_matrix), np.sum(data_matrix)
        
        # 初始化phi_list
        phi_list = self.phi_list_init
        phi_hist = []
        reward_hist = []
        theta_hist = []
        last_reward = 0
        last_theta = 0
        flag = 0
        # 计算新的phi_list[T]
        for idc in range(self.fix_max_iter):
            var, res = self.estimate_reward_theta(phi_list)
            reward, theta = var
            if flag == 0 and abs(reward - last_reward) <= 3 and abs(theta - last_theta) <= 0.05:
                print('model')
                time.sleep(5)
                self.reward_lb = reward - 7
                self.reward_ub = reward + 7
                self.theta_lb = theta - 0.1
                self.theta_ub = theta + 0.1
                self.pop = self.pop * 10
                flag = 1
            last_reward = reward
            last_theta = theta
            increment_matrix, data_matrix, _= self.estimate_D(phi_list, reward, theta)
            print('******************************************************')
            print('{}, {}, {}, {}, {}'.format(idc, phi_list, reward, theta, res))
            print('{}'.format(data_matrix))
            
            next_phi_list = np.sum(data_matrix * ((1 / self.delta_list).reshape(self.num_client, 1)), axis=0)
            # 判断收敛
            if np.max(np.abs(next_phi_list - phi_list)) > self.fix_eps_2:
                phi_list = next_phi_list 
                
                if idc == 0 or idc == 1:
                    continue
                x1 = np.arange(self.reward_lb, self.reward_ub, 1)             
                y1 = []
                y2 = []
                y3 = []
                y = []
                F1 = []
                F2 = []
                for i in x1:
                    e1, e2, e3, e, f1, f2 = func(phi_list, i, theta)
                    
                    y1.append(e1)
                    y2.append(e2)
                    y3.append(e3)
                    y.append(e)
                    F1.append(f1)
                    F2.append(f2)
                plt.subplot(3,6,1)
                plt.plot(x1, y1)
                plt.subplot(3,6,2)
                plt.plot(x1, y2)
                plt.subplot(3,6,3)
                plt.plot(x1, y3)
                plt.subplot(3,6,4)
                plt.plot(x1, y)
                plt.subplot(3,6,5)
                plt.plot(x1, F1)
                plt.subplot(3,6,6)
                plt.plot(x1, F2)
                
                x2 = np.arange(self.theta_lb, self.theta_ub, 0.1)         
                y1 = []
                y2 = []
                y3 = []
                y = []
                F1 = []
                F2 = []
                for i in x2:
                    e1, e2, e3, e, f1, f2 = func(phi_list, reward, i)
                    y1.append(e1)
                    y2.append(e2)
                    y3.append(e3)
                    y.append(e)
                    F1.append(f1)
                    F2.append(f2)
                    
                plt.subplot(3,6,7)
                plt.plot(x2, y1)
                plt.subplot(3,6,8)
                plt.plot(x2, y2)
                plt.subplot(3,6,9)
                plt.plot(x2, y3)
                plt.subplot(3,6,10)
                plt.plot(x2, y)
                plt.subplot(3,6,11)
                plt.plot(x2, F1)
                plt.subplot(3,6,12)
                plt.plot(x2, F2)
                
                phi_hist.append(phi_list)
                reward_hist.append(reward)
                theta_hist.append(theta)
                plt.subplot(3,6,13)
                plt.plot(phi_hist)
                plt.subplot(3,6,14)
                plt.plot(reward_hist)
                plt.subplot(3,6,15)
                plt.plot(theta_hist)
                
                plt.savefig('../../logs/fedavg/3.png')
                
                
                # fig = plt.figure()
                # X1, X2 = np.meshgrid(x1, x2)
                # # print(len(x1))
                # # print(len(x2))
                # # print(X1.shape)
                # # print(X2.shape)
                # # exit(0)
                # y1_matrix = []
                # y2_matrix = []
                # y3_matrix = []
                # y_matrix = []
                # for i in range(len(x2)):
                #     y1_list = []
                #     y2_list = []
                #     y3_list = []
                #     y_list = []
                #     for j in range(len(x1)):
                #         e1, e2, e3, e, _ = func(phi_list, X1[i][j], X2[i][j])
                #         y1_list.append(e1)
                #         y2_list.append(e2)
                #         y3_list.append(e3)
                #         y_list.append(e)
                #     y1_matrix.append(y1_list)
                #     y2_matrix.append(y2_list)
                #     y3_matrix.append(y3_list)
                #     y_matrix.append(y_list)
                # y1_matrix = np.array(y1_matrix)
                # y2_matrix = np.array(y2_matrix)
                # y3_matrix = np.array(y3_matrix)
                # y_matrix = np.array(y_matrix)
                # ax1 = fig.add_subplot(3,5,11, projection='3d')
                # ax1.plot_surface(X1, X2, y1_matrix, rstride=1, cstride=1, cmap='rainbow')
                # ax2 = fig.add_subplot(3,5,12, projection='3d')
                # ax2.plot_surface(X1, X2, y2_matrix, rstride=1, cstride=1, cmap='rainbow')
                # ax3 = fig.add_subplot(3,5,13, projection='3d')
                # ax3.plot_surface(X1, X2, y3_matrix, rstride=1, cstride=1, cmap='rainbow')
                # ax4 = fig.add_subplot(3,5,14, projection='3d')
                # ax4.plot_surface(X1, X2, y_matrix, rstride=1, cstride=1, cmap='rainbow')
                # fig.savefig('../../logs/fedavg/3.png')
                
            else:
                # c = [[1] * self.num_round] * self.num_client
                # for k in range(self.num_client):
                #     for t in range(1, self.num_round):
                #         c[k][t] = c[k][t-1] * i * b[k][t-1] / b[k][t] + 1
                
                x = np.arange(0, 1, 0.05)
                for i in x:
                    a, b, c = self.estimate_D(phi_list, reward, i)
                    plt.subplot(3,6,16)
                    plt.plot(a[1])
                    plt.subplot(3,6,17)
                    plt.plot(b[1])
                    plt.subplot(3,6,18)
                    plt.plot(c[1])
                    plt.savefig('../../logs/fedavg/3.png')
                with open('../../logs/fedavg/phi_{}.txt'.format(self.reward_ub), 'a') as file:
                    for item in phi_hist:
                        file.write('{}\n'.format(item))   
                    file.write('------------------')                 
                print('triumph2')
                return next_phi_list
            
        print('failure2')
        return next_phi_list


    def online_train(self):
        
        # 正式训练前定好一切
        phi_list = self.estimate_phi() # [T]
        var, res = self.estimate_reward_theta(phi_list) # [K, T]
        reward = var[0]
        theta = var[1]
        _, data_matrix, _ = self.estimate_D(phi_list, reward, theta)
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
                
            plt.subplot(3,6,18)
            plt.plot(accuracy_list)
            plt.savefig('../../logs/fedavg/3.png')
        
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