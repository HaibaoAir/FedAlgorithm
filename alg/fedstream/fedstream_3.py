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
    'save_path': '../../logs/fedavg/3.png',
    'save_path_3D': '../../logs/fedavg/3_3D.png',
    
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
    'reward_lb': 50,
    'reward_ub': 60,
    'theta_lb': 0.4,
    'theta_ub': 0.5,
    'pop': 3000, 
    'pso_eps': 1e-5,
    'pso_max_iter': 500,
    
    'fix_eps_1': 1e-2,
    'fix_eps_2': 3,
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
        print(1)
        self.client_group = Client_Group(self.dev,
                                         self.num_client,
                                         self.dataset_name,
                                         self.is_iid,
                                         self.a,
                                         self.net_name,
                                         self.learning_rate,
                                         )
        print(2)
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
        self.save_path_3D = args['save_path_3D']

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
            
            tmp = 0
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
                tmp += (item_1 + item_2 + item_3)
            return tmp, res
        
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
            increment_matrix, data_matrix, stale_matrix = self.estimate_D(phi_list, reward, theta)
            print('******************************************************')
            print('{}, {}, {}, {}, {}'.format(idc, phi_list, reward, theta, res))
            print('{}'.format(data_matrix))
            
            next_phi_list = np.sum(data_matrix * ((1 / self.delta_list).reshape(self.num_client, 1)), axis=0)
            # 判断收敛
            max_diff = np.max(np.abs(next_phi_list - phi_list))
            print('max_diff_phi:{}'.format(max_diff))
            
            if max_diff > self.fix_eps_2:
                phi_list = next_phi_list 
                
                if idc == 0 or idc == 1:
                    continue
                
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
                
                # 绘制三维图
                x1 = np.arange(self.reward_lb, self.reward_ub, (self.reward_ub - self.reward_lb) / 100)
                x2 = np.arange(self.theta_lb, self.theta_ub, (self.theta_ub - self.theta_lb) / 100)
                X1, X2 = np.meshgrid(x1, x2)
                tmp_matrix = []
                res_matrix = []
                for i in range(len(x2)):
                    tmp_list = []
                    res_list = []
                    for j in range(len(x1)):
                        tmp, res = func(phi_list, X1[i][j], X2[i][j])
                        tmp_list.append(tmp)
                        res_list.append(res)
                    tmp_matrix.append(tmp_list)
                    res_matrix.append(res_list)
                tmp_matrix = np.array(tmp_matrix)
                res_matrix = np.array(res_matrix)
                
                fig = plt.figure()
                ax4 = fig.add_subplot(1,2,1, projection='3d')
                ax4.set_xlabel('reward')
                ax4.set_ylabel('theta')
                ax4.set_zlabel('loss')
                ax4.plot_surface(X1, X2, tmp_matrix, rstride=1, cstride=1, cmap='rainbow')
                ax4.contour(X1, X2, tmp_matrix)
                ax5 = fig.add_subplot(1,2,2, projection='3d')
                ax5.set_xlabel('reward')
                ax5.set_ylabel('theta')
                ax5.set_zlabel('cost')
                ax5.set_zlim(0.185, 0.195)

                ax5.plot_surface(X1, X2, res_matrix, rstride=1, cstride=1, cmap='rainbow')
                ax5.contour(X1, X2, tmp_matrix)
                plt.tight_layout()
                plt.savefig(self.save_path_3D, dpi=200)
                
                print('triumph2')
                return next_phi_list, reward, theta, res, increment_matrix, data_matrix, stale_matrix
            
        print('failure2')
        return next_phi_list


    def online_train(self):
        
        # 正式训练前定好一切
        var = self.estimate_phi() # [T]
        phi_list = var[0]
        reward = var[1]
        theta = var[2]
        res = var[3]
        increment_matrix = var[4]
        data_matrix = var[5]
        stale_matrix = var[6]
        exit(0)
        # print(data_matrix)
        
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
            
            # plt.subplot(2,4,5)
            # plt.plot(accuracy_list)
            # plt.savefig(self.save_path)
        
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