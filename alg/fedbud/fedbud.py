import os
from tkinter import NO

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim

import numpy as np
import pandas as pd
import random
import math
from tqdm import tqdm
from sko.GA import GA
import matplotlib.pyplot as plt

from alg.Clients import ClientsGroup
from model.Mnist import Mnist_2NN, Mnist_CNN, LinearNet
from model.Cifar import Cifar10_CNN, CIFAR10_Model

def test_mkdir(path):
    if not os.path.isdir(path):
        # os.mkdir(path)
        os.makedirs(path)

args = {
    # 模型
    'alg': 'fedbud',
    'model_name': 'cifar10_cnn',
    'model_save_path': '../checkpoints',
    'log_save_path': '../logs/logs.txt',
    'save_freq': 10,
    'val_freq': 10,
    'gpu': '0',
    
    # 不动点相关
    'phi_init': 100,
    'num_iteration': 50,
    'eps': 1e-7,
    
    # 通信/客户端/数据集
    'num_comm': 2,
    'num_client': 10,
    'dataset': 'cifar10',
    'IID': 1,
    'dilich': 1.0, # 迪利克雷系数
    'resize': 32, # cifar10图像大小调整
    'is_local_acc': 0,
    'epoch': 5,
    'batchsize': 16,
    'learning_rate': 0.01,
    
    # 资源限制条件约束
    'V': 1,
    'W': 0,
    'l_energy': 10, # 1 / 10 / 100
    'u_energy': 30, # 3 / 30 / 300
    'l_privacy': 10, # 1 / 10 / 100
    'u_privacy': 30, # 3 / 30 / 300
    'kappa': 0.9,
    'gamma': 1e-10, # 平衡第一部分和第三部分
    
    'xi': 0.5,
    'c': 1,
    'f': 1,
}


class FEDBUDServer(object):
    
    def __init__(self, args):
        self.log_save_path = args['log_save_path']
        self.model_save_path = args['model_save_path']
        os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
        self.args = args
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        # 模型
        self.net = None
        self.num_outputs = 10
        self.model = args['model_name']
        if self.model == 'mnist_2nn':
            self.net = Mnist_2NN(num_outputs=self.num_outputs)
        elif self.model == 'mnist_cnn':
            self.net = Mnist_CNN(num_outputs=self.num_outputs)
        elif self.model == 'mnist_linear':
            self.net = LinearNet(num_outputs=self.num_outputs)
        elif args['model_name'] == 'cifar10_cnn':
            self.net = Cifar10_CNN(num_classes=self.num_outputs)
        elif args['model_name'] == 'cifar10_model':
            self.net = CIFAR10_Model(num_classes=self.num_outputs)

        if torch.cuda.device_count() > 1:
            self.net = torch.nn.DataParallel(self.net)

        # 初始全局参数
        self.global_parameters = {}

        for m in self.net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        for key, var in self.net.state_dict().items():
            self.global_parameters[key] = var.clone()
        
        self.net = self.net.to(self.dev)
        # 损失函数
        self.loss_func = F.cross_entropy
        # 优化器
        self.optimers = {}
        for i in range(args['num_client']):
            self.optimers['client{}'.format(i)] = optim.SGD(self.net.parameters(), lr=args['learning_rate'])
            
        # ----------------------------------------------------------------------------------
        # 不动点算法相关
        self.phi_init = args['phi_init'] # 不动点算法的初始值
        self.num_iteration = args['num_iteration'] # 不动点算法的最大迭代次数
        self.eps = args['eps'] # 不动点算法容许的误差
        
        # 定义通信次数
        self.num_comm = args['num_comm']
        
        # 定义客户端和数据
        self.num_client = args['num_client']
        self.client_set = [i for i in range(0, args['num_client'])]
        self.dataset = args['dataset']
        self.isIID = args['IID']
        self.dilich = args['dilich'] if 'dilich' in args else 1.0
        self.resize = args['resize'] if 'resize' in args else 32
        self.split = args["split"] if "split" in args else "letters"
        self.rearrange = args["rearrange"] if "rearrange" in args else 0
        self.myClients = ClientsGroup(
                                    dataSetName=self.dataset, 
                                    isIID=self.isIID, 
                                    numOfClients=self.num_client, 
                                    dev=self.dev, 
                                    alpha=self.dilich, 
                                    resize=self.resize, 
                                    split=self.split,  
                                    )
        self.testDataLoader = self.myClients.test_data_loader
        self.is_local_acc = args['is_local_acc'] if 'is_local_acc' in args else False
        
        # 客户端的其他参数
        self.epoch = args['epoch']
        self.batchsize = args['batchsize']
        self.lr = args['learning_rate']
        self.V = args['V']
        self.W = args['W']
        self.Q = [self.W for _ in range(self.num_client)]
        self.Z = [self.W for _ in range(self.num_client)]
        self.xi = [args['xi'] for _ in range(self.num_client)]
        self.c = [args['c'] for _ in range(self.num_client)]
        self.f = [args['f'] for _ in range(self.num_client)]
        self.alpha = np.random.uniform(0.01, 0.05, self.num_client)
        self.beta = np.random.uniform(0.01, 0.05, self.num_client)
        self.n = np.random.uniform(args['l_energy'] * self.num_comm, args['u_energy'] * self.num_comm, self.num_client)
        self.m = np.random.uniform(args['l_privacy'] * self.num_comm, args['u_privacy'] * self.num_comm, self.num_client)
        self.kappa = args['kappa']
        self.gamma = args['gamma']
        
    
    def find_B_E(self, phi, R, t):
        list_B = []
        list_E = []
        for k in range(self.num_client):
            X = (self.Q[k] * self.xi[k] * self.c[k] * self.f[k] ** 2)  / (4 * self.V * self.alpha[k])
            Y = (self.Z[k]) / (4 * self.V * self.beta[k])
            B = math.sqrt(R / (2 * self.alpha[k] * phi) + X ** 2) - X
            E = math.sqrt(R / (2 * self.beta[k] * phi) + Y ** 2) - Y
            list_B.append(B)
            list_E.append(E)
            
        return {'list_B': list_B, 'list_E':list_E}
    
    
    def find_R(self, phi, t):
        
        def C(input):
            R = input[0]
            
            result = self.find_B_E(phi, R, t)
            list_B = result['list_B']
            list_E = result['list_E']
            
            sum_B = 0
            sum_E = 0
            for k in range(self.num_client):
                sum_B += list_B[k]
                sum_E += 1 / list_E[k]
            part_1 = self.kappa * (1 / sum_B**2) * sum_E
            
            sum_C = 0
            for k in range(self.num_client):
                sum_C += self.c[k] * list_B[k] / self.f[k]
            part_2 = 0
            for k in range(self.num_client):
                item = (self.c[k] * list_B[k] / self.f[k]) / sum_C
                part_2 += item * math.log(item)

            part_3 = self.gamma * R
            # print('t:', t, 'phi:', phi, 'R:', R, 'part_1:', part_1, 'part_2:', part_2, 'part_3:', part_3, 'sum_B:', sum_B, 'sum_E:', sum_E)
            return part_1 + part_3
        
        ga = GA(
            func=C, # 目标函数
            n_dim=1, # 控制变量个数
            size_pop=50, # 种群规模
            max_iter=50,  # 最大迭代次数
            prob_mut=0.01, # 变异概率
            lb=0, # 定义域
            ub=1e7,  # 值域
            precision=1e-5, # 精度
            )
        R = ga.run()
        
        X_history = ga.generation_best_X
        return R, X_history
    
    
    def find_phi(self, t):
        list_phi = []
        list_R = []
        phi = self.phi_init
        # 多次迭代
        for i in range(self.num_iteration):
            list_phi.append(phi)
            # 得到R
            tmp = self.find_R(phi, t)
            R = tmp[0][0][0]
            X_history = tmp[1]
            list_R.append(R)
            # 计算得到所有B和E，并得到phi_new
            phi_new = 0
            for k in range(self.num_client):
                X = (self.Q[k] * self.xi[k] * self.c[k] * self.f[k] ** 2)  / (4 * self.V * self.alpha[k])
                Y = (self.Z[k]) / (4 * self.V * self.beta[k])
                B = math.sqrt(R / (2 * self.alpha[k] * phi) + X ** 2) - X
                E = math.sqrt(R / (2 * self.beta[k] * phi) + Y ** 2) - Y
                phi_new += math.log(B * E)
            if abs(phi_new - phi) <= self.eps:
                # plt.subplot(1, 2, 1)
                # plt.plot(list_phi)
                # plt.xlabel('iter')
                # plt.ylabel('phi*')
                
                # plt.subplot(1, 2, 2)
                # plt.plot(list_R)
                # plt.xlabel('iter')
                # plt.ylabel('R*')
                # plt.show()
                return phi_new
            phi = phi_new
        return phi
    
    
    def run(self):
        # 初始化
        list_phi = []
        list_R = []
        list_list_B = [[] for k in range(self.num_client)]
        list_list_E = [[] for k in range(self.num_client)]
        list_list_Q = [[self.W] for k in range(self.num_client)]
        list_list_Z = [[self.W] for k in range(self.num_client)]        
        
        accuracy_list = []
        loss_list = []
        accuracy, loss = self.eval(t=0)
        accuracy_list.append(accuracy)
        loss_list.append(loss)

        # 迭代过程
        for t in tqdm(range(self.num_comm)):
            # 每次都要算！队列的更新
            phi = self.find_phi(t)
            R = self.find_R(phi, t)[0][0]
            list_phi.append(phi)
            list_R.append(R)
            
            result = self.find_B_E(phi, R, t)
            list_B_temp = result['list_B']
            list_B = [int(item) for item in list_B_temp]
            list_E = result['list_E']
            theta_list = [list_B[k] / sum(list_B) for k in range(self.num_client)]
            for k in range(self.num_client):
                list_list_B[k].append(list_B[k])
                list_list_E[k].append(list_E[k])
            
            sum_parameters = None
            for k in range(self.num_client):
                local_parameters = self.myClients.clients_set['client'+str(k)].localPartialUpdate(
                                                                                self.epoch, 
                                                                                self.batchsize, 
                                                                                self.net,
                                                                                self.loss_func, 
                                                                                self.optimers['client'+str(k)], 
                                                                                self.global_parameters,
                                                                                list_B[k],
                                                                                list_E[k],
                                                                                )
                # # 验证本地模型
                # if self.is_local_acc and t == self.num_comm - 1:
                #     self.local_eval(t=t, local_parameters=local_parameters, client=k)
                
                # 更新队列
                self.Q[k] = max(self.Q[k] + self.xi[k] * self.c[k] * self.f[k] ** 2 * list_B[k] - self.n[k] / self.num_comm, 0)
                self.Z[k] = max(self.Z[k] + list_E[k] - self.m[k] / self.num_comm, 0)
                list_list_Q[k].append(self.Q[k])
                list_list_Z[k].append(self.Z[k])
                
                if sum_parameters is None:
                    sum_parameters = {}
                    for key, var in local_parameters.items():
                        sum_parameters[key] = theta_list[k] * var.clone()
                else:
                    for key in sum_parameters:
                        sum_parameters[key] = sum_parameters[key] + theta_list[k] * local_parameters[key]
                
            for key in self.global_parameters:
                self.global_parameters[key] = sum_parameters[key]

            accuracy, loss = self.eval(t=t)
            accuracy_list.append(accuracy)
            loss_list.append(loss)

            if t % 10 == 0:
                # 绘制acc和loss
                plt.subplot(2, 4, 1)
                plt.plot(accuracy_list)
                plt.xlabel('rounds')
                plt.ylabel('acc')
                
                plt.subplot(2, 4, 5)
                plt.plot(loss_list)
                plt.xlabel('rounds')
                plt.ylabel('loss')
                
                # 绘制phi和R
                plt.subplot(2, 4, 2)
                plt.plot(list_phi)
                plt.xlabel('iterations')
                plt.ylabel('phi')
                
                plt.subplot(2, 4, 6)
                plt.plot(list_R)
                plt.xlabel('iterations')
                plt.ylabel('R')
                
                choice = np.random.randint(0, len(list_list_B), 3)
                # 绘制list_Q和list_Z
                plt.subplot(2, 4, 3)
                plt.plot(list_list_Q[choice[0]], label='client_{}'.format(choice[0]))
                plt.plot(list_list_Q[choice[1]], label='client_{}'.format(choice[1]))
                plt.plot(list_list_Q[choice[2]], label='client_{}'.format(choice[2]))
                plt.legend()
                plt.xlabel('rounds')
                plt.ylabel('Queue_Q')
       
                plt.subplot(2, 4, 7)
                plt.plot(list_list_Z[choice[0]], label='client_{}'.format(choice[0]))
                plt.plot(list_list_Z[choice[1]], label='client_{}'.format(choice[1]))
                plt.plot(list_list_Z[choice[2]], label='client_{}'.format(choice[2]))
                plt.legend()
                plt.xlabel('rounds')
                plt.ylabel('Queue_Z')
        
                #绘制list_B和list_E
                x = np.arange(t + 1)
                width = 0.25
                plt.subplot(2, 4, 4)
                plt.bar(x - width, list_list_B[choice[0]], width, label='client_{}'.format(choice[0]))
                plt.bar(x, list_list_B[choice[1]], width, label='client_{}'.format(choice[1]))
                plt.bar(x + width, list_list_B[choice[2]], width, label='client_{}'.format(choice[2]))
                plt.legend()
                plt.xlabel('rounds')
                plt.ylabel('B')                
                
                # plt.subplot(2, 4, 8)
                # for k in range(len(list_list_E)):
                #     plt.bar(x, list_list_E[k])
                # plt.legend()
                # plt.xlabel('rounds')
                # plt.ylabel('E')
                
                plt.show()
        
        # 记录结果
        s = 'accuracy_list:{}\n loss_list:{}\n list_Q:{}\n list_Z:{}\n list_B:{}\n list_E:{}\n'.format(
                                                                                                    accuracy_list,
                                                                                                    loss_list,
                                                                                                    list_list_Q,
                                                                                                    list_list_Z,
                                                                                                    list_list_B,
                                                                                                    list_list_E)
        with open(self.log_save_path, 'a', encoding = 'utf-8') as f:   
            f.write(s)
            f.write('---------------------------------------------------------------')
        
    def eval(self, t, is_loss=False):
        sum_accu, num = 0, 0
        self.net.load_state_dict(self.global_parameters, strict=True)
        with torch.no_grad():
            loss_list = []
            for data, label in self.testDataLoader:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = self.net(data)
                loss = self.loss_func(preds, label)
                loss_list.append(loss.mean())
                preds = torch.argmax(preds, dim=1)
                sum_accu += (preds == label).float().mean()
                num += 1
            avg_loss = float(sum(loss_list) / num)
            accuracy = float(sum_accu / num)
        if (t + 1) % self.args['save_freq'] == 0:
            torch.save(self.net, os.path.join(self.model_save_path, '{}_{}_num_comm{}_E{}_B{}_lr{}_num_clients{}'.format(
                                                                                                                                self.args['alg'], 
                                                                                                                                self.args['model_name'],
                                                                                                                                t, 
                                                                                                                                self.args['epoch'], 
                                                                                                                                self.args['batchsize'], 
                                                                                                                                self.args['learning_rate'], 
                                                                                                                                self.args['num_client'],
                                                                                                                                )))
        return accuracy, avg_loss
    
    
    def local_eval(self, t, local_parameters, client):
        sum_accu, num = 0, 0
        self.net.load_state_dict(local_parameters, strict=True)
        with torch.no_grad():
            for data, label in self.testDataLoader:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = self.net(data)
                preds = torch.argmax(preds, dim=1)
                sum_accu += (preds == label).float().mean()
                num += 1
            accuracy = float(sum_accu / num)

        acc_avg_file = os.path.join(self.log_save_dir, 'loc_acc.txt')
        with open(acc_avg_file, 'a') as f:
            f.write('t = {}, client = {}, acc = {}\n'.format(t, client, accuracy))


if __name__=="__main__":
    fedbud = FEDBUDServer(args)                      
    fedbud.run()
