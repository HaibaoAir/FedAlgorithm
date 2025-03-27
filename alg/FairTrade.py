import sys
sys.path.append('../')
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

from util.Server import Server

args = {
    'num_client': 3,
    'num_sample': 3,
    'dataset': 'adult',
    'is_iid': 1,
    'alpha': 1,
    'model': 'mlp',
    'learning_rate': 0.01,
    'num_round': 100,
    'num_epoch': 3,
    'batch_size': 32,
    'eval_freq': 1,
    
    'beta': 0.5, 
}

class FairTrade_Server(Server):
    def __init__(self, args):
        super(FairTrade_Server, self).__init__(args)
        self.beta = args['beta']
        self.factor_list = []
        
    def run(self):
        '''
        改造损失函数，弱势群体的损失优化力度大
        '''
        # 预训练
        factor_list = []
        for idx in range(self.num_client):
            factor = self.client_group.clients[idx].FairBatch_pre_update()
            factor_list.append(factor)
        
        # 正式训练
        self.global_parameter = {}
        for key, var in self.net.state_dict().items():
            self.global_parameter[key] = var.clone()
            
        accuracy_list = []
        equopp_list = []
        for round in tqdm(range(self.num_round)):
            next_global_parameter = {}
            diff_list = []
            for idx in range(self.num_client):
                local_parameter, diff = self.client_group.clients[idx].FairBatch_local_update(
                                                                                self.num_epoch,
                                                                                self.batch_size,
                                                                                self.global_parameter,
                                                                                factor,
                                                                                )
                for item in local_parameter.items():
                    if item[0] not in next_global_parameter.keys():
                        next_global_parameter[item[0]] = self.weight_list[idx] * item[1]
                    else:
                        next_global_parameter[item[0]] += self.weight_list[idx] * item[1]
                
                diff_list.append(diff)
                
            # 更新全局参数
            self.global_parameter = next_global_parameter
            
            # 更新factor
            diff_list = torch.tensor(diff_list)
            diff_norm = torch.norm(diff_list, 2)
            for idx in range(self.num_client):
                factor_list[idx] = factor_list[idx] + self.beta * diff_list[idx] / diff_norm
            equopp_list.append(torch.mean(diff_list))
            print(equopp_list)
            
            # 开始验证
            if self.num_round % self.eval_freq == 0:
                # 加载模型
                self.net.load_state_dict(self.global_parameter)
                
                # 验证
                correct = 0
                total = 0
                with torch.no_grad():
                    for batch in self.test_dataloader:
                        data, label, sign = batch
                        data = data.to(self.dev)
                        label = label.to(self.dev)
                        pred = self.net(data)
                        if args['dataset'] == 'adult':
                            pred = (pred > 0.5).float() # [batch_size， 1]，输出的是概率
                        elif args['dataset'] == 'mnist' or args['dataset'] == 'fmnist':
                            pred = torch.argmax(pred, dim=1) # [batch_size， 10]，输出的是概率
                        else:
                            raise NotImplementedError('{}'.format(args['dataset']))
                        correct += (pred == label).sum().item()
                        total += label.shape[0]
                acc = correct / total
                accuracy_list.append(acc)
                print(accuracy_list)
                    
        with open('../logs/fedavg/accuracy.txt', 'a') as file:
            file.write('{}\n'.format(time.asctime()))
            for accuracy in accuracy_list:
                file.write('{:^7.5f} '.format(accuracy))
            file.write('\n')
        
        plt.plot(np.arange(0, self.num_round, self.eval_freq), accuracy_list)
        plt.title('Acc')
        plt.savefig('../logs/fedavg/Acc_2.jpg')
        plt.close()
        
        plt.plot(np.arange(0, self.num_round, self.eval_freq), equopp_list)
        plt.title('EOD')
        plt.savefig('../logs/fedavg/EOD_2.jpg')
        
server = FairTrade_Server(args)
server.run()