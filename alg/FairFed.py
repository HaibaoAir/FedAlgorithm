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
    'is_iid': 0,
    'alpha': 1,
    'model': 'mlp',
    'learning_rate': 0.001,
    'num_round': 20,
    'num_epoch': 3,
    'batch_size': 32,
    'eval_freq': 1,
    
    'beta': 1,
}

class FairFed_Server(Server):
    def __init__(self, args):
        super(FairFed_Server, self).__init__(args)
        self.beta = args['beta']
        
    def run(self):
        '''
        改造聚合权重，不公平的客户端权重越小
        '''
        self.global_parameter = {}
        for key, var in self.net.state_dict().items():
            self.global_parameter[key] = var.clone()
        
        sum_score_protected = 0
        sum_score_non_protected = 0
        sum_tp_fn_protected = 0
        sum_tp_fn_non_protected = 0
        local_score_list = []
        
        
        accuracy_list = []
        equoop_list = []
        for round in tqdm(range(self.num_round)):
            next_global_parameter = {}
            
            for idx in range(self.num_client):
                local_parameter, res = self.client_group.clients[idx].FairFed_local_update(
                                                                                self.num_epoch,
                                                                                self.batch_size,
                                                                                self.global_parameter,
                                                                                )
                for item in local_parameter.items():
                    if item[0] not in next_global_parameter.keys():
                        next_global_parameter[item[0]] = self.weight_list[idx] * item[1]
                    else:
                        next_global_parameter[item[0]] += self.weight_list[idx] * item[1]
                
                sum_score_protected += self.weight_list[idx] * res[0]
                sum_score_non_protected += self.weight_list[idx] * res[1]
                sum_tp_fn_protected += res[2]
                sum_tp_fn_non_protected += res[3]
                local_score_list.append(res[4])
            
            # 更新全局参数    
            self.global_parameter = next_global_parameter
            
            # 计算全局EOD得分
            part_1 = sum_tp_fn_protected / sum(self.scale_list)
            part_2 = sum_tp_fn_non_protected / sum(self.scale_list)
            global_score = (1 / part_2) * sum_score_non_protected - (1 / part_1) * sum_score_protected
            equoop_list.append(global_score)
            
            delta_score_list = [abs(global_score - local_score) for local_score in local_score_list]
            avg_delta_score = sum(delta_score_list) / len(delta_score_list)
            
            # 更新权重
            delta_weight_list = [self.beta * (delta_score - avg_delta_score) for delta_score in delta_score_list]
            self.weight_list = [max(self.weight_list[idx] - delta_weight_list[idx], 0) for idx in range(self.num_client)]
            self.weight_list = [(weight / sum(self.weight_list)) for weight in self.weight_list] # 归一化
            print(self.weight_list)
            
            # 开始验证
            if self.num_round % self.eval_freq == 0:  
                correct = 0
                total = 0
                with torch.no_grad():
                    for batch in self.test_dataloader:
                        data, label = batch
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
                # print(accuracy_list)
                    
        with open('../logs/fedavg/accuracy.txt', 'a') as file:
            file.write('{}\n'.format(time.asctime()))
            for accuracy in accuracy_list:
                file.write('{:^7.5f} '.format(accuracy))
            file.write('\n')
        
        plt.plot(np.arange(0, self.num_round, self.eval_freq), accuracy_list)
        plt.title('Acc')
        plt.savefig('../logs/fedavg/Acc_2.jpg')
        plt.close()
        
        plt.plot(np.arange(0, self.num_round, self.eval_freq), equoop_list)
        plt.title('EOD')
        plt.savefig('../logs/fedavg/EOD_2.jpg')
        
server = FairFed_Server(args)
server.run()