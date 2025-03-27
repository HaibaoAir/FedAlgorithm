import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from model.adult import Adult_MLP

class Local_Dataset(Dataset):
    def __init__(self,
                 local_data,
                 local_label,
                 trans,
                 ):
        self.local_data = local_data
        self.local_label = local_label
        self.trans = trans
        self.init_sign(2)
    
    def init_sign(self, protected_idx):
        self.signs = []
        
        protected_attr = self.local_data[:][:,protected_idx]
        labels = self.local_label[:]
        
        saValue = 0
        for i in range(len(protected_attr)):
            if labels[i] == 1:
                if protected_attr[i] == saValue:
                    self.signs.append(0)
                else:
                    self.signs.append(1)
            else:
                self.signs.append(2)
        
        self.negative_rate = self.signs.count(2) / len(self.signs)
        self.positive_protected_rate = self.signs.count(0) / len(self.signs)
        self.positive_non_protected_rate = self.signs.count(1) / len(self.signs)
            
    def __getitem__(self, idx):
        data = self.trans(self.local_data[idx])
        label = self.local_label[idx]
        sign = self.signs[idx]
        return data, label, sign
    
    def __len__(self):
        return self.local_label.shape[0]


class Client(object):
    def __init__(self, 
                 dataset,
                 dev):
        self.dataset = dataset
        self.dev = dev
        
        self.net = Adult_MLP(101)
        self.net.to(self.dev)
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.optimizer = torch.optim.Adam(self.net.parameters(), 0.01)
        
    def FairBatch_pre_update(self):
        return self.dataset.positive_protected_rate
        
    def FairBatch_local_update(self,
                               num_epoch,
                               batch_size,
                               global_parameters,
                               factor,
                               ):
        
        # 加载模型
        self.net.load_state_dict(global_parameters, strict=True)
        
        # 训练模型
        epoch_loss_protected_list = []
        epoch_loss_non_protected_list = []
        dataloader = DataLoader(self.dataset, batch_size, True)
        for epoch in range(num_epoch):
            epoch_loss_protected = 0
            epoch_loss_non_protected = 0
            for batch in dataloader:
                data, label, sign = batch
                data = data.to(self.dev)
                label = label.to(self.dev)
                pred = self.net(data)
                loss = self.criterion(pred, label)
                
                loss_protected_list = []
                loss_non_protected_list = []
                loss_negative_list = []
                for idx in range(len(batch)):
                    if sign[idx] == 0:
                        loss_protected_list.append(loss[idx])
                    elif sign[idx] == 1:
                        loss_non_protected_list.append(loss[idx])
                    else:
                        loss_negative_list.append(loss[idx])
                
                if len(loss_protected_list) == 0:
                    loss_protected = torch.tensor(0)
                else:
                    loss_protected = sum(loss_protected_list) / len(loss_protected_list)
                if len(loss_non_protected_list) == 0:
                    loss_non_protected = torch.tensor(0)
                else:
                    loss_non_protected = sum(loss_non_protected_list) / len(loss_non_protected_list)
                if len(loss_negative_list) == 0:
                    loss_negative = torch.tensor(0)
                else:
                    loss_negative = sum(loss_negative_list) / len(loss_negative_list)
                
                part_1 = factor * loss_protected
                part_2 = (1 - self.dataset.negative_rate - factor) * loss_non_protected
                part_3 = self.dataset.negative_rate * loss_negative
                loss_weight = part_1 + part_2 + part_3
                
                self.optimizer.zero_grad()
                loss_weight.backward()
                self.optimizer.step()
                
                epoch_loss_protected += loss_protected.item()
                epoch_loss_non_protected += loss_non_protected.item()
            
            epoch_loss_protected /= len(dataloader)
            epoch_loss_non_protected /= len(dataloader)
            epoch_loss_protected_list.append(epoch_loss_protected)
            epoch_loss_non_protected_list.append(epoch_loss_non_protected)
        
        diff = epoch_loss_protected_list[-1] - epoch_loss_non_protected_list[-1]
        return self.net.state_dict(), diff
        
    def FairFed_local_update(self,
                     num_epoch,
                     batch_size,
                     global_parameters,
                     ):
        
        # 加载模型
        self.net = Adult_MLP(101)
        self.net.load_state_dict(global_parameters, strict=True)
        self.net.to(self.dev)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), 0.01)
        
        # 本地训练
        loss_list = []
        pred_list = []
        dataloader = DataLoader(self.dataset, batch_size, True)
        for epoch in range(num_epoch):
            epoch_loss = 0
            epoch_pred = []
            for batch in dataloader:
                data, label = batch
                data = data.to(self.dev)
                label = label.to(self.dev)
                pred = self.net(data)
                loss = self.criterion(pred, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                epoch_pred.append(pred)
                
            epoch_loss /= len(dataloader)
            loss_list.append(epoch_loss)
            
            epoch_pred = torch.cat(epoch_pred, dim=0)
            pred_list.append(epoch_pred)
            
        res  = self.find_equal_opportunity_score_distributed(predictions=(pred_list[-1] > 0.5).float())
        
        # 上传模型
        return self.net.state_dict(), res
    
    def find_equal_opportunity_score_distributed(self, predictions):
        """
        EOD公平性指标，只关心在正类上的预测结果的差异
        """
        
        protected_attr = self.dataset[:][0][:,2]
        labels = self.dataset[:][1]
        # print(protected_attr.shape)
        # print(labels.shape)
        # print(predictions.shape)
        # print(protected_attr)
        # exit(0)
        
        tp_protected = 0
        tn_protected = 0
        fp_protected = 0
        fn_protected = 0
        
        tp_non_protected = 0
        tn_non_protected = 0
        fp_non_protected = 0
        fn_non_protected = 0
        
        saValue = 0
        for i in range(len(protected_attr)):
            if protected_attr[i] == saValue:
                if labels[i] == 1:
                    if predictions[i] == 1:
                        tp_protected += 1
                    else:
                        fn_protected += 1
                else:
                    if predictions[i] == 1:
                        fp_protected += 1
                    else:
                        tn_protected += 1
                    
            else:                
                if labels[i] == 1:
                    if predictions[i] == 1:
                        tp_non_protected += 1
                    else:
                        fn_non_protected += 1
                else:
                    if predictions[i] == 1:
                        fp_non_protected += 1
                    else:
                        tn_non_protected += 1
                    
        score_protected = tp_protected / len(protected_attr)
        score_non_protected = tp_non_protected / len(protected_attr)
        
        num_tp_fn_protected = tp_protected + fn_protected
        num_tp_fn_non_protected = tp_non_protected + fn_non_protected
        
        # local_EOD
        tmp_1 = tp_protected / (tp_protected + fn_protected)
        tmp_2 = tp_non_protected / (tp_non_protected + fn_non_protected)
        local_score = tmp_2 - tmp_1

        return score_protected, score_non_protected, num_tp_fn_protected, num_tp_fn_non_protected, local_score
            

class Client_Group(object):
    def __init__(self,
                 dev,
                 num_client,
                 dataset_name,
                 is_iid,
                 alpha,):
        self.dev = dev
        self.num_client = num_client
        self.clients = []
        self.scales = []
        self.dataset_name = dataset_name
        self.is_iid = is_iid
        self.alpha = alpha
        self.test_dataloader = None
        self.dataset_allocation()
    
    def load_adult(self):
        # 加载数据集 [32561, 14]
        column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                    'hours-per-week', 'native-country', 'income']
        url = '../data/adult.csv'
        data = pd.read_csv(url)
        data.drop('fnlwgt', axis=1, inplace=True)
        data.drop('capital.gain', axis=1, inplace=True)
        data.drop('capital.loss', axis=1, inplace=True)
     
        # 替换?并填充缺失值
        data.replace('?', pd.NA, inplace=True)
        data.fillna(data.mode().iloc[0], inplace=True)
        
        # 性别与收入转化成二进制形式
        data.replace('>50K', 1, inplace=True)
        data.replace('<=50K', 0, inplace=True)
        data.replace('Male', 1, inplace=True)
        data.replace('Female', 0, inplace=True)
        
        # 类别型特征进行独热编码
        data = pd.get_dummies(data)

        # 划分特征与标签
        label = data['income']
        data.drop(columns=['income'], axis=1, inplace=True)
        data = np.array(data).astype(float)
        
        # 数据分为训练集和测试集
        train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.2)

        # 转化为tensor张量
        train_data = torch.tensor(train_data, dtype=torch.float32)
        test_data = torch.tensor(test_data, dtype=torch.float32)
        train_label = torch.tensor(train_label.values, dtype=torch.float32).reshape(-1, 1)
        test_label = torch.tensor(test_label.values, dtype=torch.float32).reshape(-1, 1)

        return train_data, train_label, test_data, test_label, transforms.Compose([])
        
    def load_mnist(self):
        # 下载：[60000, 28, 28], tensor + tensor
        train_dataset = torchvision.datasets.MNIST(root='../data', download=True, train=True)
        test_dataset = torchvision.datasets.MNIST(root='../data', download=True, train=False)
        train_data = train_dataset.data
        test_data = test_dataset.data
        train_label = train_dataset.targets
        test_label = test_dataset.targets
        # 预处理：先划分到各客户端再预处理，因此分开
        trans = transforms.Compose([transforms.ToPILImage(),
                                    transforms.ToTensor(), # [0, 255] -> [0, 1]
                                    transforms.Normalize((0.13065973,),(0.3015038,))])  # [0, 1] -> [-1, 1]
        return train_data, train_label, test_data, test_label, trans
    
     
    def load_cifar10(self):
        # 下载：[50000, 32, 32, 3], tensor + list
        train_dataset = torchvision.datasets.CIFAR10(root='../../data', download=True, train=True)
        test_dataset = torchvision.datasets.CIFAR10(root='../../data', download=True, train=False)
        train_data = torch.tensor(train_dataset.data).permute(0, 3, 1, 2)
        train_label = torch.tensor(train_dataset.targets)
        test_data = torch.tensor(test_dataset.data).permute(0, 3, 1, 2)
        test_label = torch.tensor(test_dataset.targets)
        trans = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize(32),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2033, 0.1994, 0.2010))])
        return train_data, train_label, test_data, test_label, trans
    
    # 划分独立同分布数据
    def iid_split(self, train_label, num_client):
        all_idcs = [i for i in range(train_label.shape[0])]
        num_local_dataset = int(train_label.shape[0] / num_client)
        client_idcs = []
        for i in range(num_client):
            client_idcs.append(np.random.choice(all_idcs, num_local_dataset))
            all_idcs = list(set(all_idcs) - set(client_idcs[-1]))
        return client_idcs
    
    # 划分非独立同分布数据
    def dirichlet_split(self, train_label, alpha, num_client):
        train_label = np.array(train_label)
        num_class = int(train_label.max()) + 1
        # (K, N) class label distribution matrix X, record how much each client occupies in each class
        label_distribution = np.random.dirichlet([alpha] * num_client, num_class) 
        # Record the sample subscript corresponding to each K category
        class_idcs = [np.argwhere(train_label==y).flatten() for y in range(num_class)]
        # Record the index of N clients corresponding to the sample set respectively
        client_idcs = [[] for _ in range(num_client)]
        for c, fracs in zip(class_idcs, label_distribution):
            # np.split divides the samples of class k into N subsets according to the proportion
            # for i, idcs is to traverse the index of the sample set corresponding to the i-th client
            for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
                client_idcs[i] += [idcs]
        client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
        return client_idcs        
    
    # 下载、预处理、训练集划分和封装、测试集封装
    def dataset_allocation(self):
        if self.dataset_name == 'adult':
            train_data, train_label, test_data, test_label, trans = self.load_adult()
        elif self.dataset_name == 'mnist':
            train_data, train_label, test_data, test_label, trans = self.load_mnist()
        elif self.dataset_name == 'cifar10':
            train_data, train_label, test_data, test_label, trans = self.load_cifar10()
        else:
            raise NotImplementedError('{}'.format(self.dataset_name))

        if self.is_iid:
            client_idcs = self.iid_split(
                train_label=train_label,
                num_client=self.num_client
            )
        else:
            client_idcs = self.dirichlet_split(
                train_label=train_label,
                alpha=self.alpha,
                num_client=self.num_client
            )
        
        # # 刻画结果
        # plt.figure(figsize=(12, 8))
        # plt.hist([train_label[idc]for idc in client_idcs], stacked=True,
        #         bins=np.arange(min(train_label)-0.5, max(train_label) + 1.5, 1),
        #         label=["Client {}".format(i) for i in range(self.num_client)],
        #         rwidth=0.5)
        # plt.xticks(np.arange(10))
        # plt.xlabel("Label type")
        # plt.ylabel("Number of samples")
        # plt.legend(loc="upper right")
        # plt.title("Display Label Distribution on Different Clients")
        # plt.savefig('../../logs/data_distribution.jpg')
        
        for idc in client_idcs:
            local_data = train_data[idc]
            local_label = train_label[idc]
            local_dataset = Local_Dataset(local_data, local_label, trans)
            client = Client(local_dataset, self.dev)
            self.clients.append(client)
            self.scales.append(len(idc))

        self.test_dataset = Local_Dataset(test_data, test_label, trans)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=100, shuffle=False)
        
group = Client_Group(torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 5, 'adult', True, 0.1)