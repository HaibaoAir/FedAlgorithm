import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, Subset # 两件套
from torchvision import transforms
import matplotlib.pyplot as plt

class Local_Dataset(Dataset):
    def __init__(self,
                 local_data,
                 local_label,
                 trans):
        self.local_data = local_data
        self.local_label = local_label
        self.trans = trans
        
    def __getitem__(self, index):
        data = self.trans(self.local_data[index])
        label = self.local_label[index]
        return data, label
    
    def __len__(self):
        return self.local_label.shape[0]


class Client(object):
    def __init__(self, 
                 dataset,
                 dev):
        self.dataset = dataset
        self.dev = dev
        
    def local_update(self, 
                     num_epoch, 
                     batch_size, 
                     net,
                     criterion,
                     optim,
                     global_parameters,
                     idcs):
        
        if idcs == 1:
            self.dataset = Subset(self.dataset, range(6000))
        elif idcs == 2:
            self.dataset = Subset(self.dataset, range(600))
        # 加载模型
        net.load_state_dict(global_parameters, strict=True)
        
        # 本地训练
        loss_list = []
        print('length:', len(self.dataset))
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(num_epoch):
            for batch in dataloader:
                data, label = batch
                data = data.to(self.dev)
                label = label.to(self.dev)
                
                pred = net(data)
                loss = criterion(pred, label)
                optim.zero_grad()
                loss.backward()
                optim.step()
                loss_list.append(loss.cpu().detach().numpy())
                break
        
        print('loss:', sum(loss_list) / len(loss_list))
        # 上传模型
        return net.state_dict()
            

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
    
    
    def load_mnist(self):
        # 下载：[60000, 28, 28], tensor + tensor
        train_dataset = torchvision.datasets.MNIST(root='../../data', download=True, train=True)
        test_dataset = torchvision.datasets.MNIST(root='../../data', download=True, train=False)
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
        num_class = train_label.max() + 1
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
        if self.dataset_name == 'mnist':
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
        
        # 刻画结果
        plt.figure(figsize=(12, 8))
        plt.hist([train_label[idc]for idc in client_idcs], stacked=True,
                bins=np.arange(min(train_label)-0.5, max(train_label) + 1.5, 1),
                label=["Client {}".format(i) for i in range(self.num_client)],
                rwidth=0.5)
        plt.xticks(np.arange(10))
        plt.xlabel("Label type")
        plt.ylabel("Number of samples")
        plt.legend(loc="upper right")
        plt.title("Display Label Distribution on Different Clients")
        plt.savefig('../../logs/data_distribution.jpg')
        
        for idc in client_idcs:
            local_data = train_data[idc]
            local_label = train_label[idc]
            local_dataset = Local_Dataset(local_data, local_label, trans)
            client = Client(local_dataset, self.dev)
            self.clients.append(client)
            self.scales.append(len(idc))
        
        self.test_dataset = Local_Dataset(test_data, test_label, trans)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=100, shuffle=False)