import math
import random 
import numpy as np
import torch
from copy import deepcopy, copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sko.PSO import PSO

a = 1
print(a)
b = 1
for i in range(10):
    b += 1
    print(b)
exit(0)

# a = [1,2,3]
# b = [4,5,6]
# c = [2,2,2]
# print(a * b / c)
# exit(0)

a = torch.tensor([1,2,3])
b = torch.tensor([4,5,6])
print(a * b)
exit(0)

a = torch.tensor([[1,2,3],[4,5,6]])
b = torch.tensor([[4,5,6],[7,8,9]])
print(b / a)
c = torch.tensor([0.5, 0.5, 0.5])
print(c * (b / a))
exit(0)


c = [a,b]
print(sum(c) / len(c))
exit()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(10, 1)
        
    def forward(self, x):
        x = self.fc(x)
        return x

seed = 10
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

net1 = Net()
criterion1 = F.mse_loss
optimizer1 = optim.SGD(net1.parameters(), lr=1e-2)
input1 = torch.randn(1, 10)
target1 = torch.randn(1, 1)

loss_list1 = []
for epoch in range(4):
    output = net1(input1)
    loss = criterion1(output, target1)
    optimizer1.zero_grad()
    loss.backward()
    optimizer1.step()
    loss_list1.append(loss.item())
    
print(loss_list1)
weigh1 = deepcopy(net1.state_dict())
print('trace 1: id = {}, value = {}'.format(id(weigh1), weigh1['fc.bias']))

#####################################################################
net2 = Net()
criterion2 = F.mse_loss
optimizer2 = optim.SGD(net2.parameters(), lr=1e-2)
input2 = torch.randn(1, 10)
target2 = torch.randn(1, 1)

loss_list2 = []
for epoch in range(4):
    output = net2(input2)
    loss = criterion2(output, target2)
    optimizer2.zero_grad()
    loss.backward()
    optimizer2.step()
    loss_list2.append(loss.item())

print(loss_list2)
weigh2 = net2.state_dict()
print('trace 2: id = {}, value = {}'.format(id(weigh2), weigh2['fc.bias']))

##################################
# 最新实验，只要不
net1.load_state_dict(weigh1)
criterion1 = F.mse_loss
optimizer1 = optim.SGD(net1.parameters(), lr=1e-2)
input1 = torch.randn(1, 10)
target1 = torch.randn(1, 1)

loss_list1 = []
for epoch in range(4):
    output = net1(input1)
    loss = criterion1(output, target1)
    optimizer1.zero_grad()
    loss.backward()
    optimizer1.step()
    loss_list1.append(loss.item())
    
print(loss_list1)
weigh1_1 = net1.state_dict()
print('trace 1_1: id = {}, value = {}'.format(id(weigh1_1), weigh1_1['fc.bias']))
print('trace 1: id = {}, value = {}'.format(id(weigh1), weigh1['fc.bias']))
exit(0)

#####################################################################

weigh = weigh1
for key in weigh.keys():
    weigh[key] += (weigh2[key] + weigh3[key])
    weigh[key] /= 3
print('trace 0: id = {}, value = {}'.format(id(weigh), weigh['fc.bias']))

#######################################################################

net1.load_state_dict(weigh)
criterion1 = F.mse_loss
optimizer1 = optim.SGD(net1.parameters(), lr=1e-2)
input1 = torch.randn(1, 10)
target1 = torch.randn(1, 1)

loss_list1 = []
for epoch in range(4):
    output = net1(input1)
    loss = criterion1(output, target1)
    optimizer1.zero_grad()
    loss.backward()
    optimizer1.step()
    loss_list1.append(loss.item())
    
print(loss_list1)
weigh1 = net1.state_dict()
print('trace 1: id = {}, value = {}'.format(id(weigh1), weigh1['fc.bias']))

exit(0)


a = {'a':1, 'b':2, 'c':3}
b = a
print(b)
a['c'] = 6
print(b)

c = {}
for key in a.keys():
    c[key] = a[key]
a['c'] = 8
print(c)

d = {'a':1, 'b':2, 'c':3}
print(id(d))
d = copy(a)
print(id(d))
print(id(a))
exit()

a = 3
b = a
print(b)
a = 5
print(b)
exit()



a = 3
b = 3
assert a == b
exit()

a = torch.tensor([1,2,3])
b = torch.tensor([3,2,1])
print(torch.square(a))
print(torch.sum(torch.square(a)))
exit(0)
print(abs(a-b))
exit(0)

eps = 1e-3

def func(x):
    return math.cos(x)

def pretrain():
    
    x = 2.1
    for i in range(100):
        x_new = func(x)
        print(x_new)
        if abs(x_new - x) < eps:
            return
        else:
            x = x_new

pretrain()
exit(0)


a = None
print(type(a))
print(a == None)
exit(0)

a = None
print(len(a))
exit(0)

a = [1,2,3,4]
b = random.sample(a, 3)
print(b)
for item in b:
    a.remove(item)
print(a)

exit(0)

a = [1,2,3,4]
b = [1,2]
print(a-b)
exit(0)

print(x for x in range(10))
print(sum(x for x in range(10)))
exit(0)

def func_1(x):
    return (x[0] + x[1]) ** 2

def func_2(x):
    return (x[0] + x[1]) ** 3

def demo_func(x):
    return func_1(x) + func_2(x)


pso = PSO(func=demo_func,
          dim=2,
          pop=40,
          max_iter=150,
          lb=[-3, -3],
          ub=[3, 3],
          w=0.8,
          c1=0.5,
          c2=0.5)
pso.run()
print(pso.gbest_x)
print(pso.gbest_y)
exit(0)


# class Batch(Dataset):
#     def __init__(self, dataset, indices):
#         self.dataset = dataset
#         self.indices = indices
        
#     def __getitem__(self, idx):
#         return self.dataset[self.indices[idx]]
    
#     def __len__(self):
#         return len(self.indices)
 
# def normal_distribution(x, mean, sigma):
#     return np.exp(-1*((x-mean)**2)/(2*(sigma**2)))/(math.sqrt(2*np.pi) * sigma)
 

# mean2, sigma2 = 0, 2
# x2 = np.linspace(mean2 - 6*sigma2, mean2 + 6*sigma2, 100)
 
 
# y2 = normal_distribution(x2, mean2, sigma2)
 
# plt.plot(x2, y2, 'g', label='m=0,sig=2')
# plt.show()


# x = [1, 2, 3]
# print(random.sample(x, 2))

# x = np.arange(0.1, 10, 0.01)
# y = [(-1) * item * math.log(item) for item in x]
# plt.plot(x, y)
# plt.show()
# a = [[1,2,3],[4,5,6]]
# print([a[i][0] for i in range(2)])
