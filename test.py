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
import numpy as np
from matplotlib import pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

a = [1,2,3]
b = a[:2]
c = a[-1:]
print(b)
print(c)
print(b + c)
exit(0)

x = np.arange(0, 5, 0.1)
y = x + (1/x)
print(x)
print(y)
plt.plot(x, y)
plt.savefig('test.png')
exit(0)

# 示例函数
def example_function(input_scalar):
    x = np.linspace(-5, 5, 10)
    y = np.linspace(-5, 5, 10)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X**2 + Y**2) * input_scalar
    return X, Y, Z

# 输入标量
input_scalar = 3
X, Y, Z = example_function(input_scalar)

# 绘制3D表面图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis')
fig.colorbar(surf, label='Value')  # 添加颜色条
ax.set_title(f"3D Surface Plot for Input {input_scalar}")
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Value')
plt.savefig('test.png')
exit()

reward_list = []
theta_list = []
# 0.25, 0.4, 0.41, 0.415, 0.42, 
sigma_list = [0.475, 0.75, 1, 1.25]
poison_list = [0.07, 0.25, 0.5, 0.8]
path = 'logs/fedstream/pre_estimate_5_2'
for i in sigma_list:
    result = np.load(path + '_{}.npy'.format(i))
    print(result[0])
    reward_list.append(result[0][0])
    theta_list.append(result[0][1])
    
plt.plot(theta_list, reward_list, marker='o', markerfacecolor='white', color='k', linestyle='--')
plt.xlabel('Conservation Rate $\\theta$', fontproperties = 'Times New Roman', size = 14)
plt.ylabel('Reward $R$', fontproperties = 'Times New Roman', size = 14)
plt.xlim(0, 1)
plt.ylim(30, 100)
for poison, theta, reward in zip(poison_list, theta_list, reward_list):
    plt.text(theta, reward, '$\sigma={}$ \n $\\theta={:.2f}, R={:.2f}$'.format(poison, theta, reward), fontproperties = 'Times New Roman', size = 8)

plt.grid(True)
plt.savefig('test.png', dpi=200)
exit(0)

a = [random.randint(500, 1000) for _ in range(30)]
print(a)

b = [random.uniform(0, 200) for _ in range(30)]
print(b)
exit(0)

seed = 10
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

a = np.random.randint(10)
print(a)
b = np.random.randint(10)
print(b)
exit(0)

x = np.arange(1,10,1)
y = np.arange(1,10,1)

for k in range(0, 1, 0.2):
    print(k)
    
exit(0)

fig = plt.figure()
ax = fig.add_subplot(2,2,1,projection='3d')
a = np.arange(-1, 1, 0.01)
b = np.arange(-1, 1, 0.01)
c, d = np.meshgrid(a, b)

e_matrix = []
for i in range(len(a)):
    e_list = []
    for j in range(len(b)):
        e_list.append(math.cos(c[i][j]) + math.sin(d[i][j]))
    e_matrix.append(e_list)
e = np.array(e_matrix)    
ax.plot_surface(c,d,e, rstride=1, cstride=1, cmap='rainbow')
plt.savefig('test.png')

exit(0)

b = np.array([[1,2,3],[4,5,6],[7,8,9]])

a = np.array([1,2,3])
print(b * (1 / a).reshape(3,1))
exit(0)

a = 4
print(np.array([a]))
exit(0)

print(type(np.random.randint(1,4)))
exit(0)
a = np.ndarray(5)
l = []
l.append(a)
l.append(a)
print(type(l[1]))
exit(0)



a = 2.25e-06 - 1.98e-06
print(2.25e-06)
print(a>0)
exit(0)

a = np.array([1,2,3])
b = np.array([2,4,6])
c = np.array([[1,1],[1,1],[1,1]])
print(np.sum(c * (a / b).reshape(3,1), axis=0))
exit(0)

a = np.array([[1,2,3],[4,5,6]])
print(np.sum(a, axis=0))
print(a[:, 2])
exit(0)

exit(0)
c = torch.tensor([1,2,3])
print(c.shape)
d = torch.tensor([4,5,6])
print(c / d)
a = torch.tensor([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print(a * (c / d).reshape(3,1))
exit(0)
b = torch.tensor([1,2]).reshape(2,1)
print(a * b)
exit(0)

a = 1
print(a)
b = 1
for i in range(10):
    b += 1
    print(b)
exit(0)

# a = [1,2,3]
# b = [4,5,6
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

