import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Softmax(nn.Module):
    def __init__(self):
        super(Softmax, self).__init__()
        self.fc = nn.Linear(784, 10)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.softmax(self.fc(x), dim=1)
        return x
    
net = Softmax()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

train_data = torchvision.datasets.MNIST(
    root='../data',
    download=True,
    train=True,
    transform=torchvision.transforms.ToTensor())

test_data = torchvision.datasets.MNIST(
    root='../data',
    download=True,
    train=False,
    transform=torchvision.transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(
    dataset=train_data,
    batch_size=64,
    shuffle=True)

test_loader = torch.utils.data.DataLoader(
    dataset=test_data,
    batch_size=64,
    shuffle=False)

for epoch in range(50):
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
    if (epoch + 1) % 2 == 0:
        acc = 0
        tot = 0
        for i, (images, labels) in enumerate(test_loader):
            preds = net(images)
            preds = preds.argmax(dim=1)
            acc += (preds == labels).sum().item()
            tot += labels.size(0)
        acc /= tot
        print('Epoch {:d}, Loss: {:<5.4f}, Accuracy: {:<5.4f}'.format(epoch + 1, loss.item(), acc))