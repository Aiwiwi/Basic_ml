import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

def get_target(a):
    res = []
    for i in range(10):
        if i == a.item():
            res.append(1)
            continue
        res.append(0)
    return torch.tensor(res)

class model(nn.Module):
    def __init__(self, batch_normalization:bool, dropout:bool, lr=1e-4, l2=0., n_classes=10):
        super(model,self).__init__()
        if batch_normalization:
            self.bn = nn.BatchNorm1d(128)
        self.batchnorm = batch_normalization
        self.dropout = dropout
        
        self.input=nn.Linear(28*28, 128)
        self.fc1=nn.Linear(128,128)
        self.fc2=nn.Linear(128,n_classes)
        
        self.optim = torch.optim.SGD(self.parameters(),lr=lr, weight_decay=l2)
        
    def forward(self, x):
        x = x.view(-1,28*28)
        x = torch.sigmoid(self.input(x))
        if self.batchnorm:
            x = self.bn(x)
        x = torch.sigmoid(self.fc1(x))
        if self.dropout:
            x = F.dropout(x, 0.5)
        x = torch.relu(self.fc2(x))
        return x
    
    def loss(self, output, target):
        return F.cross_entropy(output,target)
         
    
    def train(self, x_train, y_train, x_test, y_test, epochs):
        losses = []
        total_loss = 0
        for i in range(epochs):
            j = 0
            for x, y in zip(x_train.float(), y_train):
                j += 1
                if j % 10000 ==0:
                    losses.append(total_loss/10000)
                    print(f'Loss: {total_loss/10000} iteration: {j}/60000. Epoch {i+1}')                    
                    total_loss = 0
                self.optim.zero_grad()
                output = self.forward(x)
                loss = self.loss(output,y.unsqueeze(0))
                loss.backward()
                self.optim.step()
                total_loss += loss.item()
                
        iters = range(1,len(losses)+1)
        fig =plt.figure()
        plt.plot(iters, losses, 'bo-', label='Потеря на обучающей выборке')
        plt.title('Потеря')
        plt.xlabel('Эпохи')
        plt.ylabel('Потеря')               
        plt.show()
    
    
            
            
train_dataset = datasets.FashionMNIST(root='D:/fashionmnist', train=True, download=False)
x_train = train_dataset.data
y_train = train_dataset.targets

test_dataset = datasets.FashionMNIST(root='D:/fashionmnist', train=False, download=False)
x_test = test_dataset.data
y_test = test_dataset.targets



Model = model(False,False)
Model.train(x_train=x_train, y_train=y_train,x_test=x_test, y_test=y_test, epochs=5)