import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torchvision import datasets

class Model(nn.Module):
    def __init__(self, batch_normalization: bool, dropout: bool, lr=1e-4, l2=0., n_classes=10):
        super(Model, self).__init__()
        self.batchnorm = batch_normalization
        self.dropout = dropout
        
        self.input = nn.Linear(28 * 28, 256)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_classes)
        
        if self.batchnorm:
            self.bn1 = nn.BatchNorm1d(256)
            self.bn2 = nn.BatchNorm1d(128)

        self.optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=l2)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.tanh(self.input(x))
        if self.batchnorm:
            x = self.bn1(x)
        x = torch.tanh(self.fc1(x))
        if self.batchnorm:
            x = self.bn2(x)
        x = torch.relu(self.fc2(x))
        if self.dropout:
            x = F.dropout(x, 0.5)
        x = self.fc3(x)
        return x

    def loss(self, output, target):
        return F.cross_entropy(output, target)

    def train_model(self, x_train, y_train, epochs):
        for i in range(epochs):
            for x, y in zip(x_train.float(), y_train):
                self.optim.zero_grad()
                output = self.forward(x.unsqueeze(0))  # Добавляем размерность для батча
                loss = self.loss(output, y.unsqueeze(0))  # Добавляем размерность для батча
                loss.backward()
                self.optim.step()

            k = np.random.randint(0,len(x_train))
            randx = x_train[k].float()  
            randy = y_train[k]
            out = self.forward(randx.unsqueeze(0))  
            _loss = self.loss(out, randy.unsqueeze(0))  

            print(f'Loss: {_loss.item()}    epoch: {i + 1}')

    def test(self, val_x, val_y):
        correct = 0
        total = len(val_x)

        with torch.no_grad():
            for x, y in zip(val_x.float(), val_y):
                output = self.forward(x.unsqueeze(0))
                _, predicted = torch.max(output.data, 1)
                correct += (predicted == y).sum().item()

        print(f'Accuracy: {100 * correct / total:.2f}%')

# Загрузка данных
train_dataset = datasets.FashionMNIST(root='D:/fashionmnist', train=True, download=False)
x_train = train_dataset.data
y_train = train_dataset.targets

test_dataset = datasets.FashionMNIST(root='D:/fashionmnist', train=False, download=False)
x_test = test_dataset.data
y_test = test_dataset.targets

# Инициализация и обучение модели
model_instance = Model(batch_normalization=False, dropout=False)
model_instance.train_model(x_train=x_train, y_train=y_train, epochs=10)