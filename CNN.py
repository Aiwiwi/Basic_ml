import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms


class Encoder(nn.Module):
    def __init__(self,latent_size):
        super(Encoder,self).__init__()
        self.fc1 = nn.Linear(1568, 128)
        self.fc2 = nn.Linear(128,latent_size)
        
    def forward(self,x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

class Decoder(nn.Module):
    def __init__(self, latent_size):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_size,128)
        self.fc2 = nn.Linear(128,28*28)
        
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x.view(-1,28,28)
        


class Net(nn.Module):
    def __init__(self,bn=False, dropout=False, latent_size=16, rho=0.05, lr=1e-4, l2=0.):
        super(Net, self).__init__()
        self.latent_size = latent_size
        self.rho = rho
        self.drop = dropout
        self.batchnorm = bn
        
        self.conv1 = nn.Conv2d(1,16,3, padding=1)
        self.conv2 = nn.Conv2d(16,32,3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.encoder = Encoder(latent_size)
        self.decoder = Decoder(latent_size)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=l2)
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x) 
        
        x = x.view(-1,1568)
        
        x = self.encoder(x)
        self.data_rho = x.mean(0)
        out = self.decoder(x)
        return out
    
    def rho_loss(self):
        eps = 1e-10

        data_rho = self.data_rho.clamp(eps, 1 - eps)
        rho = torch.tensor(self.rho).clamp(eps, 1 - eps)

        kl = self.rho * (torch.log(rho) - torch.log(data_rho)) + \
             (1 - self.rho) * (torch.log(1 - rho) - torch.log(1 - data_rho))
        self._rho_loss = kl.mean()

    
    def loss(self,out,target):
        return F.mse_loss(out,target)
    
    
    def train(self,epochs ,x_train, y_train, x_test, y_test):
            
        x_train = x_train.unsqueeze(1).float() / 255.0 
        y_train = x_train.clone()
        for epoch in range(epochs):
            total_loss = 0
            temp_loss=0
            
            test_samples = x_test[:5].unsqueeze(1).float() / 255.0  
            for i,(x, y) in enumerate(zip(x_train.float(), y_train)):
                x = x.unsqueeze(0)
                y = y.unsqueeze(0)
                
                self.optimizer.zero_grad()
                out = self.forward(x)
                
                self.rho_loss()
                reconstruction_loss = self.loss(out, y)
                delta = 0.1
                loss = reconstruction_loss + self._rho_loss*delta
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                temp_loss += loss.item()

                if i%10000 == 0:
                    print(f'Epoch {epoch+1} iteration - [{i}/60000] - mean loss = {temp_loss/10000}')
                    temp_loss=0
                    print("Reconstruction loss:", reconstruction_loss.item())
                    print("KL loss:", self._rho_loss.item())
                    
            print(f'Epoch finished, mean loss - {total_loss/len(x_train)}')
        self.test(test_samples)
            
    def test(self, test_samples):
        num_imgs = test_samples.size(0)
        fig, axes = plt.subplots(2, num_imgs, figsize=(2*num_imgs, 4))

        if num_imgs == 1:
            axes = axes.reshape(2, 1)  

        with torch.no_grad():
            reconstructed = self.forward(test_samples)

            for i in range(num_imgs):
                original_img = test_samples[i].squeeze().numpy()
                reconstructed_img = reconstructed[i].squeeze().numpy()

                axes[0, i].imshow(original_img, cmap='gray')
                axes[0, i].axis('off')

                axes[1, i].imshow(reconstructed_img, cmap='gray')
                axes[1, i].axis('off')

        plt.tight_layout()
        plt.show()
    
train_dataset = datasets.MNIST(root='D:/BSU-ML/MNIST', train=True, download=False)
x_train = train_dataset.data
y_train = train_dataset.targets

test_dataset = datasets.MNIST(root='D:/BSU-ML/MNIST', train=False, download=False)
x_test = test_dataset.data
y_test = test_dataset.targets


net = Net()
net.train(25, x_train,y_train,x_test, y_test)