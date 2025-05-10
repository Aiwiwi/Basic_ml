import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

class generator(nn.Module):
    def __init__(self):
        pass
    
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.fc1 = nn.Linear(28*28,128)
        self.fc2 = nn.Linear(128,10)
    def forward(self,x):
        x = x.view(-1,28*28)
        x = torch.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x
    
    
class Encoder(nn.Module):
    def __init__(self, latent_size):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(28*28,latent_size)
    
    def forward(self, x):
        x = torch.relu(self.fc(x))
        return x
    
class Decoder(nn.Module):
    def __init__(self, latent_size):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_size, 28*28)
    
    def forward(self,x):
        x = self.fc(x)
        x = x.view(-1,28,28)
        return x
        
class aae(nn.Module):
    def __init__(self, latent_size, lr=1e-4):
        super(aae, self).__init__()
        self.encoder = Encoder(latent_size)
        self.decoder = Decoder(latent_size)
        self.disc = discriminator()
        
        self.optim = torch.optim.Adam(self.parameters(), lr)
        
        
    def forward(self, x):
        x = x.view(-1,28*28)
        x = self.encoder(x)
        decoded = self.decoder(x)
        
        return decoded.view(28,28)
        
    def loss(self, t,y, out, x):
        return F.mse_loss(t,y)+F.mse_loss(out,x)
    
    def plot_mnist(self, images, reconstructed_images):
        n = len(images)
        fig, axes = plt.subplots(2,n, figsize = (n,2))
        
        for i, img in enumerate(images):
            if n==1:
                ax = axes[0]
            else:
                ax = axes[0,i]
            img = img.squeeze()
            ax.imshow(img.detach().numpy(), cmap='gray')    
            ax.axis('off')
            
        for i, img in enumerate(reconstructed_images):
            if n == 1:
                ax = axes[1]
            else:
                ax = axes[1,i]
            img = img.squeeze()
            ax.imshow(img.detach().numpy(), cmap='gray')    
            ax.axis('off')
            
            
        plt.tight_layout()
        plt.show()
        n = len(images)
        fig, axes = plt.subplots(2,n, figsize = (n,2))

        for i, img in enumerate(images):
            if n==1:
                ax = axes[0]
            else:
                ax = axes[0,i]
            ax.imshow(img.squeeze(), cmap='gray')    
            ax.axis('off')

        for i, img in enumerate(reconstructed_images):
            if n == 1:
                ax = axes[1]
            else:
                ax = axes[1,i]
            ax.imshow(img.squeeze(), cmap='gray')    
            ax.axis('off')

        plt.tight_layout()
        plt.show()
    
    def train(self, epoch, x_train, y_train):
        x_train = x_train.unsqueeze(1).float()
        y_train = y_train.unsqueeze(1).float()
        total_loss = 0
        images = []
        reconstructed_images = []
        for e in range(epoch):
            i = 0
            for x, y in zip(x_train, y_train):
                x = x.unsqueeze(0)
                y = y.unsqueeze(0)
                
                self.zero_grad()
                self.optim.zero_grad()
                out = self.forward(x)
                disc = self.disc(out)
                loss = self.loss(disc,y, out, x)
                loss.backward()
                self.optim.step()
                
                i+= 1
                total_loss += loss.item()
                if i%10000 == 0:
                    print(f'Epoch {e+1} iteration {i} mean loss {total_loss/10000}')
                    total_loss = 0
            k = np.random.randint(0,len(x_train))
            images.extend([x_train[k], x_train[k+1]])
            reconstructed_images.extend([self.forward(x_train[k]), self.forward(x_train[k+1])])
            
        self.plot_mnist(images, reconstructed_images) 
            
train_dataset = datasets.MNIST(root='D:/BSU-ML/MNIST', train=True)
x_train = train_dataset.data
y_train = train_dataset.targets

coder = aae(10,1e-2)
coder.train(10,x_train,y_train)                 