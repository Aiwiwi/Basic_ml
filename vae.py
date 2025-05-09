import torch
from torchvision import datasets 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = datasets.MNIST(root='D:/BSU-ML/MNIST', train=True)
x_train = train_dataset.data
y_train = train_dataset.targets


class Encoder(nn.Module):
    def __init__(self, latent_size, dropout = .1):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(28*28*32, 128)
        self.fc2 = nn.Linear(128, latent_size)
        
        self.dropout = nn.Dropout(dropout)
        self.to(device)
        
    def forward(self, x):
        x = self.dropout(self.fc1(x))
        x = torch.tanh(x)
        x = self.dropout(self.fc2(x))
        x = torch.tanh(x)
        return x        
        



class Decoder(nn.Module):
    def __init__(self, latent_size, dropout):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_size, 128)
        self.fc2 = nn.Linear(128, 28*28)
        
        self.dropout = nn.Dropout(dropout)
        self.to(device)
        
    def forward(self, x):
        x = self.dropout(self.fc1(x))
        x = torch.tanh(x)
        x = self.dropout(self.fc2(x))
        x = torch.tanh(x)
        return x.view(-1,28,28)
    
    
class VAE(nn.Module):
    def __init__(self, latent_size, dropout, lr=1e-4,l2=0, rho=0.05):
        super(VAE, self).__init__()
        self.conv1 = nn.Conv2d(1,16,2,1,padding='same')
        self.conv2 = nn.Conv2d(16,32,2,1,padding='same')
        self.pool = nn.MaxPool2d(3,1,1)
        self.encoder = Encoder(latent_size,dropout)
        self.z_mean = nn.Linear(latent_size,latent_size)
        self.z_logvar = nn.Linear(latent_size,latent_size)
        self.decoder = Decoder(latent_size,dropout)
        
        
        self.optim = torch.optim.Adadelta(self.parameters(), lr)
        self.to(device)
        self.rho = rho

            
    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.pool(x)
        
        x = x.view(-1,28*28*32)
        encoded = torch.tanh(self.encoder(x))

        mu = self.z_mean(encoded)
        logvar = self.z_logvar(encoded)
        z = self.reparametrize(mu, logvar)
        
        decoded = self.decoder(z)
        return decoded, mu, logvar
    
    
    def train(self, epoch, x_train):
        x_train = x_train.unsqueeze(1).float() / 255.0
        x_train = x_train.to(device)
        images = []
        reconstructed_images = []
        for e in range(epoch):
            total_loss = 0
            for i,x in enumerate(x_train):
                x = x.unsqueeze(0)
                self.zero_grad()
                out,mu,logvar = self.forward(x)
                
                loss = self.loss_function(out, x, mu, logvar)
                
                loss.backward()
                self.optim.step()
                
                total_loss += loss.item()
                if i%10000 == 0:
                    print(f'Epoch {e+1}, iteration:[{i}] mean loss:{total_loss/10000}')
                    total_loss = 0
            k = np.random.randint(0,len(x_train))
            images.extend([x_train[k], x_train[k+1]])
            reconstructed_images.extend([self.forward(x_train[k], self.forward[k+1])])
        self.plot_mnist(images, reconstructed_images) 
            
                 
        
    
        
    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.mse_loss(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD
    
    def plot_mnist(self, images, reconstructed_images):
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
            
        plt.tight_layout()
        plt.show()
    

                
vae = VAE(10, 0.1)
vae.train(1,x_train)