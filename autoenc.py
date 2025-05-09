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

class net(nn.Module):
    def __init__(self, batchnorm:bool=False, lr=1e-4, dropout:bool=False, l2 = 0.):
        super(net, self).__init__()
        self.fc1 = nn.Linear(28*28,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,64)
        self.fc4 = nn.Linear(64,10)
        
        self.bn = batchnorm
        if(batchnorm):
            self.batchnorm = nn.BatchNorm1d(256)
        self.dropout = dropout
        
        self.optim = torch.optim.SGD(self.parameters(),lr=lr, weight_decay=l2)
    
    def loss(self, t, y):
        return F.cross_entropy(t, y)
    
    def forward(self, x):
        x = x.view(-1,28*28)
        x = torch.tanh(self.fc1(x))
        if(self.bn):
            x = torch.tanh(self.batchnorm(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        if(self.dropout):
            x = torch.tanh(F.dropout(x,0.2))
        x = torch.tanh(self.fc4(x))
        return x
            
            
    def predict(self,x):
        out = self.forward(x)
        res = np.argmax(out.detach().numpy())
        return res
    
    def test(self,x_test,y_test):
        accuracy = 0 
        loss = 0
        n = len(x_test)
        for x,y in zip(x_test.float(), y_test):
            out = self.forward(x.unsqueeze(0))
            predict = np.argmax(out.detach().numpy())
            
            if predict == y:
                accuracy += 1
                
            loss += self.loss(out, y.unsqueeze(0))
        loss /= n
        accuracy /= n
        print(f'Accuracy - {accuracy*100}%\tMean loss - {loss}')
    
    def train(self, x_train, y_train,x_test, y_test, epochs):
        for ep in range(epochs):
            i = 0
            for x,y in zip(x_train.float(), y_train):
                self.optim.zero_grad()
                out = self.forward(x.unsqueeze(0))
                loss =self.loss(out, y.unsqueeze(0))
                loss.backward()
                self.optim.step()
            self.test(x_test, y_test)

class Encoder(nn.Module):
    def __init__(self,latent_size):
        super(Encoder,self).__init__()
        self.fc1 = nn.Linear(28*28, latent_size)
    
    def forward(self,x):
        return torch.sigmoid(self.fc1(x))
    
class Decoder(nn.Module):
    def __init__(self,latent_size):
        super(Decoder,self).__init__()
        self.fc1 = nn.Linear(latent_size,28*28)
    
    def forward(self,x):
        x = torch.sigmoid(self.fc1(x))
        return x.view(-1,28,28)
    
    
class autoencoder(nn.Module):
    def __init__(self,latent_size, loss_fn=F.mse_loss,lr=1e-4,l2=0.):
        super(autoencoder,self).__init__()
        self.encoder = Encoder(latent_size)
        self.decoder = Decoder(latent_size)
        self.loss_fn = loss_fn
        self._rho_loss = None
        self._loss = None
        self.latent_size = latent_size
        self.optim = torch.optim.Adam(self.parameters(),lr=lr,weight_decay=l2)
    
    def forward(self,x):
        x = x.view(-1,28*28)
        h = self.encoder.forward(x)
        self.data_rho = h.mean(0)
        out = self.decoder.forward(h)
        return out
    
    
    def train(self,epochs,x_train, y_train,x_test,y_test, rho=0.05):
        x_train = x_train.unsqueeze(1).float() / 255.0 
        y_train = x_train.clone()
        
        test_images = []
        for epoch in range(epochs):
            total_loss = 0.
            
            i = 0
            for x, y in zip(x_train.float(), y_train):
                self.zero_grad()
                
                x = x.unsqueeze(0)
                y = y.unsqueeze(0)
                
                output = self.forward(x)
                rho_loss = self.rho_loss(rho)
                
                reconstruction_loss = self.loss_fn(output, y)
                loss = reconstruction_loss + rho_loss
                
                loss.backward()
                self.optim.step()
                total_loss += loss.item()
                
                i+= 1
                if i % 10000 == 0:
                    print(f'Epoch {epoch+1} iteration - [{i}/60000] - mean loss = {total_loss/10000}')
                    total_loss = 0
            
            with torch.no_grad():
                idx = np.random.randint(0, len(x_test))
                test_img = x_test[idx].unsqueeze(0)
                if epoch % 2 == 0:
                    test_images.append(half_image(test_img))
                else:
                    test_images.append(test_img)
        self.test(test_images)
                
                
            
                
    def decode(self, h):
        with torch.no_grad():
            return self.decoder.forward(h)
    
    def loss(self, out, y, rho_loss):
        a = self.loss_fn(out,y)
        return a+rho_loss
    
    def rho_loss(self, rho, size_average=True):
        """
        D_KL(P||Q) = sum(p*log(p/q)) = -sum(p*log(q/p)) = -p*log(q/p) - (1-p)log((1-q)/(1-p))
        """
        dkl = - torch.log(self.data_rho/rho) * rho - torch.log((1-self.data_rho)/(1-rho)) * (1-rho)
        if size_average:
            self._rho_loss = dkl.mean()
        else:
            self._rho_loss = dkl.sum()
        return self._rho_loss
    
    def test(self, imgs):
        num_imgs = len(imgs)
 
        fig, axes = plt.subplots(2, num_imgs, figsize=(2*num_imgs, 4))

        with torch.no_grad():
            for i, img in enumerate(imgs):
                
                img_tensor = torch.tensor(img, dtype=torch.float) / 255.0
                img_tensor = img_tensor.view(-1, 28*28)

                
                reconstructed = self.forward(img_tensor)

               
                original_img = img_tensor.view(28, 28).numpy()
                reconstructed_img = reconstructed.view(28, 28).numpy()

                
                axes[0, i].imshow(original_img, cmap='gray')
                axes[0, i].set_title(f'Original {i+1}')
                axes[0, i].axis('off')

                axes[1, i].imshow(reconstructed_img, cmap='gray')
                axes[1, i].set_title(f'Reconstructed {i+1}')
                axes[1, i].axis('off')

        plt.tight_layout()
        plt.show()
            
        
def plot_mnist(images, shape):
    fig = plt.figure()
    for i in range(len(images)):
        ax = fig.add_subplot(shape[0],shape[1],i+1)
        ax.matshow(images[i])
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    

def half_image(image):
    with torch.no_grad():
        image = image.view(-1,28*28)
        image[:, 392:] = 0
        image = image.view(-1,28,28)
    return image

def normal_noise(image, noise_intensity=80, max_noise=255.):
    with torch.no_grad():
        original_shape=image.shape
        image=image.view(-1,28*28).float()
        
        batch_size = original_shape[0]
        idxs = torch.randint(0,28*28,(batch_size,noise_intensity))
        
        for i in range(batch_size):
            image[i, idxs[i]] = torch.rand(noise_intensity)*max_noise
            
        image = image.view(original_shape)
    return image

train_dataset = datasets.MNIST(root='D:/BSU-ML/MNIST', train=True, download=False)
x_train = train_dataset.data
y_train = train_dataset.targets

test_dataset = datasets.MNIST(root='D:/BSU-ML/MNIST', train=False, download=False)
x_test = test_dataset.data
y_test = test_dataset.targets
  
ae = autoencoder(32)
ae.train(20,x_train,y_train, x_test,y_test)

