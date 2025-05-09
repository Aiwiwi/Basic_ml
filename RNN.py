import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import os
import sys

sys.path.append('../')

class Alphabet(object):
    def __init__(self):
        self.symbol_to_indx = {}
        self.idx_to_symbol = []
        self._len = 0
        
    def add_symbol(self,s):
        if s not in self.symbol_to_indx:
            self.symbol_to_indx[s] = self._len
            self.idx_to_symbol.append(s)
            self._len += 1
            
    def __len__(self):
        return self._len
    
class Texts(object):
    def __init__(self,path):
        self.dictionary = Alphabet()
        self.train_data = self.tokensize(os.path.join(path,'train.txt'))
        self.test_data = self.tokensize(os.path.join(path,'test.txt'))
        self.validation_data = self.tokensize(os.path.join(path,'valid.txt'))
        
    def tokensize(self, path):
        assert os.path.exists(path)
        
        tokens = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                tokens += len(line)
                for s in line:
                    self.dictionary.add_symbol(s)
                    
        with open(path, 'r', encoding='utf-8') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                for s in line:
                    ids[token] = self.dictionary.symbol_to_indx[s]
                    token += 1
            
        return ids


class TextLoader(object):
    def __init__(self, dataset, batch_size=128, sequence_length=30):
        self.data = dataset
        self.batch_size = batch_size
        self.seq_len = sequence_length 
        self._batchify()
        
    def _batchify(self):          
        self.nbatch = self.data.size(0) // self.batch_size
        data = self.data.narrow(0, 0, self.nbatch * self.batch_size)
        self.batch_data = data.view(self.batch_size, -1).t().contiguous()
        
    def _get_batch(self, i):
        seq_len = min(self.seq_len, len(self.batch_data)-1-i)
        data = self.batch_data[i:i+seq_len]
        target = self.batch_data[i+1:i+1+seq_len].view(-1)
        return data, target
    
    def __iter__(self):
        for i in range(0, self.batch_data.size(0)-1, self.seq_len):
            data, targets = self._get_batch(i)
            yield data, targets
            
    def __len__(self):
        return self.batch_data.size(0)
    
    
    

class RNN(nn.Module):
    def __init__(self,n_inputs,n_token, rnn_type: str, n_hidden, n_layers, dropout=0.5):
        super(RNN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.Embedding(n_token, n_inputs)
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(n_inputs, n_hidden, n_layers)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(n_inputs, n_hidden, n_layers)
        else:
            print('ERERERERERrr')
        
        self.decoder = nn.Linear(n_hidden, n_token)
        
        self.init_weights()
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.rnn_type = rnn_type
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.n_inputs = n_inputs
    
    
    def init_weights(self):
        initrage = 0.1
        self.encoder.weight.data.uniform_(-initrage,initrage)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrage,initrage)
        
        
    def forward(self, x, hidden=None):
        emb = self.dropout(self.encoder(x))
        output, hidden= self.rnn(emb, hidden)
        output = self.dropout(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden        
        
        
    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type=='LSTM':
            return(weight.new(self.n_layers, bsz, self.n_hidden).zero_(),
                   weight.new(self.n_layers, bsz, self.n_hidden).zero_())
        elif self.rnn_type=='GRU':
            return weight.new(self.n_layers, bsz, self.n_hidden).zero_()
            
    
    def train_iteration(self,train_loader,corpus, lr=1e-4):
        self.train()
        total_loss = 0
        ntokens=len(corpus.dictionary)
        for batch,(data, targets) in enumerate(train_loader):
            self.zero_grad()
            output, hidden = self.forward(data)
            loss = self.criterion(output.view(-1, ntokens),targets)
            loss.backward()
            
            for p in self.parameters():
                p.data.add_(p.grad.data, alpha=-lr)
                
            total_loss += loss
            if (batch+1)%100==0:
                print(f'batches - [{batch+1}] -- Mean Loss - {total_loss/100}')
                total_loss = 0
            
    def evaluate(self, data_loader):
        self.eval()
        total_loss = 0
        ntokens=len(corpus.dictionary)
        hidden = self.init_hidden(batch_size)
        for batch, (x, y) in enumerate(data_loader):
            output, hidden = self.forward(x)
            output_flat = output.view(-1,ntokens)
            total_loss += len(x)+self.criterion(output_flat, y).item()
        return total_loss/len(data_loader) 
    
    def generate(self, n=50, temp=.75):
        self.eval()
        ntokens = len(corpus.dictionary)
        x = torch.rand(1,1).mul(ntokens).long()
        hidden = None
        out = []
        for i in range(n):
            output, hidden = self.forward(x,hidden)
            s_weights = output.squeeze().data.div(temp).exp()
            s_idx = torch.multinomial(s_weights, 1)[0]
            x.data.fill_(s_idx)
            s = corpus.dictionary.idx_to_symbol[s_idx]
            out.append(s)
        return ''.join(out)
    
    
    def train_model(self, epochs,train_data_loader,validation_data_loader,corpus,lr=4.,generation_length=30):
        best_val_loss = float('INF')
        for epoch in range(epochs):
            print('='*90)
            
            self.train_iteration(train_data_loader, corpus,lr)
            
            print('='*90)
            print('-'*90)
            
            if (epoch+1)%50 == 0:
                generated1=self.generate(n=10000, temp=1.)
                with open(f'./RNN_test/1_epoch{epoch+1}.txt', 'w', encoding='utf-8') as f:
                    f.write(generated1)
                
            
            with torch.no_grad():
                print(f'sample: //{self.generate(generation_length)}//')
            val_loss = self.evaluate(validation_data_loader)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            else:
                if(lr > 1e-3):
                    lr /= 4.
            
            temp = ''
            with open(results, 'r', encoding='utf-8') as rf:
                for line in rf:
                    for s in line:
                        temp += s
            
            with open(results, 'w', encoding='utf-8') as rf:
                rf.write(f'{temp}\n\nMODEL HYPERPARAMETERS:\n\tNumber of layers:{self.n_layers}\n\tNumber of hidden neurons:{self.n_hidden}\n\tNumber of input neurons:{self.n_inputs}\n\tBatch size:{train_data_loader.batch_size}\n\tRNN type:{self.rnn_type}\nVALIDATION LOSS:{val_loss}\n{'='*100}\nEPOCH:{epoch+1}\nGENERATED:{self.generate(1000)}\n\n')
            
            print(f'EPOCH {epoch+1} finished. VALIDATION LOSS: {val_loss}')
            print('-'*90)
            
            
            print('&'*90)
            

            
corpus = Texts('D:/BSU-ML/ru_dataset/')
results = 'D:/BSU-ML/RNN_test/results.txt'

    
batch_size=128
sequence_length=30
grad_clip = 0.1
lr = 4.

            
def main():
    train_loader = TextLoader(corpus.train_data,batch_size,sequence_length)
    test_loader = TextLoader(corpus.test_data,batch_size,sequence_length)
    validation_loader = TextLoader(corpus.validation_data, batch_size,sequence_length)
    
    ntokens = len(corpus.dictionary)
    rnn_model = RNN(n_inputs=128,n_token=ntokens,rnn_type='GRU',n_hidden=128,n_layers=4,dropout=0.4)
    
    rnn_model.train_model(200, train_loader, validation_loader,corpus,lr,generation_length=50)
    generated = rnn_model.generate(n=100000,temp=1.)
    with open('./RNN_test/Final.txt','w',encoding='utf-8') as f:
        f.write(generated)
    
    torch.save(rnn_model.state_dict(), 'bel_rnn.pth')

main()  