#!/usr/bin/env python
# coding: utf-8

# # Spectrograms NN 

# First import the necessary packages 

# In[51]:


import numpy as np
import torch
import sys
import time

import torch.optim as optim
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
from torchvision import transforms


cuda = torch.cuda.is_available()
cuda


# ### Dataloader

# Will be getting two dataset, a train and labels.

# In[52]:


# class MyDataset(data.Dataset):
#     def __init__(self, X, Y, k):
#         self.X = X
#         self.Y = Y
#         self.Xcat = np.load(self.X, allow_pickle = True)
#         self.Ycat = np.concatenate(np.load(self.Y, allow_pickle = True), axis = 0)
        
#         self.k = k
#         self.length = np.array([0.0])

#         if self.k > 0:
#             for i in range(len(self.Xcat)):
#                 self.length = np.append(self.length, self.length[-1] + len(self.Xcat[i]))
#                 zeros = np.zeros((self.k, 40)).astype(float)
#                 self.Xcat[i] = np.append(zeros, self.Xcat[i], axis = 0)  
#                 self.Xcat[i] = np.append(self.Xcat[i], zeros, axis = 0)
                
#         self.Xcat = np.concatenate(self.Xcat, axis = 0)
#         self.length = np.sort(self.length)
        
#     def __len__(self):
#         return len(self.Ycat)

#     def __getitem__(self, index):
#         # need to index this properly.  Hmmm 
#         # every time we hit value in array, increase index x by 12
#         ind = np.argwhere(self.length <= index)
#         maxind = np.argmax(ind)
#         indexx = index + self.k*(maxind + 1)
        
#         X = self.Xcat[slice(indexx - self.k, indexx + self.k + 1)].reshape(-1).astype(float)
#         Y = self.Ycat[index]
#         return X, Y
    
    
    
class MyDataset(data.Dataset):
    def __init__(self, X, Y, k):
        self.X = X
        self.Y = Y
        self.Xcat = np.concatenate(np.load(self.X, allow_pickle = True), axis = 0)
        self.Ycat = np.concatenate(np.load(self.Y, allow_pickle = True), axis = 0)

        self.k = k
        if self.k > 0:
            zeros = np.zeros((self.k, 40)).astype(float)
            self.Xcat = np.append(zeros, self.Xcat, axis = 0)  
            self.Xcat = np.append(self.Xcat, zeros, axis = 0)   
    
    def __len__(self):
        return len(self.Ycat)

    def __getitem__(self, index):
        # turn Y to long to pass into cross entropy?
        
        X = self.Xcat[slice(index, index+2*self.k+1)].reshape(-1).astype(float)
        Y = self.Ycat[index]
        return X, Y


# In[53]:


num_workers = 4 if cuda else 0 
batch_size = 512 if cuda else 64
k = 25

# Training
train_dataset = MyDataset('train.npy', 'train_labels.npy', k)

train_params = dict(shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True) if cuda                    else dict(shuffle=True, batch_size=batch_size)

train_load = data.DataLoader(train_dataset, **train_params)

# Testing
test_dataset = MyDataset('dev.npy', 'dev_labels.npy', k)

test_params = dict(shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True) if cuda                    else dict(shuffle=False, batch_size=batch_size)

test_load = data.DataLoader(test_dataset, **test_params)


# In[54]:


# for (x,y) in train_load:
#     print(x.shape, y.shape)
    


# ### Define a NN Class

# In[55]:


class simpleMLP(nn.Module):
    def __init__(self, linear_layers, num_batch_norm, num_dropout):
        super(simpleMLP, self).__init__()

        layers = []
        self.linear_layers = linear_layers
        self.num_batch_norm = num_batch_norm
        self.bn = num_batch_norm > 0
        self.dn = num_dropout > 0

        # create the layers
        for i in range(len(linear_layers) - 2):
            layers.append(nn.Linear(linear_layers[i], linear_layers[i+1]))
            # add drop out conditional statement
            if self.dn and i >= num_dropout: 
                layers.append(nn.Dropout(p = 0.5))
            if self.bn and i <= num_batch_norm:
                layers.append(nn.BatchNorm1d(linear_layers[i+1]))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(linear_layers[-2], linear_layers[-1]))

        # combine final net and assign
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ### Create the model and define our loss criterion

# May want to add optimal learning rate schedule
# 

# In[61]:



input_size = 40*(2*k + 1)
model = simpleMLP([input_size, 2048, 2048, 1024, 1024, 1024, 512, 256, 138], 5, 6)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 2, threshold = 0.005)
device = torch.device("cuda" if cuda else "cpu")
model.to(device)
print(model, criterion)


# ### Training function

# Basically we do a foward pass, then compute the loss, backward pass, take a step and reset.

# In[57]:


def train_epoch(model, train_load, criterion, optimizer):
    # indicate that we are training 
    model.train()
    running_loss = 0.0
    
    # start timer and start iterating
    start_train = time.time()
    for batch_idx, (data, target) in enumerate(train_load):
        data = data.to(device)
        target = target.to(device) # all data & model on same device
        
        # forward, then backward, then step
        outputs = model(data.float())
        loss = criterion(outputs, target.long())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()   # .backward() accumulates gradients

        # accumulate loss
        running_loss += loss.item()
    
    # end timer and take average loss
    end_train = time.time()
    running_loss /= len(train_load)
    
    return end_train, start_train, running_loss


# ### Test function

# In[58]:


def test_model(model, test_load, criterion, testout, batch_size):
    with torch.no_grad():
        # indicate that we are evaluating model
        model.eval()
        
        # initialize errors to 0
        running_loss = 0.0
        total_predictions = 0.0
        correct_predictions = 0.0    
        
        # start iterating
        for batch_idx, (data, target) in enumerate(test_load):   
            data = data.to(device)
            target = target.to(device)

            # run forward pass and then compute loss
            outputs = model(data.float())
            loss = criterion(outputs, target.long()).detach()
            
            # get predictions 
            _, predicted = torch.max(outputs.data, 1)
            
            # calculate correct predictions / loss
            total_predictions += target.size(0)            
            correct_predictions += (predicted == target).sum().item()
            running_loss += loss.item()
            
        # calculate average loss and accuract
        running_loss /= len(test_load)
        acc = (correct_predictions/total_predictions)*100.0
        return running_loss, acc


# ### Initialize and run our model

# In[60]:


n_epochs = 10

Train_loss = []
Test_loss = []
Test_acc = []

testout = open("submission.csv", "w")
model.load_state_dict(torch.load('Adam Weight Decay: 7.pt'))

for i in range(n_epochs):
    # Training and outputting loss
    end_train, start_train, train_loss = train_epoch(model, train_load, criterion, optimizer)
    print('Training Loss: ', train_loss, 'Time: ',end_train - start_train, 's')
    
    # Test and ouputting
    test_loss, test_acc = test_model(model, test_load, criterion, testout, batch_size)
    print('Testing Loss: ', test_loss)
    print('Testing Accuracy: ', test_acc, '%')    
    
    scheduler.step(test_loss)
    Train_loss.append(train_loss)
    Test_loss.append(test_loss)
    Test_acc.append(test_acc)
    
    torch.save(model.state_dict(), 'Adam Weight Decay: ' + str(i+10) + '.pt')
    
    print('='*20)


# In[ ]:





# In[ ]:




