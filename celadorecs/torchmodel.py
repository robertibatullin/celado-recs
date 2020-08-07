#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 16:20:49 2020

@author: robert
"""

import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, n_features):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, 4)
        self.fc5 = nn.Linear(4, 2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = self.fc3(x)
        x = F.leaky_relu(x)
        x = self.fc4(x)
        x = F.leaky_relu(x)
        x = self.fc5(x)
        x = torch.softmax(x, 1)
        return x
    
    def fit(self, x, y, n_iterations = 1000):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), 
                              lr=0.001, momentum=0.01)
        for epoch in range(1, n_iterations+1):
            running_loss = 0.0
            inputs = torch.FloatTensor(x.values)
            labels = torch.LongTensor(y)
            optimizer.zero_grad()
            outputs = self(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print('Iteration: {0}\tLoss: {1:.4f}'.format(
                epoch, 
                running_loss,))
        return self
            
    def predict_proba(self, x):
        inputs = torch.FloatTensor(x.values)
        return self(inputs).detach().numpy()
