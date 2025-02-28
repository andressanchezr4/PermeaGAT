# -*- coding: utf-8 -*-
"""
Created on 2023

@author: andres.sanchez
"""

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.nn import Linear
from torch_geometric.nn import GATConv, global_mean_pool
import warnings
warnings.filterwarnings("ignore")

class GraphModel(torch.nn.Module):
    def __init__(self, hidden_channels=128, heads=5):
        super(GraphModel, self).__init__()
        self.heads = heads  
        self.conv1 = GATConv(43, hidden_channels, heads=heads, edge_dim=10)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, edge_dim=10)
        self.conv3 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, edge_dim=10)
        self.lin = Linear(hidden_channels * heads, 1)

    def forward(self, x, edge_index, batch, edge_attr):
        x = self.conv1(x, edge_index, edge_attr).relu()
        x = self.conv2(x, edge_index, edge_attr).relu()
        x = self.conv3(x, edge_index, edge_attr)
        x = global_mean_pool(x, batch)
        return self.lin(x).squeeze() 

class Trainer:
    def __init__(self, model, train_loader, criterion, optimizer, scheduler):
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.count = 1
    
    def train_epoch(self):
        self.model.train()
        all_accuracy, all_precision, all_recall, all_f1 = [], [], [], []
        
        for data in self.train_loader:
            out = self.model(data.x.float(), data.edge_index, data.batch, data.edge_attr).squeeze()
            real = data.y.float().squeeze()  
            
            loss = self.criterion(out, real)  
            loss.backward()
            
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            pred = (out > 0.5).long() 
            
            y_true = data.y.detach().numpy()
            y_pred = pred.detach().numpy()
            
            all_accuracy.append(accuracy_score(y_true, y_pred))
            all_precision.append(precision_score(y_true, y_pred))
            all_recall.append(recall_score(y_true, y_pred))
            all_f1.append(f1_score(y_true, y_pred))
        
        self.scheduler.step()
        print(f'--------------- Epoch: {self.count} ---------------')
        print(f'TRAIN -> Prec: {sum(all_precision)/len(all_precision):.4f}, Rec: {sum(all_recall)/len(all_recall):.4f}')
        print('----------------------------')
        self.count += 1     

class Evaluator:
    def __init__(self, model, test_loader, info):
        self.model = model
        self.test_loader = test_loader
        self.info = info
        
    def evaluate(self):
        self.model.eval()
        all_accuracy, all_precision, all_recall, all_f1 = [], [], [], []

        with torch.no_grad():
            for data in self.test_loader:
                out = self.model(data.x.float(), data.edge_index, data.batch, data.edge_attr).squeeze()
                pred = (out > 0.5).long()
                
                y_true = data.y.detach().numpy()
                y_pred = pred.detach().numpy()
                
                all_accuracy.append(accuracy_score(y_true, y_pred))
                all_precision.append(precision_score(y_true, y_pred))
                all_recall.append(recall_score(y_true, y_pred))
                all_f1.append(f1_score(y_true, y_pred))
        
        print(f'{self.info} | Prec: {sum(all_precision)/len(all_precision):.3f},  Rec: {sum(all_recall)/len(all_recall):.3f}')
        
            
