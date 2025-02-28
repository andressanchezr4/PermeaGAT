# -*- coding: utf-8 -*-
"""
Created on 2023

@author: andres.sanchez
"""

from Tools import DataHandler
from Model import GraphModel, Trainer, Evaluator
import torch

data_path = "C:/Users/andres.sanchez/Desktop/graph_model/toproduction/data/gutper_set2.csv"
n_epochs = 50

# Data preparation
data_handler = DataHandler(data_path)
train_loader, ext_loader, ext_loader_FL, ext_loader_noFL = data_handler.prepare_data()
print('Data Prepared!')

# Model generation
model = GraphModel()
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.1)

trainer = Trainer(model, train_loader, criterion, optimizer, scheduler)

evaluator_all = Evaluator(model, ext_loader, 'ALL CLASES')
evaluator_FL = Evaluator(model, ext_loader_FL, 'FL')
evaluator_noFL = Evaluator(model, ext_loader_noFL, 'noFL')

print('Let\'s train!')
for epoch in range(1, n_epochs):
    # Model training
    trainer.train_epoch()
    
    # Model evaluation
    evaluator_all.evaluate()
    evaluator_noFL.evaluate()
    evaluator_FL.evaluate()
    