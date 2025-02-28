# -*- coding: utf-8 -*-
"""
Created on 2023

@author: andres.sanchez
"""

import random
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
import torch
from torch_geometric.data import Batch
from rdkit import Chem
import Smiles2Graphs as mg
torch.manual_seed(42)
random.seed(42)

class DataHandler(object):
    def __init__(self, path, k = 8, overrepresented_classes = ["Glycerolipids",
                                                        "Glycerophospholipids",
                                                        "Fatty Acyls",
                                                        "Sphingolipids"]):
        self.path = path
        self.overrepresented_classes = overrepresented_classes 
        
        self.df = self.load_data()
        self.df_ext, self.df_cv, self.df_ext_FL, self.df_ext_noFL = self.split_data(k)
    
    def custom_collate(self, data_list):
        return Batch.from_data_list(data_list)

    def load_data(self):
        df = pd.read_csv(self.path, sep=';').dropna()
        df["mol"] = df.inchi.apply(lambda x: Chem.MolFromInchi(x))
        return df
    
    def split_data(self, k):
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        
        train_idx, test_idx = next(skf.split(self.df.mol, self.df.ccl))
        
        df_ext = self.df.iloc[test_idx].reset_index(drop=True)
        df_cv = self.df.iloc[train_idx].reset_index(drop=True)
        
        df_ext_FL = df_ext[df_ext.ccl.isin(self.overrepresented_classes)].reset_index(drop=True)  
        df_ext_noFL = df_ext[~df_ext.ccl.isin(self.overrepresented_classes)].reset_index(drop=True)

        return df_ext, df_cv, df_ext_FL, df_ext_noFL
    
    def prepare_data(self):
        X, features = mg.mols2graphs_and_properties(self.df_cv.mol, self.df_cv.gutper, self.df_cv.ccl, self.df_cv.icl)
        X_FL, features_FL = mg.mols2graphs_and_properties(self.df_ext_FL.mol, self.df_ext_FL.gutper, self.df_ext_FL.ccl, self.df_ext_FL.icl)
        X_noFL, features_noFL = mg.mols2graphs_and_properties(self.df_ext_noFL.mol, self.df_ext_noFL.gutper, self.df_ext_noFL.ccl, self.df_ext_noFL.icl)
        
        ext_X = X_FL + X_noFL
        ext_features = features_FL + features_noFL 
        
        train_loader = DataLoader(X, batch_size=128, shuffle=True, collate_fn=self.custom_collate)
        ext_loader = DataLoader(ext_X, batch_size=128, shuffle=True, collate_fn=self.custom_collate)
        ext_loader_FL = DataLoader(X_FL, batch_size=128, shuffle=True, collate_fn=self.custom_collate)
        ext_loader_noFL = DataLoader(X_noFL, batch_size=128, shuffle=True, collate_fn=self.custom_collate)
        
        features_train = torch.stack(features, dim=0)
        features_train = torch.split(features_train, 128)

        ext_features = torch.stack(ext_features, dim=0)
        ext_features = torch.split(ext_features, 128)
        
        features_FL = torch.stack(features_FL, dim=0)
        features_FL = torch.split(features_FL, 128)

        features_noFL = torch.stack(features_noFL, dim=0)
        features_noFL = torch.split(features_noFL, 128)
        
        return train_loader, ext_loader, ext_loader_FL, ext_loader_noFL
