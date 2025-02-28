# -*- coding: utf-8 -*-
"""
Created on 2023

@author: andres
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit.Chem import Descriptors
import torch
from torch_geometric.data import Data
from sklearn import preprocessing

compound_classes = ['Benzenoids',
 'Endocannabinoids',
 'Fatty Acyls',
 'Glycerolipids',
 'Glycerophospholipids',
 'Hydrocarbons',
 'Nucleosides, nucleotides, and analogues',
 'Organic acids and derivatives',
 'Organic nitrogen compounds',
 'Organic oxygen compounds',
 'Organoheterocyclic compounds',
 'Organosulfur compounds',
 'Other',
 'Phenylpropanoids and polyketides',
 'Prenol lipids',
 'Sphingolipids',
 'Steroids and steroid derivatives']

ion_classes = ['Acid', 'Basic', 'Neutral', 'Zwitterion']

def one_hot_encode(vector, categories):
    vector = np.array(vector).reshape(-1, 1)

    encoder = preprocessing.OneHotEncoder(categories=[categories], sparse_output=False, handle_unknown='ignore')
    encoder.fit(vector)
 
    return encoder.transform(vector)

def calculate_properties(mol):
    tpsa = Descriptors.TPSA(mol)
    logp = Descriptors.MolLogP(mol)
    rb = Descriptors.NumRotatableBonds(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    mw = Descriptors.MolWt(mol)
    qed = Descriptors.qed(mol)
    nring = Chem.rdMolDescriptors.CalcNumRings(mol)
    naring = Chem.rdMolDescriptors.CalcNumAromaticRings(mol)
    fsp3 = Chem.rdMolDescriptors.CalcFractionCSP3(mol)

    return [tpsa, logp, rb, hbd, hba, mw, qed, nring, naring, fsp3]
    

def one_hot_encoding(x, permitted_list):
    if x not in permitted_list:
        x = permitted_list[-1]

    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]

    return binary_encoding


def get_atom_features(atom, 
                      use_chirality = True, 
                      hydrogens_implicit = True):

    # permitted_list_of_atoms =  ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As','Al','I', 'B','V','K','Tl','Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn', 'Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr','Cr','Pt','Hg','Pb','Unknown']
    permitted_list_of_atoms = ['C', 'N', 'O', 'S', 'Cl', 'Br', 'F']
    
    if hydrogens_implicit == False:
        permitted_list_of_atoms = ['H'] + permitted_list_of_atoms
    
    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)
    n_heavy_neighbors_enc = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])
    formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
    hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])
    is_in_a_ring_enc = [int(atom.IsInRing())]
    is_aromatic_enc = [int(atom.GetIsAromatic())]
    atomic_mass_scaled = [float((atom.GetMass() - 10.812)/116.092)]
    vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5)/0.6)]
    covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64)/0.76)]
    atom_feature_vector = atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc + hybridisation_type_enc + is_in_a_ring_enc + is_aromatic_enc + atomic_mass_scaled + vdw_radius_scaled + covalent_radius_scaled 
                                    
    if use_chirality == True:
        chirality_type_enc = one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
        atom_feature_vector += chirality_type_enc
    
    if hydrogens_implicit == True:
        n_hydrogens_enc = one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
        atom_feature_vector += n_hydrogens_enc

    return np.array(atom_feature_vector)


def get_bond_features(bond, 
                      use_stereochemistry = True):

    permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)
    bond_is_conj_enc = [int(bond.GetIsConjugated())]
    bond_is_in_ring_enc = [int(bond.IsInRing())]
    bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc
    
    if use_stereochemistry == True:
        stereo_type_enc = one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
        bond_feature_vector += stereo_type_enc

    return np.array(bond_feature_vector)


def mols2graphs_and_properties(mols, ys, ccl, icl):

    comp_classes_ohc = one_hot_encode(ccl, compound_classes)
    ion_classes_ohc = one_hot_encode(icl, ion_classes)
    data_list = []
    
    all_properties = []
    for (mol, y, c, ion) in zip(mols, ys, comp_classes_ohc, ion_classes_ohc):
        n_nodes = mol.GetNumAtoms()
        n_edges = 2*mol.GetNumBonds()
        unrelated_smiles = "O=O"
        unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
        n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
        n_edge_features = len(get_bond_features(unrelated_mol.GetBondBetweenAtoms(0,1)))

        X = np.zeros((n_nodes, n_node_features))

        for atom in mol.GetAtoms():
            X[atom.GetIdx(), :] = get_atom_features(atom)
            
        X = torch.tensor(X)
        
        (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
        E = torch.stack([torch_rows, torch_cols], dim = 0)
        
        EF = np.zeros((n_edges, n_edge_features))
        
        for (k, (i,j)) in enumerate(zip(rows, cols)):
            
            EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i),int(j)))
        
        EF = torch.tensor(EF, dtype = torch.float)
        
        y_tensor = torch.tensor(np.array([y]), dtype = torch.float)
        
        other_atributes = np.array(calculate_properties(mol))
        other_atributes = np.concatenate([other_atributes, c, ion])
        other_properties = torch.tensor(other_atributes, dtype = torch.float)
        all_properties.append(other_properties)
        
        data_list.append(Data(x = X, edge_index = E, edge_attr = EF, y = y_tensor))
    
    return data_list, all_properties
    
# Original code: https://www.blopig.com/blog/2022/02/how-to-turn-a-smiles-string-into-a-molecular-graph-for-pytorch-geometric/