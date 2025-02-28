# PermeaGAT
PermeaGAT is a Graph Attention Network to predict the gut permeability of a metabolite. 

### Script Description
* run.py --> End to end analysis.
* Model --> Graph Model.
* Tools.py --> Class for data preparation.
* Smiles2Graphs.py --> Functions for graph generation from SMILES. 

### Requirements
* Numpy
* Pandas
* Rdkit
* Scikit-learn
* Torch
* Torch_geometric

### Disclaimer
The dataset was mainly obtained from the Human Metabolome Database (HMDB). Since the molecules are clearly distinguished into two groups (Fatty Lipids vs. non-Fatty Lipids), validation is performed separately for each group to properly assess the effectiveness of the training.
