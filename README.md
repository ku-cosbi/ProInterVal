# ProInterVal: Validation of Protein-Protein Interfaces through Learned Interface Representations

## Overview
ProInterVal is an approach for validating protein-protein interfaces using learned interface representations. The approach involves using a graph-based contrastive autoencoder architecture and a transformer to learn representations of protein-protein interaction interfaces from unlabeled data, then validating them through learned representations with a graph neural network.

![framework](https://github.com/ku-cosbi/ProInterVal/assets/26218685/ab90466a-c805-439f-a47c-339c8fb63093)

## Installation
```
git clone https://github.com/ku-cosbi/ProInterVal
cd ProInterVal
```
### Install conda and create virtual environment
[Install conda from this link](https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html)

```
conda create -n ProInterVal
conda activate ProInterVal
```
### Install dependencies
```
pip install -r requirements.txt
```
### Exit virtual environment
Whenever you want to exit, run the following command:
```
conda deactivate
```
## Usage
You can train and test our representation learning and interface validation models using run.py script.
If you want to make predictions on our final representation learning model using your dataset, please run the following command:
```
python run.py --path=[path/to/your/dataset/directory] --mode=test --model=RL
```

If you want to obtain interface validation results, please use the following command:
```
python run.py --path=[path/to/your/dataset/directory] --mode=test --model=GNN
```

If you want to train representation learning model with your dataset run the following command:
```
python run.py --path=[path/to/your/training/dataset/directory] --mode=train --model=RL
```
After running above command, your trained model will be saved to your local directory. Now, you can make predictions using the following command:
```
import torch
import data_prep
from model import model as rep_model

test_data = data_prep.prepare_data(path/to/your/dataset/directory)
model = RepresentationLearningModel()
model.load_state_dict(torch.load(local/path/to/your/trained/model))

for i, t in enumerate(test_data):
    prediction = rep_model.get_interface_representation(model, t[1], t[2])
```
If you want to train interface validation model using your dataset, please run the following command: 
```
python run.py --path=[path/to/your/training/dataset/directory] --mode=train --model=GNN
```
After running above command, your trained model will be saved to your local directory. Now, you can make predictions using the following command:
```
import torch
import data_prep
from model import gnn
test_data = data_prep.prepare_data(path/to/your/dataset/directory)
model = gnn.GNNModel(num_features=len(test_data[0]), hidden_size=512, num_classes=2)
for i, t in enumerate(test_data):
    X, A = t[1], t[2]
    prediction = gnn.predict(model, X, A)
```
