# ProInterVal: Validation of Protein-Protein Interfaces through Learned Interface Representations

## Overview
ProInterVal is an approach for validating protein-protein interfaces using learned interface representations. The approach involves using a graph-based contrastive autoencoder architecture and a transformer to learn representations of protein-protein interaction interfaces from unlabeled data, then validating them through learned representations with a graph neural network.

![framework](https://github.com/ku-cosbi/ProInterVal/assets/26218685/ab90466a-c805-439f-a47c-339c8fb63093)

## Installation
```
git clone https://github.com/ku-cosbi/ProInterVal
cd ProInterVal
```
### Install virtualenv via pip
```
pip install virtualenv
virtualenv venv
source venv/bin/activate
```
### Install dependencies
```
pip install -r requirements.txt
```

