## Introduction
An inofficial PyTorch implementation of [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907).

## Datasets
+ citeseer
+ cora
+ pubmed
+ NELL

## GCN

## Train
```
# usage: python train.py --dataset DATASET
python train.py --dataset citeseer

or

python train.py --dataset cora

or 

python train.py --dataset pubmed
```

## Evaluate
```
# usage: python evaluate.py --dataset DATASET --checkpoint CHECKPOINT
python evaluate.py --checkpoint checkpoint/citeseer/gcn_80.pth
```

## Experiment results
+ Accuracy 
    
    |          | Citeseer(%) | Core(%) | Pubmed(%) | NELL(%) |
    | :------: | :------: | :------: | :------: | :---: |
    | this repo|             |          |          |
    |  paper   |     70.3    |  81.5   |  79.0     |  66.0 |


+ Accuracy curve
+ Loss
+ Regularization
+ Dropout
+ Nums of train samples
+ Others