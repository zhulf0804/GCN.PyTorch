## Introduction
An inofficial PyTorch implementation of [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907).

## Datasets
+ citeseer
+ cora
+ pubmed(TBD)
+ NELL(TBD)

## GCN
![](http://latex.codecogs.com/gif.latex?F=\sigma_2[D^{-\frac{1}{2}}AD^{-\frac{1}{2}}\sigma_1(D^{-\frac{1}{2}}AD^{-\frac{1}{2}}XW_1)W_2])

A is the adjacency matrix(adding **self-loops**), D is the degree matrix, X is the features, W is the parameters.
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
python evaluate.py --dataset citeseer --checkpoint checkpoints/citeseer/gcn_200.pth
```

## Experiment results
+ Accuracy 
    
    |          | Citeseer(%) | Cora(%) | Pubmed(%) | NELL(%) |
    | :------: | :------: | :------: | :------: | :---: |
    | this repo|     70.8    |  82.5   |          |
    |  paper   |     70.3    |  81.5   |  79.0     |  66.0 |


+ Accuracy and Loss curve
    + citeseer(see experiments/citeseer*.png)
        
        ```
        tensorboard --logdir=experiments/citeseer
        ```
        
    + cora(see experiments/cora*.png)
    
        ```
        tensorboard --logdir=experiments/cora
        ```


        
+ Dropout and Regularization (**results for training 200 epoches**)

    | weight_decay | dropout | train ac(%) | val ac(%) | test ac(%) |
    | :---: | :---: | :---: | :---: | :---:|
    | 5e-3  | 0.7 | 90.8 | 68.4 | 70.1 |
    | 5e-3  | 0.5 | 91.7 | 70.4 | 71.5 |
    | 5e-3  | 0.2 | 93.3 | 71.0 | **71.8** |
    | 5e-3  | 0.0 | 94.2 | 68.8 | 69.6 |
    | 5e-4  | 0.7 | 97.5 | 70.0 | 70.8 |
    | **5e-4** | **0.5** | 98.3 | 70.2 | 70.8 |
    | 5e-4  | 0.2 | 98.3 | 70.8 | 70.7 |
    | 5e-4  | 0.0 | 98.3 | 71.0 | 71.7 |
    | 5e-5  | 0.7 |100.0 | 67.8 | 68.3 |
    | 5e-5  | 0.5 |100.0 | 70.6 | 69.1 |
    | 5e-5  | 0.2 |100.0 | 68.2 | 69.5 |
    | 5e-5  | 0.0 |100.0 | 67.4 | 68.0 |
    | 0.0   | 0.7 |100.0 | 68.0 | 68.0 |
    | 0.0   | 0.5 |100.0 | 68.0 | 67.6 |
    | 0.0   | 0.2 |100.0 | 63.8 | 59.7 |
    | 0.0   | 0.0 |100.0 | 64.8 | 63.6 |
    
    **Note:** Bold parameters represents the parameters in the paper.

+ All GCNLayers with regularization

    |   | gcn1 | gcn2  | accuracy(%) |
    | :---: | :---: | :---: | :---: | 
    | experiment| reg  | reg   | 71.9 |
    | **paper** | reg  | no-reg | 70.8 |

+ Nums of train samples

    | # train samples  | accuracy(%) |
    | :---: | :---: | 
    | 120 | 70.8 | 
    | 240 | 70.6 | 
    | 360 | 71.5 |