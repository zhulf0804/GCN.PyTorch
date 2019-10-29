import argparse
import torch
from datasets import load_data, process_features
from models.gcn import GCN
from models.utils import build_optimizer

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='citeseer', help='Dataset to train')
parser.add_argument('--init_lr', type=float, default=0.036, help='Initial learing rate')
parser.add_argument('--epoches', type=int, default=200, help='Number of traing epoches')
parser.add_argument('--hidden_dims', type=list, default=[16], help='Dimensions of hidden layers')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep  probability)')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight for l2 loss on embedding matrix')
args = parser.parse_args()



features, labels, adj = load_data(args.dataset)
features = process_features(features)

model = GCN(features.shape[1], args.hidden_dims, labels.shape[1])
y = model(adj, features)

optimizer = build_optimizer(model.parameters, args.init_lr)


