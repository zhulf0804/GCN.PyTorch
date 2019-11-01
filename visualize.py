import argparse
import torch
import numpy as np
from datasets import load_data
from models.gcn import GCN
from misc.tsne_vis import tsne_vis


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='citeseer', help='Dataset to train')
parser.add_argument('--hidden_dim', type=list, default=16, help='Dimensions of hidden layers')
parser.add_argument('--checkpoint', type=str, default='', help='Directory to save checkpoints')
args = parser.parse_args()


classnames = {
    'citeseer': ['Agents', 'AI', 'DB', 'IR', 'ML', 'HCI'],
    'cora': ['Case_Based', 'Genetic_Algorithms', 'Neural_Networks', 'Probabilistc_Methods', 'Reinforcement_Learning', 'Rule_Learning', 'Theory'],
    'pubmed': ['Diabetes_Mellitus_Experimental', 'Diabetes_Mellitus_Type_1', 'Diabetes_Mellitus_Type_2'],
}


def hook_fn_forward(module, input, output):
    gcn_layer1_output.append(output.detach().numpy())


def visualize():
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(args.dataset)
    model = GCN(features.shape[1], args.hidden_dim, y_train.shape[1], 0)
    gcn_layer1 = model.gcn_layer1
    gcn_layer1.register_forward_hook(hook_fn_forward)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))
    model.eval()
    model(adj, features)

    x_all = gcn_layer1_output[0]
    y_train = np.argmax(y_train[train_mask, :].numpy(), axis=1)
    y_val = np.argmax(y_val[val_mask, :].numpy(), axis=1)
    y_test = np.argmax(y_test[test_mask, :].numpy(), axis=1)
    tsne_vis(x_all[train_mask], y_train, classnames[args.dataset], 'train_set', args.dataset)
    tsne_vis(x_all[val_mask], y_val, classnames[args.dataset], 'val_set', args.dataset)
    tsne_vis(x_all[test_mask], y_test, classnames[args.dataset], 'test_set', args.dataset)


if __name__ == '__main__':
    gcn_layer1_output = []
    visualize()
