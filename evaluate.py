import argparse
import torch
from datasets import load_data
from models.gcn import GCN
from models.utils import get_accuracy


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='citeseer', help='Dataset to train')
parser.add_argument('--hidden_dim', type=list, default=16, help='Dimensions of hidden layers')
parser.add_argument('--checkpoint', type=str, help='Directory to save checkpoints')
args = parser.parse_args()


adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(args.dataset)
model = GCN(features.shape[1], args.hidden_dim, y_train.shape[1], 0)


def evaluate(checkpoint):
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    outputs = model(adj, features)
    accuracy = get_accuracy(outputs, y_test, test_mask)
    print("Accuracy on test set is %f" %accuracy)


if __name__ == '__main__':
    evaluate(args.checkpoint)