import argparse
import torch
from datasets import load_data, process_features
from models.gcn import GCN
from models.utils import build_optimizer, get_loss

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='citeseer', help='Dataset to train')
parser.add_argument('--init_lr', type=float, default=0.036, help='Initial learing rate')
parser.add_argument('--epoches', type=int, default=200, help='Number of traing epoches')
parser.add_argument('--hidden_dim', type=list, default=16, help='Dimensions of hidden layers')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep  probability)')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight for l2 loss on embedding matrix')
args = parser.parse_args()



adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(args.dataset)
features = process_features(features)
y_train, y_val, y_test, train_mask, val_mask, test_mask = \
    torch.from_numpy(y_train), torch.from_numpy(y_val), torch.from_numpy(y_test), \
    torch.from_numpy(train_mask), torch.from_numpy(val_mask), torch.from_numpy(test_mask)

model = GCN(features.shape[1], args.hidden_dim, y_train.shape[1])

optimizer = build_optimizer(model.parameters(), args.init_lr)

for epoch in range(args.epoches):
    y = model(adj, features)
    loss = get_loss(y, y_train, train_mask)
    print(loss.detach().numpy())
    optimizer.zero_grad()  # Important
    loss.backward()
    optimizer.step()
