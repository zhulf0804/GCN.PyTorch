import argparse
import os
import torch
from datasets import load_data, process_features
from models.gcn import GCN
from models.utils import build_optimizer, get_loss, get_accuracy
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='citeseer', help='Dataset to train')
parser.add_argument('--init_lr', type=float, default=0.036, help='Initial learing rate')
parser.add_argument('--epoches', type=int, default=200, help='Number of traing epoches')
parser.add_argument('--hidden_dim', type=list, default=16, help='Dimensions of hidden layers')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep  probability)')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight for l2 loss on embedding matrix')
parser.add_argument('--log_interval', type=int, default=10, help='Print iterval')
parser.add_argument('--log_dir', type=str, default='experiments', help='Dataset to train')
args = parser.parse_args()


adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(args.dataset)
features = process_features(features)
y_train, y_val, y_test, train_mask, val_mask, test_mask = \
    torch.from_numpy(y_train), torch.from_numpy(y_val), torch.from_numpy(y_test), \
    torch.from_numpy(train_mask), torch.from_numpy(val_mask), torch.from_numpy(test_mask)


model = GCN(features.shape[1], args.hidden_dim, y_train.shape[1])
optimizer = build_optimizer(model.parameters(), args.init_lr)


def evaluate(outputs, labels, mask):
    model.eval()
    accuracy = get_accuracy(outputs, labels, mask)
    model.train()
    return accuracy


def train():
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    writer = SummaryWriter(args.log_dir)
    for epoch in range(args.epoches):
        outputs = model(adj, features)
        loss = get_loss(outputs, y_train, train_mask)
        val_loss = get_loss(outputs, y_val, val_mask).detach().numpy()
        optimizer.zero_grad()  # Important
        loss.backward()
        optimizer.step()
        train_accuracy = evaluate(outputs, y_train, train_mask)
        val_accuracy = evaluate(outputs, y_val, val_mask)
        writer.add_scalars('loss', {'train_loss': loss.detach().numpy(), 'val_loss': val_loss}, epoch)
        writer.add_scalars('accuracy', {'train_ac': train_accuracy, 'val_ac': val_accuracy}, epoch)
        if epoch % args.log_interval == 0:
            print("Epoch: %d, train loss: %f, val loss: %f, train ac: %f, val ac: %f"
                  %(epoch, loss.detach().numpy(), val_loss, train_accuracy, val_accuracy))
    writer.close()


if __name__ == '__main__':
    train()


