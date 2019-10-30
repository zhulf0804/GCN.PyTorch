import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction='none')
    def forward(self, output, labels, mask):
        labels = torch.argmax(labels, dim=1)
        loss = self.loss(output, labels)
        mask = mask.float()
        mask /= torch.mean(mask)
        loss *= mask
        return torch.mean(loss)


def build_optimizer(params, lr):
    opt = optim.Adam(params, lr)
    return opt


def get_lr():
    pass


def get_loss(output, labels, mask):
    loss = Loss()
    return loss(output, labels, mask)


def get_accuracy(outputs, labels, mask):
    outputs = torch.argmax(outputs, dim=1)
    labels = torch.argmax(labels, dim=1)
    outputs = outputs.cpu().numpy()
    labels = labels.cpu().numpy()
    correct = outputs == labels
    #print(correct)
    mask = mask.float().numpy()
    tp = np.sum(correct * mask)
    return tp / np.sum(mask)
