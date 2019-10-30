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


def build_optimizer(model, lr, weight_decay):
    gcn1, gcn2 = [], []
    for name, p in model.named_parameters():
        if 'layer1' in name:
            gcn1.append(p)
        else:
            gcn2.append(p)
    opt = optim.Adam([{'params': gcn1, 'weight_decay': weight_decay},
                      {'params': gcn2}
                      ], lr=lr)
    return opt


def get_lr():
    pass


def get_loss(output, labels, mask):
    loss = Loss()
    return loss(output, labels, mask)


def get_accuracy(outputs, labels, mask):
    outputs = torch.argmax(outputs, dim=1)
    labels = torch.argmax(labels, dim=1)
    outputs = outputs.numpy()
    labels = labels.numpy()
    correct = outputs == labels
    #print(correct)
    mask = mask.float().numpy()
    tp = np.sum(correct * mask)
    return tp / np.sum(mask)
