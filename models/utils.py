import torch.optim as optim
def build_optimizer(params, lr):
    opt = optim.Adam(params, lr)
    return opt


def get_lr():
    pass


def get_loss():
    pass