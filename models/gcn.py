import numpy as np
import torch
import torch.nn.functional as Func


def preprocess_adj(A):
    '''
    Pre-process adjacency matrix
    :param A: adjacency matrix
    :return:
    '''
    I = np.eye(A.shape[0])
    A_hat = A + I # add self-loops
    D_hat_diag = np.sum(A_hat, axis=1)
    D_hat_diag_inv_sqrt = np.power(D_hat_diag, -0.5)
    D_hat_diag_inv_sqrt[np.isinf(D_hat_diag_inv_sqrt)] = 0.
    D_hat_inv_sqrt = np.diag(D_hat_diag_inv_sqrt)
    return np.dot(np.dot(D_hat_inv_sqrt, A_hat), D_hat_inv_sqrt)


class GCNLayer():
    def __init__(self):
        pass
    def __call__(self, A, F, W):
        tmp = torch.mm(A, F)
        return Func.relu(torch.mm(tmp, W))


class GCN():
    def __init__(self, input_dim, hidden_dims, num_classes):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.hidden_dims.insert(0, self.input_dim)
        self.num_classes = num_classes
        self.gcn_layer = GCNLayer()
    def __call__(self, A, X):
        A = torch.from_numpy(preprocess_adj(A))
        print(A)
        F = torch.from_numpy(X)
        for i in range(len(self.hidden_dims)-1):
            W = torch.randn((self.hidden_dims[i], self.hidden_dims[i+1]), dtype=torch.double, requires_grad=True)
            F = self.gcn_layer(A, F, W)
        W = torch.randn((self.hidden_dims[-1], self.num_classes), dtype=torch.double, requires_grad=True)
        self.outputs = self.gcn_layer(A, F, W)
        return self.outputs


def test_gcn():
    A = np.random.choice([0, 1], (3000, 3000))
    X = np.random.randn(3000, 4000)
    gcn = GCN(4000, [16], 6)
    y = gcn(A, X)
    print(y)

if __name__ == '__main__':
    test_gcn()
