import networkx as nx
import numpy as np
import os
import pickle
import torch


def process_features(features):
    row_sum_diag = np.sum(features, axis=1)
    row_sum_diag_inv = np.power(row_sum_diag, -1)
    row_sum_diag_inv[np.isinf(row_sum_diag_inv)] = 0.
    row_sum_inv = np.diag(row_sum_diag_inv)
    return np.dot(row_sum_inv, features)


def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset):
    ## get data
    data_path = 'data'
    suffixs = ['x', 'y', 'allx', 'ally', 'tx', 'ty', 'graph']
    objects = []
    for suffix in suffixs:
        file = os.path.join(data_path, 'ind.%s.%s'%(dataset, suffix))
        objects.append(pickle.load(open(file, 'rb'), encoding='latin1'))
    x, y, allx, ally, tx, ty, graph = objects
    x, allx, tx = x.toarray(), allx.toarray(), tx.toarray()

    # test indices
    test_index_file = os.path.join(data_path, 'ind.%s.test.index'%dataset)
    with open(test_index_file, 'r') as f:
        lines = f.readlines()
    indices = [int(line.strip()) for line in lines]
    min_index, max_index = min(indices), max(indices)

    # preprocess test indices and combine all data
    tx_extend = np.zeros((max_index - min_index + 1, tx.shape[1]))
    features = np.vstack([allx, tx_extend])
    features[indices] = tx
    ty_extend = np.zeros((max_index - min_index + 1, ty.shape[1]))
    labels = np.vstack([ally, ty_extend])
    labels[indices] = ty

    # get adjacency matrix
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph)).toarray()

    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)
    idx_test = indices

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])
    zeros = np.zeros(labels.shape)
    y_train = zeros.copy()
    y_val = zeros.copy()
    y_test = zeros.copy()
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    features = torch.from_numpy(process_features(features))
    y_train, y_val, y_test, train_mask, val_mask, test_mask = \
        torch.from_numpy(y_train), torch.from_numpy(y_val), torch.from_numpy(y_test), \
        torch.from_numpy(train_mask), torch.from_numpy(val_mask), torch.from_numpy(test_mask)

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
