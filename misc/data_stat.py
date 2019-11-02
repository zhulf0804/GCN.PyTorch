import numpy as np
import os
import sys
from datasets import load_data


def get_citeseer():
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('citeseer')
    y_train, y_val, y_test = y_train.numpy(), y_val.numpy(), y_test.numpy()
    train_num = np.sum(y_train)
    val_num = np.sum(y_val)
    test_num = np.sum(y_test)
    print("="*20, "citeseer", "="*20)
    print("train set num is %d, val set num is %d, test set num is %d" %(train_num, val_num, test_num))
    classes = [[] for _ in range(3)]
    for i in range(6):
        classes[0].append(np.sum(y_train[:, i]))
        classes[1].append(np.sum(y_val[:, i]))
        classes[2].append(np.sum(y_test[:, i]))
    types = ['train set', 'val set', 'test set']
    classes = np.array(classes, dtype=np.int)
    for i in range(3):
        print("each class num in %s: "%types[i], classes[i])


def get_cora():
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('cora')
    y_train, y_val, y_test = y_train.numpy(), y_val.numpy(), y_test.numpy()
    train_num = np.sum(y_train)
    val_num = np.sum(y_val)
    test_num = np.sum(y_test)
    print("=" * 20, "citeseer", "=" * 20)
    print("train set num is %d, val set num is %d, test set num is %d" % (train_num, val_num, test_num))
    classes = [[] for _ in range(3)]
    for i in range(7):
        classes[0].append(np.sum(y_train[:, i]))
        classes[1].append(np.sum(y_val[:, i]))
        classes[2].append(np.sum(y_test[:, i]))
    types = ['train set', 'val set', 'test set']
    classes = np.array(classes, dtype=np.int)
    for i in range(3):
        print("each class num in %s: " % types[i], classes[i])


def get_pubmed():
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('pubmed')
    y_train, y_val, y_test = y_train.numpy(), y_val.numpy(), y_test.numpy()
    train_num = np.sum(y_train)
    val_num = np.sum(y_val)
    test_num = np.sum(y_test)
    print("=" * 20, "citeseer", "=" * 20)
    print("train set num is %d, val set num is %d, test set num is %d" % (train_num, val_num, test_num))
    classes = [[] for _ in range(3)]
    for i in range(3):
        classes[0].append(np.sum(y_train[:, i]))
        classes[1].append(np.sum(y_val[:, i]))
        classes[2].append(np.sum(y_test[:, i]))
    types = ['train set', 'val set', 'test set']
    classes = np.array(classes, dtype=np.int)
    for i in range(3):
        print("each class num in %s: " % types[i], classes[i])


father_path = os.path.dirname(sys.path[0])
os.chdir(father_path)
get_citeseer()
get_cora()
get_pubmed()