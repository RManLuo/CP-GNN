import pickle
import dgl
from dgl.data import *
import scipy.io
import scipy.sparse as sp
import torch
import json

data_list = ['AIFB', 'MUTAG', 'BGS', 'AM']


def trans(name):
    if name == "MUTAG":
        dataset = MUTAGDataset()
    elif name == 'AIFB':
        dataset = AIFBDataset()
    elif name == "BGS":
        dataset = BGSDataset()
    elif name == 'AM':
        dataset = AMDataset()
    g = dataset[0]
    category = dataset.predict_category
    num_classes = dataset.num_classes
    train_mask = g.nodes[category].data.pop('train_mask')
    test_mask = g.nodes[category].data.pop('test_mask')
    labels = g.nodes[category].data.pop('labels').tolist()

    train_idx = torch.nonzero(train_mask).squeeze().tolist()
    test_idx = torch.nonzero(test_mask).squeeze().tolist()
    edge_list = []
    mat_dict = {}
    for srctype, etype, dsttype in g.canonical_etypes:
        canonical_etypes = (srctype, etype, dsttype)
        edge_type = srctype.strip('_')+'||'+etype.strip('_')+'||' + dsttype.strip('_')
        mat_dict[edge_type] = g.adj(scipy_fmt='coo', etype=canonical_etypes)
        edge_list.append(edge_type)

    info_dict = {'num_classes': num_classes, 'predict_category': category, 'train_idx': train_idx,
                 'test_idx': test_idx, 'labels': labels, 'ntypes': g.ntypes, 'etypes': g.etypes, 'edge_list': edge_list}
    scipy.io.savemat('{}.mat'.format(name), mat_dict)
    with open('{}_info.json'.format(name), 'w') as f:
        json.dump(info_dict, f)
    # data = scipy.io.loadmat('{}.mat'.format(name))
    # print(data.keys)
    # print()

[trans(name) for name in data_list]
