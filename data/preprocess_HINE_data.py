#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/1/14 14:00
# @Author  : Raymond luo
# @Mail    : luolinhao1998@gmail.com
# @File    : preprocess_HINE_data.py
# @Software: PyCharm

import scipy.io
import sys
import os
from dgl.contrib.sampling.sampler import LayerSampler
from dgl.sampling import sample_neighbors
import torch as th
from torch.utils.data import DataLoader
import dgl
import numpy as np

sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from config import data_config, train_config
import config
from utils import load_data
import json

if __name__ == "__main__":
    data_config['dataset'] = 'AIFB'
    data_config['data_name'] = 'AIFB.mat'
    data_config['primary_type'] = 'Personen'

    data_loader = load_data(data_config)
    if data_config['dataset'] in ['AIFB', 'AM', 'BGS', 'MUTAG']:
        data_config['primary_type'] = data_loader.predict_category
    features, labels, num_classes, train_idx, test_idx = data_loader.load_classification_data()
    hg = data_loader.heter_graph
    save_path = os.path.dirname(__file__) + os.sep + '../' + 'OpenHINE/dataset/' + data_config['dataset']
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fin_edges = []
    nodes_info = ""

    if data_config['dataset'] in ['AIFB', 'AM', 'BGS', 'MUTAG']:
        ntype_dict = {}
        init_label = 'a'
        for ntype in hg.ntypes:
            if ntype not in ntype_dict:
                ntype_dict[ntype] = init_label
                nodes_info += init_label
                init_label = chr(ord(init_label) + 1)
        with open(os.path.join(save_path, 'ntype_dict.json'), 'w') as f:
            json.dump(ntype_dict, f)
        edges_info = ""
        for srctype, etype, dsttype in hg.canonical_etypes:
            src_id, dst_id = hg.all_edges(form='uv', etype=(srctype, etype, dsttype))
            src_id = src_id.tolist()
            dst_id = dst_id.tolist()
            srctype = ntype_dict[srctype]
            dsttype = ntype_dict[dsttype]
            edges_info += srctype + '-' + dsttype + "+"
            edge_type_list = [srctype + '-' + dsttype] * len(src_id)
            weight_list = [1] * len(src_id)
            edges = list(zip(src_id, dst_id, edge_type_list, weight_list))
            fin_edges.extend(edges)
    else:
        for ntype in hg.ntypes:
            nodes_info += ntype
        edges_info = ""
        for srctype, etype, dsttype in hg.canonical_etypes:
            src_id, dst_id = hg.all_edges(form='uv', etype=(srctype, etype, dsttype))
            src_id = src_id.tolist()
            dst_id = dst_id.tolist()
            edges_info += srctype + '-' + dsttype + "+"
            edge_type_list = [srctype + '-' + dsttype] * len(src_id)
            weight_list = [1] * len(src_id)
            edges = list(zip(src_id, dst_id, edge_type_list, weight_list))
            fin_edges.extend(edges)

    with open(os.path.join(save_path, 'edge.txt'), 'w') as f:
        for src_id, dst_id, etype, weight in fin_edges:
            f.write("%s\t%s\t%s\t%s\n" % (src_id, dst_id, etype, weight))

    x_idx = np.concatenate([train_idx, test_idx])
    y = labels[x_idx]
    label_data = list(zip(x_idx, y))
    with open(os.path.join(save_path, 'label.txt'), 'w') as f:
        for x, y in label_data:
            f.write("%s%s\t%s\n" % (data_config['primary_type'], x, y))
    with open(os.path.join(save_path, 'edges_info.txt'), 'w') as f:
        f.write(nodes_info + '\n')
        f.write(edges_info)

