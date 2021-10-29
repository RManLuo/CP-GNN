#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/8/4 22:38
# @Author  : Raymound luo
# @Mail    : luolinhao1998@gmail.com
# @File    : config.py
# @Software: PyCharm
# @Describe:

import os

config_path = os.path.dirname(__file__)
data_config = {
    'data_path': os.path.join(config_path, 'data'),
    'dataset': 'DBLP',
    'data_name': 'DBLP.mat',
    'primary_type': 'a',
    'task': ['CF', 'CL'],
    'K_length': 3,
    'ignore_edge_type': True,
    'resample': False,
    'random_seed': 123,
    'test_ratio': 0.8
}

model_config = {
    'primary_type': data_config['primary_type'],
    'auxiliary_embedding': 'non_linear',  # auxiliary embedding generating method: non_linear, linear, embedding
    'K_length': data_config['K_length'],
    'embedding_dim': 256,
    'in_dim': 256,
    'out_dim': 256,
    'num_heads': 8,
    'merge': 'linear',  # Multi head Attention merge method: linear, mean, stack
    'g_agg_type': 'mean',  # Graph representation encoder: mean, sum
    'drop_out': 0.3,
    'cgnn_non_linear': True,  # Enable non linear activation function for CGNN
    'multi_attn_linear': False,  # Enable atten K/Q-linear for each type
    'graph_attention': True,
    'kq_linear_out_dim': 256,
    'path_attention': False,  # Enable Context path attention
    'c_linear_out_dim': 8,
    'enable_bilinear': False,  # Enable Bilinear for context attention
    'gru': True,
    'add_init': False
}

train_config = {
    'continue': False,
    'lr': 0.05,
    'l2': 0,
    'factor': 0.2,
    'total_epoch': 10000000,
    'batch_size': 1024,
    'pos_num_for_each_hop': [20, 20, 20, 20, 20, 20, 20, 20, 20],
    'neg_num_for_each_hop': [3, 3, 3, 3, 3, 3, 3, 3, 3],
    'sample_workers': 8,
    'patience': 15,
    'checkpoint_path': os.path.join(config_path, 'checkpoint', data_config['dataset'])
}

evaluate_config = {
    'method': 'LR',
    'save_heat_map': True,
    'result_path': os.path.join('result', data_config['dataset']),
    'random_state': 123,
    'max_iter': 500,
    'n_jobs': 1,
    'save_heat_map': False
}
