# CP-GNN

----

Official code implementation for CIKM 21 paper Detecting Communities from Heterogeneous Graphs: A Context Path-based Graph Neural Network Model

> Linhao Luo, Yixiang Fang, Xin Cao, Xiaofeng Zhang, and Wenjie Zhang. 2021. Detecting Communities from Heterogeneous Graphs: A Context Path-based Graph Neural Network Model. In Proceedings of the 30th ACM International Conference on Information & Knowledge Management (CIKM '21). Association for Computing Machinery, New York, NY, USA, 1170–1180. DOI:https://doi.org/10.1145/3459637.3482250

## Environments

python == 3.6

CPU: I7-8700K 

RAM: 64GB

GPU: RTX 3080 

CUDA: 10.1

## Requirements

```bash
torch==1.6.0
matplotlib==2.2.3
networkx==2.4
dgl==0.4.3.post2
numpy==1.16.6
scipy==1.4.1
scikit-learn==0.21.3
```

## Config

```
vi config.py
```
config.py example

```python
import os

config_path = os.path.dirname(__file__)
data_config = {
    'data_path': os.path.join(config_path, 'data'),
    'dataset': 'ACM', # ACM, DBLP, IMDB, AIFB
    'data_name': 'ACM.mat', # ACM.mat, DBLP.mat, IMDB.mat, AIFB.mat
    'primary_type': 'p', # p, a, m, Personen
    'task': ['CF', 'CL'],
    'K_length': 4, # Context path length K
    'resample': False, # Whether resample the training and testing dataset
    'random_seed': 123,
    'test_ratio': 0.8
}

model_config = {
    'primary_type': data_config['primary_type'],
    'auxiliary_embedding': 'non_linear',  # auxiliary embedding generating method: non_linear, linear, emb
    'K_length': data_config['K_length'],
    'embedding_dim': 128,
    'in_dim': 128,
    'out_dim': 128,
    'num_heads': 8,
    'merge': 'linear',  # Multi head Attention merge method: linear, mean, stack
    'g_agg_type': 'mean',  # Graph representation encoder: mean, sum
    'drop_out': 0.3,
    'cgnn_non_linear': True,  # Enable non linear activation function for CGNN
    'multi_attn_linear': False,  # Enable atten K/Q-linear for each type
    'graph_attention': True,
    'kq_linear_out_dim': 128,
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
    'batch_size': 1024 * 20,
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
}
```

## Train and Evaluate
``` bash
python3 main.py
```

## BibTex
```tex
@inproceedings{10.1145/3459637.3482250,
author = {Luo, Linhao and Fang, Yixiang and Cao, Xin and Zhang, Xiaofeng and Zhang, Wenjie},
title = {Detecting Communities from Heterogeneous Graphs: A Context Path-Based Graph Neural Network Model},
year = {2021},
isbn = {9781450384469},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3459637.3482250},
doi = {10.1145/3459637.3482250},
booktitle = {Proceedings of the 30th ACM International Conference on Information & Knowledge Management},
pages = {1170–1180},
numpages = {11},
keywords = {unsupervised learning, graph neural network, community detection, heterogeneous graphs, context path},
location = {Virtual Event, Queensland, Australia},
series = {CIKM '21}
}
```
