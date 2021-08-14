#!/usr/bin/python3
# -*-coding:utf-8 -*-
# Reference:**********************************************
# @Time     : 2020/9/17 15:32
# @Author   : Raymond Luo
# @File     : generate_graph.py
# @User     : luoli
# @Software: PyCharm
# Reference:**********************************************


import scipy.io
import scipy.sparse as sp
import numpy as np
import pickle

# data_file_path = 'DBLP.mat'
# data = scipy.io.loadmat(data_file_path)
# print(list(data.keys()))
# print(data['a_feature'].shape)
# print(data['p_vs_a'].shape)
# print(data['p_vs_c'].shape)
# print(data['p_vs_t'].shape)
# exit()

with open("author_features_334.pickle", "rb") as f:
    author_features = pickle.load(f)

author_idx_map = {}
conf_idx_map = {}
paper_idx_map = {}
term_idx_map = {}


def text2map(f, mapdict):
    for i, l in enumerate(f.readlines()):
        l = l.replace("\n", "")
        idx, item = l.split("\t")
        if item not in mapdict:
            mapdict[int(idx)] = i


def build_edge(f, edgelist, start_map_dict, end_map_dict):
    for i, l in enumerate(f.readlines()):
        l = l.replace("\n", "")
        start, end = l.split("\t")
        if int(start) in start_map_dict and int(end) in end_map_dict:
            if not [start_map_dict[int(start)], end_map_dict[int(end)]] in edgelist:
                edgelist.append([start_map_dict[int(start)], end_map_dict[int(end)]])


with open('author.txt') as f:
    text2map(f, author_idx_map)
with open('conf.txt') as f:
    text2map(f, conf_idx_map)
with open('paper.txt', encoding='gbk') as f:
    text2map(f, paper_idx_map)
with open('term.txt') as f:
    text2map(f, term_idx_map)

paper_author_edges = []
paper_conf_edges = []
paper_term_edges = []

with open('paper_author.txt') as f:
    build_edge(f, paper_author_edges, paper_idx_map, author_idx_map)
with open('paper_conf.txt') as f:
    build_edge(f, paper_conf_edges, paper_idx_map, conf_idx_map)
with open('paper_term.txt') as f:
    build_edge(f, paper_term_edges, paper_idx_map, term_idx_map)

paper_author_edges = np.array(paper_author_edges)
paper_conf_edges = np.array(paper_conf_edges)
paper_term_edges = np.array(paper_term_edges)

paper_author_adj = sp.coo_matrix(
    (np.ones(paper_author_edges.shape[0]), (paper_author_edges[:, 0], paper_author_edges[:, 1])),
    shape=(len(paper_idx_map), len(author_idx_map)), dtype=np.int32)

paper_conf_adj = sp.coo_matrix(
    (np.ones(paper_conf_edges.shape[0]), (paper_conf_edges[:, 0], paper_conf_edges[:, 1])),
    shape=(len(paper_idx_map), len(conf_idx_map)), dtype=np.int32)

paper_term_adj = sp.coo_matrix(
    (np.ones(paper_term_edges.shape[0]), (paper_term_edges[:, 0], paper_term_edges[:, 1])),
    shape=(len(paper_idx_map), len(term_idx_map)), dtype=np.int32)

scipy.io.savemat('DBLP.mat', {"a_feature": author_features, "p_vs_a": paper_author_adj, "p_vs_c": paper_conf_adj,
                              "p_vs_t": paper_term_adj})
