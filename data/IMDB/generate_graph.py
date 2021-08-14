#!/usr/bin/python3
# -*-coding:utf-8 -*-
# Reference:**********************************************
# @Time     : 2020/9/17 15:32
# @Author   : Raymond Luo
# @File     : generate_graph.py
# @User     : luoli
# @Software: PyCharm
# Reference:**********************************************
import pickle
import scipy.io
import scipy.sparse as sp
import numpy as np

original_file = './movie_metadata_3class.csv'

with open("movie_feature_vector_6334.pickle", "rb") as f:
    movies_features = pickle.load(f)

movie_idx_map = {}
actor_idx_map = {}
director_idx_map = {}
keyword_idx_map = {}


def text2map(f, mapdict):
    for i, l in enumerate(f.readlines()):
        l = l.replace("\n", "")
        idx, item = l.split(",")
        if not item:
            item = "Unknown"
        if item not in mapdict:
            mapdict[item] = int(idx)
        else:
            print(l)
            print(mapdict[item])


with open('actor_index.txt') as f:
    text2map(f, actor_idx_map)
with open('director_index.txt') as f:
    text2map(f, director_idx_map)
with open('movie_index.txt') as f:
    text2map(f, movie_idx_map)
with open('keyword_index.txt') as f:
    text2map(f, keyword_idx_map)

movie_actor_edges = []
movie_director_edges = []
movie_keyword_edges = []

with open(original_file, 'r') as f:
    next(f)
    lines = f.readlines()
    for line in lines:
        line = line.split(',')
        movie = line[11]
        actor_1 = line[6]
        actor_2 = line[10]
        actor_3 = line[14]
        director = line[1]
        keywords = line[16].split('|')
        if movie not in movie_idx_map:
            continue
        if actor_1 in actor_idx_map:
            if [movie_idx_map[movie], actor_idx_map[actor_1]] not in movie_actor_edges:
                movie_actor_edges.append([movie_idx_map[movie], actor_idx_map[actor_1]])
        if actor_2 in actor_idx_map:
            if [movie_idx_map[movie], actor_idx_map[actor_2]] not in movie_actor_edges:
                movie_actor_edges.append([movie_idx_map[movie], actor_idx_map[actor_2]])
        if actor_3 in actor_idx_map:
            if [movie_idx_map[movie], actor_idx_map[actor_3]] not in movie_actor_edges:
                movie_actor_edges.append([movie_idx_map[movie], actor_idx_map[actor_3]])
        if director in director_idx_map:
            if [movie_idx_map[movie], director_idx_map[director]] not in movie_director_edges:
                movie_director_edges.append([movie_idx_map[movie], director_idx_map[director]])
        for keyword in keywords:
            if keyword in keyword_idx_map:
                keyword_idx = keyword_idx_map[keyword]
                if [movie_idx_map[movie], keyword_idx] not in movie_keyword_edges:
                    movie_keyword_edges.append([movie_idx_map[movie], keyword_idx])
        if movie not in movie_idx_map:
            continue
        if actor_1 in actor_idx_map:
            if [movie_idx_map[movie], actor_idx_map[actor_1]] not in movie_actor_edges:
                movie_actor_edges.append([movie_idx_map[movie], actor_idx_map[actor_1]])
        if actor_2 in actor_idx_map:
            if [movie_idx_map[movie], actor_idx_map[actor_2]] not in movie_actor_edges:
                movie_actor_edges.append([movie_idx_map[movie], actor_idx_map[actor_2]])
        if actor_3 in actor_idx_map:
            if [movie_idx_map[movie], actor_idx_map[actor_3]] not in movie_actor_edges:
                movie_actor_edges.append([movie_idx_map[movie], actor_idx_map[actor_3]])
        if director in director_idx_map:
            if [movie_idx_map[movie], director_idx_map[director]] not in movie_director_edges:
                movie_director_edges.append([movie_idx_map[movie], director_idx_map[director]])
        for keyword in keywords:
            if keyword in keyword_idx_map:
                keyword_idx = keyword_idx_map[keyword]
                if [movie_idx_map[movie], keyword_idx] not in movie_keyword_edges:
                    movie_keyword_edges.append([movie_idx_map[movie], keyword_idx])

movie_actor_edges = np.array(movie_actor_edges)
movie_actor_adj = sp.coo_matrix(
    (np.ones(movie_actor_edges.shape[0]), (movie_actor_edges[:, 0], movie_actor_edges[:, 1])),
    shape=(len(movie_idx_map), len(actor_idx_map)), dtype=np.int32)
# movie_actor_adj =movie_actor_adj.todense()
movie_director_edges = np.array(movie_director_edges)
movie_director_adj = sp.coo_matrix(
    (np.ones(movie_director_edges.shape[0]), (movie_director_edges[:, 0], movie_director_edges[:, 1])),
    shape=(len(movie_idx_map), len(director_idx_map)), dtype=np.int32)
# movie_director_adj =movie_director_adj.todense()
movie_keyword_edges = np.array(movie_keyword_edges)
movie_keyword_adj = sp.coo_matrix(
    (np.ones(movie_keyword_edges.shape[0]), (movie_keyword_edges[:, 0], movie_keyword_edges[:, 1])),
    shape=(len(movie_idx_map), len(keyword_idx_map)), dtype=np.int32)

scipy.io.savemat('IMDB.mat', {"m_feature": movies_features, "m_vs_a": movie_actor_adj, "m_vs_d": movie_director_adj,
                              "m_vs_k": movie_keyword_adj})
import scipy.io

# data_file_path = 'IMDB.mat'
# data = scipy.io.loadmat(data_file_path)
# print(list(data.keys()))
