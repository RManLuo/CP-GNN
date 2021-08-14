#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/8/5 10:08
# @Author  : Raymound luo
# @Mail    : luolinhao1998@gmail.com
# @File    : preprocess.py
# @Software: PyCharm
# @Describe:
import json
import dgl
import numpy as np
import os
import pickle

import torch
from dgl.sampling import sample_neighbors
from scipy.sparse import coo_matrix
from scipy import io as sio
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, Dataset


class EdgesDataset(Dataset):
    def __init__(self, hg, pos_sample_num, neg_sample_num):
        self.hg = hg
        self.pos_sample_num = pos_sample_num
        self.neg_sample_num = neg_sample_num

    def __len__(self):
        return self.hg.number_of_nodes()

    def __getitem__(self, idx):
        pos_graph = sample_neighbors(self.hg, [idx], self.pos_sample_num, edge_dir='out')
        pos_src, pos_dst = pos_graph.edges()
        neg_src = pos_src.repeat(self.neg_sample_num)
        neg_dst = torch.randint(0, self.hg.number_of_nodes(), neg_src.shape, dtype=torch.long)
        return pos_src, pos_dst, neg_src, neg_dst

    @staticmethod
    def collate(batches):
        pos_src = []
        pos_dst = []
        neg_src = []
        neg_dst = []
        for item in batches:
            pos_src.append(item[0])
            pos_dst.append(item[1])
            neg_src.append(item[2])
            neg_dst.append(item[3])
        return torch.cat(pos_src), torch.cat(pos_dst), torch.cat(neg_src), torch.cat(neg_dst)


class GraphDataLoader(object):
    def __init__(self, data_config, remove_self_loop):
        self.data_config = data_config
        self.base_data_path = os.path.join(data_config['data_path'], data_config['dataset'])
        self.data_path = os.path.join(data_config['data_path'], data_config['dataset'], data_config['data_name'])
        self.train_data_path = os.path.join(data_config['data_path'], data_config['dataset'], 'train_data')
        if not os.path.exists(self.train_data_path):
            os.mkdir((self.train_data_path))
        self.k_hop_graph_path = os.path.join(self.train_data_path, 'graph')
        if not os.path.exists(self.k_hop_graph_path):
            os.mkdir(self.k_hop_graph_path)

    def load_raw_matrix(self):
        raise NotImplementedError("Not Implement load_raw_matrix")

    def load_k_hop_train_data(self):
        raise NotImplementedError("Not Implement load_k_hop_train_data method")

    def load_classification_data(self):
        raise NotImplementedError("Not Implement load_classification_data method")

    def load_links_prediction_data(self):
        raise NotImplementedError("Not Implement load_links_prediction_data method")

    def _load_k_hop_graph(self, hg, k, primary_type):
        '''
        Return k hop neighbors graph
        :param hg: DGLHeteroGraph
        :param k: hop neighbors
        :param primary_type: primary_type
        :return: k-hop graph of primary type graph, DGLHeteroGraph
        '''
        print('Process: {} hop graph'.format(k))
        k_hop_graph_path = os.path.join(self.k_hop_graph_path,
                                        '{}_{}_hop_graph.pkl'.format(primary_type, k))
        if not os.path.exists(k_hop_graph_path):
            ntype = hg.ntypes
            primary_type_id = ntype.index(primary_type)
            homo_g = dgl.to_homo(hg)
            p_nodes_id = homo_g.filter_nodes(
                lambda nodes: (nodes.data['_TYPE'] == primary_type_id))  # Find the primary nodes ID
            min_p = torch.min(p_nodes_id).item()
            max_p = torch.max(p_nodes_id).item()
            raw_adj = homo_g.adjacency_matrix()  # It is a square matrix
            # Speed up with torch
            raw_adj = raw_adj.to_dense().float()
            adj_k = torch.matrix_power(raw_adj, k)  # K-hop neighbors
            p_adj = adj_k[min_p:max_p, min_p:max_p].cpu()  # Get primary sub graph
            row, col = torch.nonzero(p_adj, as_tuple=True)
            p_g = dgl.graph((row, col))
            with open(k_hop_graph_path, 'wb') as f:
                pickle.dump(p_g, f, protocol=4)
        else:
            with open(k_hop_graph_path, 'rb') as f:
                p_g = pickle.load(f)
        return p_g

    def load_train_k_context_edges(self, hg, K, primary_type, pos_num_for_each_hop, neg_num_for_each_hop):
        edges_data_dict = {}
        for k in range(1, K + 2):
            k_hop_primary_graph = self._load_k_hop_graph(hg, k, primary_type)
            k_hop_edge = EdgesDataset(k_hop_primary_graph, pos_num_for_each_hop[k], neg_num_for_each_hop[k])
            edges_data_dict[k] = k_hop_edge
        return edges_data_dict


class ACMDataLoader(GraphDataLoader):
    def __init__(self, data_config, remote_self_loop):
        super(ACMDataLoader, self).__init__(data_config, remote_self_loop)

        self.heter_graph, self.raw_matrix = self.load_raw_matrix()

    def load_raw_matrix(self):
        data = sio.loadmat(self.data_path)
        '''
        ['__header__', '__version__', '__globals__', 'TvsP', 'PvsA', 'PvsV', 'AvsF', 'VvsC', 'PvsL', 'PvsC', 'A', 'C', 'F', 'L', 'P', 'T', 'V', 'PvsT', 'CNormPvsA', 'RNormPvsA', 'CNormPvsC', 'RNormPvsC', 'CNormPvsT', 'RNormPvsT', 'CNormPvsV', 'RNormPvsV', 'CNormVvsC', 'RNormVvsC', 'CNormAvsF', 'RNormAvsF', 'CNormPvsL', 'RNormPvsL', 'stopwords', 'nPvsT', 'nT', 'CNormnPvsT', 'RNormnPvsT', 'nnPvsT', 'nnT', 'CNormnnPvsT', 'RNormnnPvsT', 'PvsP', 'CNormPvsP', 'RNormPvsP']
        P: Paper
        A：Author
        F: Facility
        C: Conference
        L: Subject
        '''
        p_vs_l = data['PvsL']  # paper-Subject
        p_vs_p = data['PvsP']  # paper-paper
        p_vs_a = data['PvsA']  # paper-author
        a_vs_f = data['AvsF']  # author-facility

        hg = dgl.heterograph({
            ('p', 'pa', 'a'): p_vs_a,
            ('a', 'ap', 'p'): p_vs_a.transpose(),
            ('p', 'pp', 'p'): p_vs_p,  # P cite P
            ('p', 'ps', 's'): p_vs_l,
            ('s', 'sp', 'p'): p_vs_l.transpose(),
            ('a', 'af', 'f'): a_vs_f,
            ('f', 'fa', 'a'): a_vs_f.transpose(),
        })

        return hg, data

    def load_classification_data(self):
        task_path = os.path.join(self.data_config['data_path'], self.data_config['dataset'], 'CF')
        if not os.path.exists(task_path):
            os.mkdir(task_path)
        task_data_path = os.path.join(self.data_config['data_path'], self.data_config['dataset'], 'CF',
                                      'data_test_{}.pkl'.format(self.data_config['test_ratio']))
        if self.data_config['resample'] or not os.path.exists(task_data_path):
            # We assign
            # (1) KDD papers as class 0 (data mining),
            # (2) SIGMOD and VLDB papers as class 1 (database),
            # (3) SIGCOMM and MOBICOMM papers as class 2 (communication)
            conf_ids = [0, 1, 9, 10, 13]
            label_ids = [0, 1, 2, 2, 1]

            p_vs_t = self.raw_matrix['PvsT']  # paper-term, bag of words, used for feature
            p_vs_c = self.raw_matrix['PvsC']  # paper-conference, labels come from that
            features = p_vs_t.toarray()  # torch.FloatTensor(p_vs_t.toarray())  # Feature

            p_selected = p_vs_c[:, conf_ids].tocoo().row
            pc_p, pc_c = p_vs_c.nonzero()
            labels = pc_c
            for conf_id, label_id in zip(conf_ids, label_ids):
                labels[labels == conf_id] = label_id
            # labels = torch.LongTensor(labels)

            num_classes = 3

            train_idx, test_idx = train_test_split(p_selected, test_size=self.data_config['test_ratio'],
                                                   random_state=self.data_config['random_seed'])
            # train_idx = torch.LongTensor(train_idx)
            # test_idx = torch.LongTensor(test_idx)
            with open(task_data_path, 'wb') as f:
                pickle.dump([features, labels, num_classes, train_idx, test_idx], f)

        else:
            with open(task_data_path, 'rb') as f:
                features, labels, num_classes, train_idx, test_idx = pickle.load(f)

        return features, labels, num_classes, train_idx, test_idx

    def load_links_prediction_data(self):
        task_path = os.path.join(self.data_config['data_path'], self.data_config['dataset'], 'LP')
        if not os.path.exists(task_path):
            os.mkdir(task_path)
        task_data_path = os.path.join(self.data_config['data_path'], self.data_config['dataset'], 'LP',
                                      'data_test_{}.pkl'.format(self.data_config['test_ratio']))
        if self.data_config['resample'] or not os.path.exists(task_data_path):
            primary_graph = self.heter_graph.node_type_subgraph(self.data_config['primary_type'])
            g = dgl.DGLGraph(primary_graph.to_networkx(), readonly=True)
            edgesampler = dgl.contrib.sampling.EdgeSampler(g, batch_size=g.number_of_edges(), num_workers=8,
                                                           negative_mode='tail',
                                                           neg_sample_size=1, exclude_positive=True)
            src_data = []
            dst_data = []
            labels = []
            for pos_g, neg_g in edgesampler:
                pos_edges = pos_g.edges()
                neg_edges = neg_g.edges()
                src_data.extend(pos_edges[0].tolist())
                dst_data.extend(pos_edges[1].tolist())
                labels.extend([1] * len(pos_edges[0]))

                src_data.extend(neg_edges[0].tolist())
                dst_data.extend(neg_edges[1].tolist())
                labels.extend([0] * len(neg_edges[0]))

            src_train, src_test, dst_train, dst_test, labels_train, labels_test = train_test_split(src_data, dst_data,
                                                                                                   labels,
                                                                                                   test_size=
                                                                                                   self.data_config[
                                                                                                       'test_ratio'],
                                                                                                   random_state=
                                                                                                   self.data_config[
                                                                                                       'random_seed'])
            p_vs_t = self.raw_matrix['PvsT']  # paper-term, bag of words, used for feature
            features = p_vs_t.toarray()  # torch.FloatTensor(p_vs_t.toarray())  # Feature
            # src_train = torch.LongTensor(src_train)
            # src_test = torch.LongTensor(src_test)
            # dst_train = torch.LongTensor(dst_train)
            # dst_test = torch.LongTensor(dst_test)
            # labels_train = torch.FloatTensor(labels_train)
            # labels_test = torch.FloatTensor(labels_test)
            if not os.path.exists(os.path.dirname(task_data_path)):
                os.mkdir(os.path.dirname(task_data_path))
            with open(task_data_path, 'wb') as f:
                pickle.dump([features, src_train, src_test, dst_train, dst_test, labels_train, labels_test], f)
        else:
            with open(task_data_path, 'rb') as f:
                features, src_train, src_test, dst_train, dst_test, labels_train, labels_test = pickle.load(f)
        return features, src_train, src_test, dst_train, dst_test, labels_train, labels_test


class DBLPDataLoader(GraphDataLoader):
    def __init__(self, data_config, remote_self_loop):
        super(DBLPDataLoader, self).__init__(data_config, remote_self_loop)

        self.heter_graph, self.raw_matrix = self.load_raw_matrix()

    def load_raw_matrix(self):
        data = sio.loadmat(self.data_path)
        '''
        ['__header__', '__version__', '__globals__', 'p_vs_a', 'p_vs_c', 'p_vs_t']
        p: Paper
        a：Author
        c: Conference
        t: Term
        '''
        p_vs_a = data['p_vs_a']  # paper-author
        p_vs_c = data['p_vs_c']  # paper-conference
        p_vs_t = data['p_vs_t']  # paper-term

        hg = dgl.heterograph({
            ('p', 'pa', 'a'): p_vs_a,
            ('a', 'ap', 'p'): p_vs_a.transpose(),
            ('p', 'pc', 'c'): p_vs_c,
            ('c', 'cp', 'p'): p_vs_c.transpose(),
            ('p', 'pt', 't'): p_vs_t,
            ('t', 'tp', 'p'): p_vs_t.transpose(),
        })

        return hg, data

    def load_classification_data(self):
        task_path = os.path.join(self.data_config['data_path'], self.data_config['dataset'], 'CF')
        if not os.path.exists(task_path):
            os.mkdir(task_path)
        node_type = 'author'
        if self.data_config['primary_type'] == 'a':
            node_type = "author"
        elif self.data_config['primary_type'] == 'p':
            node_type = "paper"
        elif self.data_config['primary_type'] == 'c':
            node_type = "conf"
        task_data_path = os.path.join(self.data_config['data_path'], self.data_config['dataset'], 'CF',
                                      'data_test_{}_{}.pkl'.format(node_type, self.data_config['test_ratio']))
        if self.data_config['resample'] or not os.path.exists(task_data_path):
            data_path = os.path.dirname(self.data_path)
            author_idx_map_path = os.path.join(data_path, "{}.txt".format(node_type))
            author_idx_map = {}
            if node_type == "paper":
                f = open(author_idx_map_path, encoding="gbk")
            else:
                f = open(author_idx_map_path)
            for i, l in enumerate(f.readlines()):
                l = l.replace("\n", "")
                idx, item = l.split("\t")
                if item not in author_idx_map:
                    author_idx_map[int(idx)] = i
            f.close()
            author_label_path = os.path.join(data_path, "{}_label.txt".format(node_type))
            num_classes = 4
            data = []
            label_dict = {}  # DBLP did not label all the author
            with open(author_label_path) as f:
                for l in f.readlines():
                    l = l.replace("\n", "").strip("\t")
                    author, label, name = l.split("\t")
                    data.append(author_idx_map[int(author)])
                    label_dict[author_idx_map[int(author)]] = int(label)
            data = np.array(data, dtype=np.int32)
            labels = np.full(np.max(data) + 1, -1)
            for idx, label in label_dict.items():
                labels[idx] = label

            labels = np.array(labels)
            train_idx, test_idx = train_test_split(data, test_size=self.data_config['test_ratio'],
                                                   random_state=self.data_config['random_seed'])
            # train_idx = torch.LongTensor(train_idx)
            # test_idx = torch.LongTensor(test_idx)
            features = self.raw_matrix['a_feature'].toarray()
            with open(task_data_path, 'wb') as f:
                pickle.dump([features, labels, num_classes, train_idx, test_idx], f)

        else:
            with open(task_data_path, 'rb') as f:
                features, labels, num_classes, train_idx, test_idx = pickle.load(f)

        return features, labels, num_classes, train_idx, test_idx

    def load_links_prediction_data(self):
        task_path = os.path.join(self.data_config['data_path'], self.data_config['dataset'], 'LP')
        if not os.path.exists(task_path):
            os.mkdir(task_path)
        task_data_path = os.path.join(self.data_config['data_path'], self.data_config['dataset'], 'LP',
                                      'data_test_{}.pkl'.format(self.data_config['test_ratio']))
        if self.data_config['resample'] or not os.path.exists(task_data_path):
            meta_path_graph = dgl.transform.metapath_reachable_graph(self.heter_graph, ["ap", "pa"])  # APA
            g = dgl.DGLGraph(meta_path_graph.to_networkx(), readonly=True)
            edgesampler = dgl.contrib.sampling.EdgeSampler(g, batch_size=g.number_of_edges(), num_workers=8,
                                                           negative_mode='tail',
                                                           neg_sample_size=1, exclude_positive=True)
            src_data = []
            dst_data = []
            labels = []
            for pos_g, neg_g in edgesampler:
                pos_edges = pos_g.edges()
                neg_edges = neg_g.edges()
                src_data.extend(pos_edges[0].tolist())
                dst_data.extend(pos_edges[1].tolist())
                labels.extend([1] * len(pos_edges[0]))

                src_data.extend(neg_edges[0].tolist())
                dst_data.extend(neg_edges[1].tolist())
                labels.extend([0] * len(neg_edges[0]))

            src_train, src_test, dst_train, dst_test, labels_train, labels_test = train_test_split(src_data, dst_data,
                                                                                                   labels,
                                                                                                   test_size=
                                                                                                   self.data_config[
                                                                                                       'test_ratio'],
                                                                                                   random_state=
                                                                                                   self.data_config[
                                                                                                       'random_seed'])
            features = self.raw_matrix['a_feature'].toarray()
            if not os.path.exists(os.path.dirname(task_data_path)):
                os.mkdir(os.path.dirname(task_data_path))
            with open(task_data_path, 'wb') as f:
                pickle.dump([features, src_train, src_test, dst_train, dst_test, labels_train, labels_test], f)
        else:
            with open(task_data_path, 'rb') as f:
                features, src_train, src_test, dst_train, dst_test, labels_train, labels_test = pickle.load(f)
        return features, src_train, src_test, dst_train, dst_test, labels_train, labels_test


class IMDBDataLoader(GraphDataLoader):
    def __init__(self, data_config, remote_self_loop):
        super(IMDBDataLoader, self).__init__(data_config, remote_self_loop)

        self.heter_graph, self.raw_matrix = self.load_raw_matrix()

    def load_raw_matrix(self):
        data = sio.loadmat(self.data_path)
        '''
        ['__header__', '__version__', '__globals__', 'm_feature', 'm_vs_a', 'm_vs_d', 'm_vs_k']
        p: Movie
        a：Actor
        d: Director
        k: Keyword
        '''
        m_vs_a = data['m_vs_a']  # paper-author
        m_vs_d = data['m_vs_d']  # paper-conference
        m_vs_k = data['m_vs_k']  # paper-term

        hg = dgl.heterograph({
            ('m', 'ma', 'a'): m_vs_a,
            ('a', 'am', 'm'): m_vs_a.transpose(),
            ('m', 'md', 'd'): m_vs_d,
            ('d', 'dm', 'm'): m_vs_d.transpose(),
            ('m', 'mk', 'k'): m_vs_k,
            ('k', 'km', 'm'): m_vs_k.transpose(),
        })

        return hg, data

    def load_classification_data(self):
        task_path = os.path.join(self.data_config['data_path'], self.data_config['dataset'], 'CF')
        if not os.path.exists(task_path):
            os.mkdir(task_path)
        task_data_path = os.path.join(self.data_config['data_path'], self.data_config['dataset'], 'CF',
                                      'data_test_{}.pkl'.format(self.data_config['test_ratio']))
        if self.data_config['resample'] or not os.path.exists(task_data_path):
            data_path = os.path.dirname(self.data_path)
            movie_label_path = os.path.join(data_path, 'index_label.txt')
            num_classes = 3
            data = []
            labels = []
            with open(movie_label_path) as f:
                for l in f.readlines():
                    l = l.replace("\n", "")
                    idx, label = l.split(",")
                    data.append(int(idx))
                    labels.append(int(label) - 1)  # 0, 1, 2
            data = np.array(data)
            labels = np.array(labels)
            train_idx, test_idx = train_test_split(data, test_size=self.data_config['test_ratio'],
                                                   random_state=self.data_config['random_seed'])
            # train_idx = torch.LongTensor(train_idx)
            # test_idx = torch.LongTensor(test_idx)
            features = self.raw_matrix['m_feature'].toarray()
            with open(task_data_path, 'wb') as f:
                pickle.dump([features, labels, num_classes, train_idx, test_idx], f)

        else:
            with open(task_data_path, 'rb') as f:
                features, labels, num_classes, train_idx, test_idx = pickle.load(f)

        return features, labels, num_classes, train_idx, test_idx

    def load_links_prediction_data(self):
        task_path = os.path.join(self.data_config['data_path'], self.data_config['dataset'], 'LP')
        if not os.path.exists(task_path):
            os.mkdir(task_path)
        task_data_path = os.path.join(self.data_config['data_path'], self.data_config['dataset'], 'LP',
                                      'data_test_{}.pkl'.format(self.data_config['test_ratio']))
        if self.data_config['resample'] or not os.path.exists(task_data_path):
            meta_path_graph = dgl.transform.metapath_reachable_graph(self.heter_graph, ["mk", "km"])  # MKM
            g = dgl.DGLGraph(meta_path_graph.to_networkx(), readonly=True)
            edgesampler = dgl.contrib.sampling.EdgeSampler(g, batch_size=g.number_of_edges(), num_workers=8,
                                                           negative_mode='tail',
                                                           neg_sample_size=1, exclude_positive=True)
            src_data = []
            dst_data = []
            labels = []
            for pos_g, neg_g in edgesampler:
                pos_edges = pos_g.edges()
                neg_edges = neg_g.edges()
                src_data.extend(pos_edges[0].tolist())
                dst_data.extend(pos_edges[1].tolist())
                labels.extend([1] * len(pos_edges[0]))

                src_data.extend(neg_edges[0].tolist())
                dst_data.extend(neg_edges[1].tolist())
                labels.extend([0] * len(neg_edges[0]))

            src_train, src_test, dst_train, dst_test, labels_train, labels_test = train_test_split(src_data, dst_data,
                                                                                                   labels,
                                                                                                   test_size=
                                                                                                   self.data_config[
                                                                                                       'test_ratio'],
                                                                                                   random_state=
                                                                                                   self.data_config[
                                                                                                       'random_seed'])
            features = self.raw_matrix['m_feature'].toarray()
            if not os.path.exists(os.path.dirname(task_data_path)):
                os.mkdir(os.path.dirname(task_data_path))
            with open(task_data_path, 'wb') as f:
                pickle.dump([features, src_train, src_test, dst_train, dst_test, labels_train, labels_test], f)
        else:
            with open(task_data_path, 'rb') as f:
                features, src_train, src_test, dst_train, dst_test, labels_train, labels_test = pickle.load(f)
        return features, src_train, src_test, dst_train, dst_test, labels_train, labels_test


class RDFDataLoader(GraphDataLoader):
    def __init__(self, data_config, remote_self_loop):
        super(RDFDataLoader, self).__init__(data_config, remote_self_loop)
        self.data_name = data_config['dataset']
        try:
            self.ignore_edge_type = data_config['ignore_edge_type']
        except:
            self.ignore_edge_type = False
        self.heter_graph, self.raw_matrix = self.load_raw_matrix()

    def load_raw_matrix(self):
        data = sio.loadmat(self.data_path)
        info_path = os.path.join(self.base_data_path, '{}_info.json'.format(self.data_name))
        with open(info_path, "r") as f:
            info = json.load(f)
        self.num_classes = info['num_classes']
        self.predict_category = info['predict_category']
        self.x_idx = np.array(info['train_idx'] + info['test_idx'])
        self.labels = np.array(info['labels'])
        edge_list = info['edge_list']

        hg_dict = {}
        for edge in edge_list:
            srctype, etype, dsttype = edge.split("||")
            if self.ignore_edge_type:
                if (srctype, srctype + '_vs_' + dsttype, dsttype) in hg_dict:
                    hg_dict[(srctype, srctype + '_vs_' + dsttype, dsttype)] += data[edge]
                else:
                    hg_dict[(srctype, srctype + '_vs_' + dsttype, dsttype)] = data[edge]
            else:
                hg_dict[(srctype, edge, dsttype)] = data[edge]
        hg = dgl.heterograph(hg_dict)

        return hg, data

    def load_classification_data(self):
        task_path = os.path.join(self.base_data_path, 'CF')
        if not os.path.exists(task_path):
            os.mkdir(task_path)
        task_data_path = os.path.join(task_path, 'data_test_{}.pkl'.format(self.data_config['test_ratio']))
        if self.data_config['resample'] or not os.path.exists(task_data_path):
            train_idx, test_idx = train_test_split(self.x_idx, test_size=self.data_config['test_ratio'],
                                                   random_state=self.data_config['random_seed'])
            with open(task_data_path, 'wb') as f:
                pickle.dump([None, self.labels, self.num_classes, train_idx, test_idx], f)

        else:
            with open(task_data_path, 'rb') as f:
                features, labels, num_classes, train_idx, test_idx = pickle.load(f)
        return None, self.labels, self.num_classes, train_idx, test_idx

    def load_links_prediction_data(self):
        return None


def load_data(data_config, remove_self_loop=False):
    dataset = data_config['dataset']
    if dataset == 'ACM':
        return ACMDataLoader(data_config, remove_self_loop)
    if dataset == 'DBLP':
        return DBLPDataLoader(data_config, remove_self_loop)
    if dataset == 'IMDB':
        return IMDBDataLoader(data_config, remove_self_loop)
    if dataset == 'AIFB' or dataset == 'BGS' or dataset == 'MUTAG' or dataset == 'AM':
        return RDFDataLoader(data_config, remove_self_loop)
    else:
        raise NotImplementedError('Unsupported dataset {}'.format(dataset))
