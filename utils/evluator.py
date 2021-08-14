#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/8/10 23:25
# @Author  : Raymound luo
# @Mail    : luolinhao1998@gmail.com
# @File    : evluator.py
# @Software: PyCharm
# @Describe:
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, roc_auc_score, normalized_mutual_info_score, adjusted_rand_score, \
    silhouette_score, cluster
from sklearn.svm import SVC
import numpy as np
import os
import json


def KNN_train(x, y):
    knn = KNeighborsClassifier()
    knn.fit(x, y)


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


class Evaluator(object):
    def __init__(self, method, CF_data, LP_data, result_path='./result', random_state=123, max_iter=150,
                 n_jobs=1):
        self.method = method
        self.CF_data = CF_data
        self.LP_data = LP_data
        self.result_path = result_path
        if not os.path.exists(self.result_path):
            os.makedirs((self.result_path))
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.result = {}

    def get_model(self):
        if self.method == "KNN":
            model = KNeighborsClassifier()
        elif self.method == "LR":
            model = LogisticRegression(solver='lbfgs', random_state=self.random_state, max_iter=self.max_iter,
                                       n_jobs=self.n_jobs, multi_class='auto')
        elif self.method == "SVM":
            model = SVC()
        return model

    def evluate_CF(self, emb_feature):
        features, labels, num_classes, train_idx, test_idx = self.CF_data
        model = self.get_model()
        model.fit(emb_feature[train_idx], labels[train_idx])
        score = model.predict(emb_feature[test_idx])  #
        micro_f1 = f1_score(labels[test_idx], score, average='micro')
        macro_f1 = f1_score(labels[test_idx], score, average='macro')
        self.result['CF'] = {'Micro f1': micro_f1, 'Macro f1': macro_f1}
        print("Node classification result: ")
        print('Micro f1: ', micro_f1)
        print('Macro f1: ', macro_f1)

    def evluate_LP(self, emb_feature):
        features, src_train, src_test, dst_train, dst_test, labels_train, labels_test = self.LP_data
        train_edges_feature = self._concat_edges_feture(emb_feature, src_train, dst_train)
        test_edges_feature = self._concat_edges_feture(emb_feature, src_test, dst_test)
        model = self.get_model()
        model.fit(train_edges_feature, labels_train)
        score = model.predict(test_edges_feature)
        f1 = f1_score(labels_test, score)
        auc_score = roc_auc_score(labels_test, score)
        self.result['LP'] = {'AUC': auc_score, 'F1': f1}
        print("Link Prediction result: ")
        print('AUC: ', auc_score)
        print('F1: ', f1)

    def evluate_CL(self, emb_feature, time=10, test_only=False):
        features, labels, num_classes, train_idx, test_idx = self.CF_data
        if test_only:
            x_idx = test_idx
        else:
            x_idx = np.concatenate((train_idx, test_idx))
        x = emb_feature[x_idx]
        y = labels[x_idx]

        estimator = KMeans(n_clusters=num_classes)
        ARI_list = []  # adjusted_rand_score(
        NMI_list = []
        Purity_list = []
        silhouette_score_list = []
        if time:
            for i in range(time):
                estimator.fit(x, y)
                y_pred = estimator.predict(x)
                score = normalized_mutual_info_score(y, y_pred, average_method='arithmetic')
                NMI_list.append(score)
                s2 = adjusted_rand_score(y, y_pred)
                ARI_list.append(s2)
                # silhouette_score
                labels = estimator.labels_
                s3 = silhouette_score(x, labels, metric='euclidean')
                silhouette_score_list.append(s3)
                s4 = purity_score(y, y_pred)
                Purity_list.append(s4)
            # print('NMI_list: {}'.format(NMI_list))
            score = sum(NMI_list) / len(NMI_list)
            s2 = sum(ARI_list) / len(ARI_list)
            s3 = sum(silhouette_score_list) / len(silhouette_score_list)
            s4 = sum(Purity_list) / len(Purity_list)
            print(
                'NMI (10 avg): {:.4f} , ARI (10avg): {:.4f}, Purity (10avg): {:.4f}, silhouette(10avg): {:.4f}'.format(
                    score, s2, s4, s3))
        else:
            estimator.fit(x, y)
            y_pred = estimator.predict(x)
            score = normalized_mutual_info_score(y, y_pred)
            print("NMI on all label data: {:.5f}".format(score))
        self.result['CL'] = {'NMI': score, 'ARI': s2, 'Purity': s4, 'silhouette': s3}

    def _concat_edges_feture(self, emb_feature, src_list, dst_list):
        src_feature = emb_feature[src_list]
        dst_feature = emb_feature[dst_list]
        edges_feature = src_feature * dst_feature
        # edges_feature = np.concatenate([src_feature, dst_feature], 1)
        return edges_feature

    def dump_result(self, p_emb, metric):
        dir_name = ''
        if 'CF' in metric:
            dir_name += "CF_{:.2f}_{:.2f}_".format(self.result['CF']['Micro f1'], self.result['CF']['Macro f1'])
        if 'LP' in metric:
            dir_name += "LP_{:.2f}_{:.2f}_".format(self.result['LP']['AUC'], self.result['LP']['F1'])
        if 'CL' in metric:
            dir_name += "CL_{:.2f}_{:.2f}_{:.2f}_{:.2f}".format(self.result['CL']['NMI'], self.result['CL']['ARI'],
                                                                self.result['CL']['Purity'],
                                                                self.result['CL']['silhouette'])
        model_path = os.path.join(self.result_path, dir_name)
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        with open(os.path.join(model_path, 'result.json'), 'w') as f:
            json.dump(self.result, f)
        np.save(os.path.join(model_path, 'p_emb.npy'), p_emb)
        print("Result save in {}".format(model_path))
        return model_path
