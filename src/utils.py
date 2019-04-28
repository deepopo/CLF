# -*- coding: utf-8 -*-
'''
Code: Deepopo

Paper:
Zhang J , Yu P S . Integrated anchor and social link predictions across social networks[C]
International Conference on Artificial Intelligence. AAAI Press, 2015.
'''
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def loadG(data_folder, filename):
    G1 = nx.Graph()
    G2 = nx.Graph()
    G1_edges = pd.read_csv(data_folder + filename + '1.edges', names = ['0', '1'])
    G1.add_edges_from(np.array(G1_edges))
    G2_edges = pd.read_csv(data_folder + filename + '2.edges', names = ['0', '1'])
    G2.add_edges_from(np.array(G2_edges))
    return G1, G2

def load_attribute(attribute_folder, filename, G1, G2):
    G1_nodes = list(G1.nodes())
    G2_nodes = list(G2.nodes())
    attribute1 = pd.read_csv(attribute_folder + filename + 'attr1.csv', header = None, index_col = 0)
    attribute2 = pd.read_csv(attribute_folder + filename + 'attr2.csv', header = None, index_col = 0)
    attribute1 = np.array(attribute1.loc[G1_nodes, :])
    attribute2 = np.array(attribute2.loc[G2_nodes, :])
    attr_cos = cosine_distance(attribute1, attribute2)
    return attr_cos, attribute1, attribute2

def cosine_distance(matrix1,matrix2):
    matrix1_matrix2 = np.dot(matrix1, matrix2.transpose())
    matrix1_norm = np.sqrt(np.multiply(matrix1, matrix1).sum(axis=1))
    matrix1_norm = matrix1_norm[:, np.newaxis]
    matrix2_norm = np.sqrt(np.multiply(matrix2, matrix2).sum(axis=1))
    matrix2_norm = matrix2_norm[:, np.newaxis]
    cosine_distance = np.divide(matrix1_matrix2, np.dot(matrix1_norm, matrix2_norm.transpose()))
    return cosine_distance

def topk_accuracy(S, G1, G2, alignment_dict_reversed, k):
    G2_nodes = list(G2.nodes())
    argsort = np.argsort(-S, axis = 1)
    G1_dict = {}
    for key, value in enumerate(list(G1.nodes())):
        G1_dict[value] = key
    G2_dict = {}
    for key, value in enumerate(list(G2.nodes())):
        G2_dict[value] = key
    L = []
    for i in range(len(argsort)):
        L.append(np.where(argsort[i, :] == G1_dict[alignment_dict_reversed[G2_nodes[i]]])[0][0] + 1)
    return np.sum(np.array(L) <= k) / len(L) * 100

def AUC(S, G1, G2):
    S = S.flatten()
    argsort = np.argsort(-S)
    n = np.min([G2.number_of_nodes(), G1.number_of_nodes()])
    thr = S[argsort[n - 1]]
    S[S>=thr] = 1
    S[S<thr] = 0
    auc_score = 100 * roc_auc_score(S,np.eye(n).flatten())
    return auc_score