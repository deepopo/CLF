# -*- coding: utf-8 -*-
'''
Code: Deepopo

Paper:
Zhang J , Yu P S . Integrated anchor and social link predictions across social networks[C]
International Conference on Artificial Intelligence. AAAI Press, 2015.
'''
import networkx as nx
import numpy as np

def CLF(G1, G2, alignment_dict, alignment_dict_reversed, alpha1, alpha2, c, align_train_prop):
    G1_nodes = list(G1.nodes())
    G2_nodes = list(G2.nodes())
    np.random.seed(0)
    seed_list1 = list(np.random.choice(list(alignment_dict.keys()), int(align_train_prop * len(alignment_dict)), replace = False))
    seed_list2 = [alignment_dict[seed_list1[x]] for x in range(len(seed_list1))]
    
    G1_dict = {}
    for key, value in enumerate(G1.nodes()):
        G1_dict[value] = key
    G2_dict = {}
    for key, value in enumerate(G2.nodes()):
        G2_dict[value] = key
        
    W1 = nx.adj_matrix(G1)
    W1 /= np.sum(W1, axis=1)
    W2 = nx.adj_matrix(G2)
    W2 /= np.sum(W2, axis=1)
    W12 = np.zeros([G1.number_of_nodes(), G2.number_of_nodes()], dtype='float')
    for i in range(G1.number_of_nodes()):
        node = G1_nodes[i]
        if node in alignment_dict.keys() and node in seed_list1:
            W12[i] = W2[G2_dict[alignment_dict[node]]]
    W21 = np.zeros([G2.number_of_nodes(), G1.number_of_nodes()], dtype='float')
    for i in range(G2.number_of_nodes()):
        node = G2_nodes[i]
        if node in alignment_dict_reversed.keys() and node in seed_list2:
            W21[i] = W1[G1_dict[alignment_dict_reversed[node]]]
            
    W_part1 = np.concatenate([alpha2 * W2, (1 - alpha1) * W12])
    W_part2 = np.concatenate([(1 - alpha2) * W21, alpha1 * W1])
    W = np.concatenate([W_part1, W_part2], axis = 1)

    L1 = np.eye(G2.number_of_nodes())
    L2 = np.zeros([G1.number_of_nodes(), G2.number_of_nodes()])
    L = np.concatenate([L1, L2]).transpose()
    p = c * np.dot(L, np.linalg.inv(np.eye(len(W)) - (1 - c) * W))
    p = np.array(p)
    social_link = p[:, :G2.number_of_nodes()]
    anchor_link = p[:, G2.number_of_nodes():]
    return social_link, anchor_link, seed_list1, seed_list2

