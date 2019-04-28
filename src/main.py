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
from time import time
from utils import *
from CLF import CLF
import argparse
def parse_args():
	'''
	Parses the CLF arguments.
	'''
	parser = argparse.ArgumentParser(description="Run CLF.")

	parser.add_argument('--data_folder', nargs='?', default='../graph/',
	                    help='Input graph path')
    
	parser.add_argument('--alignment_folder', nargs='?', default='../alignment/',
	                    help='Ground Truth path')
    
	parser.add_argument('--filename', nargs='?', default='DBLP',
	                    help='Name of file')
    
	parser.add_argument('--align_train_prop', type=float, default=0.2,
	                    help='Training rate')
    
	parser.add_argument('--alpha1', type=float, default=0.6,
	                    help='The weights of information within Graph 1')
    
	parser.add_argument('--alpha2', type=float, default=0.6,
	                    help='The weights of information within Graph 2')
    
	parser.add_argument('--c', type=float, default=0.1,
	                    help='The probability of returning to the starting point')
	
	return parser.parse_args()

def main(args):
    alignment = pd.read_csv(args.alignment_folder + args.filename + '.alignment', header = None)
    alignment_dict = {}
    alignment_dict_reversed = {}
    for i in range(len(alignment)):
        alignment_dict[alignment.iloc[i, 0]] = alignment.iloc[i, 1]
        alignment_dict_reversed[alignment.iloc[i, 1]] = alignment.iloc[i, 0]
    
    G1, G2 = loadG(args.data_folder, args.filename)
    G1_nodes = list(G1.nodes())
    G2_nodes = list(G2.nodes())
    
    W1 = np.array(nx.adjacency_matrix(G1).todense())
    W2 = np.array(nx.adjacency_matrix(G2).todense())
    
    t0 = time()
    '''
    S1: Similarity matrix of social links of G2
    S2: Similarity matrix of anchor links from G2 to G1
    Note that S2.shape = (G2.number_of_nodes(), G1.number_of_nodes())
    '''
    S1, S2, seed_list1, seed_list2 = CLF(G1, G2, alignment_dict, alignment_dict_reversed,
                                         args.alpha1, args.alpha2, args.c, args.align_train_prop)
    print('file: %s'%args.filename)
    print('Training rate: %.1f'%args.align_train_prop)
    print('Time cost: %.2fs'%(time()-t0))
    k = 30
    print('Top%d accuracy: %.2f%%'%(k, topk_accuracy(S2, G1, G2, alignment_dict_reversed, k)))
    print('AUC: %.2f%%'%(AUC(S2, G1, G2)))
    
if __name__ == '__main__':
    args = parse_args()
    main(args)

