import sys
import datetime
import copy
from pathlib import Path
import numpy as np, scipy as sp, networkx as nx
import math, time, os, sys, random
from collections import deque
import pickle
import itertools


# import scipy.sparse as sps
# from scipy.sparse import coo_matrix
# from scipy.sparse.linalg import svds, eigs
# import sparsesvd
# import scipy.spatial.distance as distance

# import sklearn
# from sklearn.model_selection import KFold
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import validation_curve
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.svm import SVC
# from sklearn import svm
# from sklearn import preprocessing
# from sklearn.cluster import KMeans
# from sklearn.decomposition import NMF, DictionaryLearning
# from sklearn.manifold import TSNE
# from sklearn.preprocessing import normalize
# from sklearn.feature_extraction import FeatureHasher

import collections
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim



# for train/valid/test input data.
def load_raw(input_file_path, delimiter):
	'''
	Format: 1st line: num. 
	The following lines: <src, dst, rel> (id)
	'''
	
	triple_num = 0
	triple_list = []
	triple_dict = {}


	fIn = open(input_file_path, 'r')

	line_1st = fIn.readline()
	triple_num = int(line_1st.strip())

	# TODO: change the order to the final version
	for line in fIn.readlines():
		parts = line.strip().split(delimiter)
		h = int(parts[0])
		t = int(parts[1])
		r = int(parts[2])

		triple_list.append( (h, t, r) )

	fIn.close()

	for triple in triple_list:
		triple_dict[triple] = True

	return triple_num, triple_list, triple_dict


def load_raw_number(input_file_path):

	fIn = open(input_file_path, 'r')
	num = int(fIn.readline())
	fIn.close()

	return num


##################################################################

def get_batch_list(triple_list, batch_size, batch_num):
	batch_list = [0] * batch_num

	for i in range(batch_num - 1):
		batch_list[i] = triple_list[i*batch_size : (i+1)*batch_size]

	batch_list[-1] = triple_list[(batch_num-1)*batch_size : ]

	return batch_list


##################################################################

# corrupt the triple by replacing its head
def corrupt_head_raw(triple, entity_num):
	
	# triple_corr = copy.deepcopy(triple)
	h_orig, t_orig, r_orig = copy.deepcopy(triple)

	# candidate_list = [ele for ele in range(entity_num)]
	# candidate_list.remove(h_orig)
	# h_new = np.random.choice(candidate_list)

	h_new = -1

	while True:
		h_new = random.randrange(entity_num)
		if h_new != h_orig:
			break
	
	triple_corr = (h_new, t_orig, r_orig)

	return triple_corr

# corrupt the triple by replacing its tail
def corrupt_tail_raw(triple, entity_num):
	
	# triple_corr = copy.deepcopy(triple)
	h_orig, t_orig, r_orig = copy.deepcopy(triple)

	# candidate_list = [ele for ele in range(entity_num)]
	# candidate_list.remove(t_orig)
	# t_new = np.random.choice(candidate_list)

	t_new = -1

	while True:
		t_new = random.randrange(entity_num)
		if t_new != t_orig:
			break

	triple_corr = (h_orig, t_new, r_orig)

	return triple_corr


# corrupt the triple by replacing its head, so that the triple is not in the dict
def corrupt_head_filter(triple, entity_num, triple_dict):


	h_orig, t_orig, r_orig = copy.deepcopy(triple)
	h_new = -1

	while True:
		h_new = random.randrange(entity_num)
		if (h_new, t_orig, r_orig) not in triple_dict:
			break

	# print(h_new)

	return (h_new, t_orig, r_orig)

# corrupt the triple by replacing its tail, so that the triple is not in the dict
def corrupt_tail_filter(triple, entity_num, triple_dict):

	h_orig, t_orig, r_orig = copy.deepcopy(triple)
	t_new = -1

	while True:
		t_new = random.randrange(entity_num)
		if (h_orig, t_new, r_orig) not in triple_dict:
			break

	return (h_orig, t_new, r_orig)

def get_elements(triple_list):

	h_list = [ele[0] for ele in triple_list]
	t_list = [ele[1] for ele in triple_list]
	r_list = [ele[2] for ele in triple_list]

	return h_list, t_list, r_list


def get_batch_idx_filter(triple_list, entity_num, triple_dict):
	
	# print('0.0')
	# print(len(triple_list))
	corr_triple_list = [ corrupt_head_filter(triple, entity_num, triple_dict) if random.random() < 0.5 
	else corrupt_tail_filter(triple, entity_num, triple_dict) for triple in triple_list ]

	h_p_list, t_p_list, r_p_list = get_elements(triple_list)
	h_n_list, t_n_list, r_n_list = get_elements(corr_triple_list)

	return h_p_list, t_p_list, r_p_list, h_n_list, t_n_list, r_n_list


def get_batch_idx_raw(triple_list, entity_num):

	corr_triple_list = [ corrupt_head_raw(triple, entity_num) if random.random() < 0.5 
	else corrupt_tail_raw(triple, entity_num) for triple in triple_list ]

	h_p_list, t_p_list, r_p_list = get_elements(triple_list)
	h_n_list, t_n_list, r_n_list = get_elements(corr_triple_list)

	return h_p_list, t_p_list, r_p_list, h_n_list, t_n_list, r_n_list



##################################################################

def dict_file_to_embedding(input_file_path, entity_num, USE_CUDA):
	
	floatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

	'''
	Input the pkl file
	'''
	fIn = open(input_file_path, 'rb')
	external_entity_embedding_dict = pickle.load(fIn)
	fIn.close()

	nid, emb = list(external_entity_embedding_dict.items())[0]
	emb_dim = len(emb)

	external_emb_t = floatTensor(entity_num, emb_dim).zero_()
	for idx in range(entity_num):

		if idx in external_entity_embedding_dict:
			external_emb_t.data[idx] = floatTensor(external_entity_embedding_dict[idx])

	external_emb_init_norm = F.normalize(external_emb_t, p=2, dim=1)

	external_emb = nn.Embedding(entity_num, emb_dim)
	external_emb.weight.data = external_emb_init_norm
	external_emb.weight.requires_grad = False	

	# print(external_emb(torch.tensor([0])))
	# sys.exit('check')
	# print(external_entity_embedding_dict[0])

	return external_emb

	












