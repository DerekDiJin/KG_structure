import sys
import datetime
from pathlib import Path
import numpy as np, scipy as sp, networkx as nx
import math, time, os, sys, random
from collections import deque
import pickle
import itertools


import scipy.sparse as sps
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds, eigs
# import sparsesvd
import scipy.spatial.distance as distance

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
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, DictionaryLearning
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

from sklearn.feature_extraction import FeatureHasher
import collections
from collections import defaultdict


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim

from data import *

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
	longTensor = torch.cuda.LongTensor
	floatTensor = torch.cuda.FloatTensor
	print('[CUDA used. Processing on GPU.]')

	device = "cuda"

else:
	longTensor = torch.LongTensor
	floatTensor = torch.FloatTensor
	print('[CUDA not used. Processing on CPU.]')

	device = "cpu"



class TransE(nn.Module):
	def __init__(self, config):
		'''
		TransE module. 
		'''
		super(TransE, self).__init__()

		self.entity_num = config.entity_num
		self.entity_dim = config.entity_dim

		self.relation_num = config.relation_num
		self.relation_dim = config.relation_dim

		# self.dist_flag = config.dist_flag

		self.norm = config.norm

		self.entity_dim_ext = config.entity_dim_ext
		# self.relation_dim_ext = config.relation_dim_ext


		#################################
		# Embedding initialization
		#################################
		entity_weight_init = floatTensor(self.entity_num, self.entity_dim)#.to(device)
		relation_weight_init = floatTensor(self.relation_num, self.relation_dim)#.to(device)
		nn.init.xavier_uniform_(entity_weight_init)
		nn.init.xavier_uniform_(relation_weight_init)

		print(entity_weight_init.device)

		entity_weight_init_norm = F.normalize(entity_weight_init, p=2, dim=1)
		relation_weight_init_norm = F.normalize(relation_weight_init, p=2, dim=1)

		self.entity_embedding = nn.Embedding(self.entity_num, self.entity_dim)
		self.relation_embedding = nn.Embedding(self.relation_num, self.relation_dim)
		self.entity_embedding.weight = nn.Parameter(entity_weight_init_norm)
		self.relation_embedding.weight = nn.Parameter(relation_weight_init_norm)

		#################################
		# External embedding initialization
		#################################

		transmit_weight_init = floatTensor(self.entity_dim, self.entity_dim_ext)
		transmit_bias_init = floatTensor(self.entity_dim)
		nn.init.xavier_uniform_(transmit_weight_init)
		nn.init.uniform_(transmit_bias_init, a=0, b=1)

		transmit_weight_init_norm = F.normalize(transmit_weight_init, p=2, dim=1)
		transmit_bias_init_init_norm = F.normalize(transmit_bias_init, p=2, dim=0)
		# transmit_weight_init_norm_T = transmit_weight_init_norm.T

		self.transmit = torch.nn.Linear(self.entity_dim_ext, self.entity_dim)
		self.transmit.weight = nn.Parameter(transmit_weight_init_norm)
		self.transmit.bias = nn.Parameter(transmit_bias_init_init_norm)


		self.external_entity_embedding = dict_file_to_embedding(config.external_entity_embedding_path, self.entity_num, USE_CUDA)

		# self.weight_in = nn.Parameter(floatTensor([0.5]))
		# self.weight_ext = nn.Parameter(floatTensor([0.5]))
		
		self.weight_in = floatTensor([1.0])
		self.weight_ext = floatTensor([0.0])

	def forward(self, h_p, t_p, r_p, h_n, t_n, r_n):
		'''
		Inputs are tensors
		'''

		h_p_emb = self.entity_embedding(h_p)
		t_p_emb = self.entity_embedding(t_p)
		r_p_emb = self.relation_embedding(r_p)
		
		h_n_emb = self.entity_embedding(h_n)
		t_n_emb = self.entity_embedding(t_n)
		r_n_emb = self.relation_embedding(r_n)
		
		#################################
		# External embedding
		#################################

		h_p_emb_ext = self.external_entity_embedding(h_p)
		t_p_emb_ext = self.external_entity_embedding(t_p)
		# r_p_emb_ext = self.relation_embedding(r_p)
		h_n_emb_ext = self.external_entity_embedding(h_n)
		t_n_emb_ext = self.external_entity_embedding(t_n)
		# r_n_emb_ext = self.relation_embedding(r_n)

		# h_p_emb_trans = torch.tanh(self.transmit(h_p_emb_ext))#.clamp(min = 0)
		# t_p_emb_trans = torch.tanh(self.transmit(t_p_emb_ext))#.clamp(min = 0)
		# h_n_emb_trans = torch.tanh(self.transmit(h_n_emb_ext))#.clamp(min = 0)
		# t_n_emb_trans = torch.tanh(self.transmit(t_n_emb_ext))#.clamp(min = 0)
	
		h_p_emb_trans = self.transmit(h_p_emb_ext).clamp(min = 0)
		t_p_emb_trans = self.transmit(t_p_emb_ext).clamp(min = 0)
		h_n_emb_trans = self.transmit(h_n_emb_ext).clamp(min = 0)
		t_n_emb_trans = self.transmit(t_n_emb_ext).clamp(min = 0)

		#################################
		# Combine in-/external embedding
		#################################

		# print('-------------')
		# print(self.weight_ext)
		# print(h_p_emb_trans)

		h_p_emb_comb = (self.weight_in) * h_p_emb + (self.weight_ext) * h_p_emb_trans
		h_n_emb_comb = (self.weight_in) * h_n_emb + (self.weight_ext) * h_n_emb_trans
		t_p_emb_comb = (self.weight_in) * t_p_emb + (self.weight_ext) * t_p_emb_trans
		t_n_emb_comb = (self.weight_in) * t_n_emb + (self.weight_ext) * t_n_emb_trans

		
		# dis_p = torch.norm((h_p_emb + r_p_emb - t_p_emb), p=self.norm, dim=1)
		# dis_n = torch.norm((h_n_emb + r_n_emb - t_n_emb), p=self.norm, dim=1)

		dis_p = torch.norm((h_p_emb_comb + r_p_emb - t_p_emb_comb), p=self.norm, dim=1)
		dis_n = torch.norm((h_n_emb_comb + r_n_emb - t_n_emb_comb), p=self.norm, dim=1)
		

		return dis_p, dis_n









