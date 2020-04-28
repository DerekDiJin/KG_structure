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


def find_delimiter(input_file_path):
	delimiter = " "
	if ".csv" in input_file_path:
		delimiter = ","
	elif ".tsv" in input_file_path:
		delimiter = "\t"

	# majority voting
	elif ".txt" in input_file_path:

		del_dict = {' ': 0, '\t': 0, ',': 0}
		overall_str = ''

		fIn = open(input_file_path, 'r')
		for line in fIn.readlines()[:10]:
			overall_str += line
		fIn.close()


		for c in del_dict:
			del_dict[c] = overall_str.count(c)

		tmp = [k for k, v in del_dict.items() if v == max(del_dict.values())]
		if len(tmp) != 1:
			sys.exit('Multiple delimiters detected.')

		delimiter = tmp[0]

	else:
		sys.exit('Format not supported.')

	return delimiter



def load_embeddings(input_file_path):
	fIn = open(input_file_path, 'r')
	node_num, size = [int(x) for x in fIn.readline().strip().split()]
	result = {}
	while 1:
		l = fIn.readline()
		if l == '':
			break
		vec = l.strip().split(' ')
		assert len(vec) == size+1
		result[int(float(vec[0]))] = [float(x) for x in vec[1:]]
	fIn.close()
	assert len(result) == node_num
	return result



def write_embedding(embedding_dict, output_file_path, N, K, T):

	fOut = open(output_file_path, 'w')
	fOut.write(str(N) + ' ' + str(K*T) + '\n')

	for i in range(N):
		cur_line = ' '.join([str(ii) for ii in embedding_dict[i]])
		fOut.write(str(i) + ' ' + cur_line + '\n')

	fOut.close()

	return







