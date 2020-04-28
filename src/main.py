#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-04-20 15:15:03
# @Author  : Di Jin (dijin@umich.edu)
# @Link    : https://derekdijin.github.io/
# @Version : $Id$

import os,sys
import argparse
import numpy as np
import time

from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim

from util import *
import model
from model import *
from data import *

# from external_evaluation import *
from sklearn.metrics.pairwise import pairwise_distances

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


class Config(object):
	def __init__(self):
		#################################
		# configuration for models
		#################################
		self.entity_num = 0
		self.relation_num = 0

		self.entity_dim = 50
		self.relation_dim = 50

		self.train_times = 100

		self.batch_size = 0
		self.batch_num = 100
		self.margin = 1.0
		self.norm = 1.0 #'L1'
		self.filter_Flag = True
		self.K = 10

		#################################
		# configuratio for external embeddings
		#################################
		self.external_entity_embedding_path = './tmp.pkl'
		self.entity_dim_ext = 128


def parse_args():
	'''
	Parses the arguments.
	'''
	parser = argparse.ArgumentParser(description="<new_proj>")

	parser.add_argument('--input', nargs='?', default='../data/FB15k_N', help='Input data dir path')

	parser.add_argument('--output', nargs='?', default='./tmp.tsv', help='Output file path')

	return parser.parse_args()


def train_BP_status(epoch, loss_v, start_time):
	cur_time = time.time()
	print('[{}]\t{:.4f}'.format('Time(s) collapsed', cur_time-start_time))
	print('[{}]\t{:d}\t{:.4f}'.format('Toral Training loss at epoch', epoch, loss_v))


def argwhere_head(triple, array_idx, triple_dict):
	
	rank = 0
	cur_h, cur_t, cur_r = triple

	for idx in array_idx:
		if cur_h == idx:
			return rank
		elif (idx, cur_t, cur_r) in triple_dict:
			continue
		else:
			rank += 1

	return rank


def argwhere_tail(triple, array_idx, triple_dict):
	
	rank = 0
	cur_h, cur_t, cur_r = triple

	for idx in array_idx:
		if cur_t == idx:
			return rank
		elif (cur_h, idx, cur_r) in triple_dict:
			continue
		else:
			rank += 1

	return rank



def eval_KG(entity_embedding, relation_embedding, filter_Flag, triple_dict, triple_test, K):
	# print(h_test_t)
	# h_test_emb = model.entity_embedding(h_test_t)
	# t_test_emb = model.entity_embedding(t_test_t)
	# r_test_emb = model.relation_embedding(r_test_t)

	# target_dist = torch.norm(h_test_emb + r_test_emb - t_test_emb, p=model.norm).repeat(model.entity_num, 1)
	# cur_h_emb = h_test_emb.repeat(model.entity_num, 1)
	# cur_t_emb = t_test_emb.repeat(model.entity_num, 1)
	# cur_r_emb = r_test_emb.repeat(model.entity_num, 1)

	# cur_h_dist = torch.norm(model.entity_embedding.weight.data + cur_r_emb - cur_t_emb, p=model.norm, dim=1).view(-1, 1)
	# cur_t_dist = torch.norm(cur_h_emb + cur_r_emb - model.entity_embedding.weight.data, p=model.norm, dim=1).view(-1, 1)

	# rank_h = torch.nonzero(nn.functional.relu(target_dist - cur_h_dist))[:,0].size()[0]
	# rank_t = torch.nonzero(nn.functional.relu(target_dist - cur_t_dist))[:,0].size()[0]

	#################################
	h_test, t_test, r_test = triple_test

	# print(entity_embedding)
	# print(entity_embedding[0])
	h_test_emb = entity_embedding[h_test]
	t_test_emb = entity_embedding[t_test]
	r_test_emb = relation_embedding[r_test]

	#################################
	t_hat_emb = (h_test_emb + r_test_emb).reshape(1,-1)
	h_hat_emb = (t_test_emb - r_test_emb).reshape(1,-1)

	if model.norm == 1.0:
		dist_tail = pairwise_distances(t_hat_emb, entity_embedding, metric='manhattan')
		dist_head = pairwise_distances(h_hat_emb, entity_embedding, metric='manhattan')
	else:
		dist_tail = pairwise_distances(t_hat_emb, entity_embedding, metric='euclidean')
		dist_head = pairwise_distances(h_hat_emb, entity_embedding, metric='euclidean')

	rank_tail_sorted_idx = np.argsort(dist_tail, axis=1)[0]
	rank_head_sorted_idx = np.argsort(dist_head, axis=1)[0]


	if filter_Flag:
		rank_tail = argwhere_tail(triple_test, rank_tail_sorted_idx, triple_dict)
		rank_head = argwhere_head(triple_test, rank_head_sorted_idx, triple_dict)
	else:
		rank_tail = np.argwhere(t_test, rank_tail_sorted_idx)
		rank_head = np.argwhere(h_test, rank_head_sorted_idx)

	#################################

	# print(rank_tail, rank_head)
	# sys.exit('.')
	return rank_tail, rank_head

	# return (rank_h + rank_t + 2) / 2

def eval_KG_comb(entity_embedding, relation_embedding, entity_embedding_ext, filter_Flag, triple_dict, triple_test, K, weight_in, weight_ext):
	# print(h_test_t)
	# h_test_emb = model.entity_embedding(h_test_t)
	# t_test_emb = model.entity_embedding(t_test_t)
	# r_test_emb = model.relation_embedding(r_test_t)

	# target_dist = torch.norm(h_test_emb + r_test_emb - t_test_emb, p=model.norm).repeat(model.entity_num, 1)
	# cur_h_emb = h_test_emb.repeat(model.entity_num, 1)
	# cur_t_emb = t_test_emb.repeat(model.entity_num, 1)
	# cur_r_emb = r_test_emb.repeat(model.entity_num, 1)

	# cur_h_dist = torch.norm(model.entity_embedding.weight.data + cur_r_emb - cur_t_emb, p=model.norm, dim=1).view(-1, 1)
	# cur_t_dist = torch.norm(cur_h_emb + cur_r_emb - model.entity_embedding.weight.data, p=model.norm, dim=1).view(-1, 1)

	# rank_h = torch.nonzero(nn.functional.relu(target_dist - cur_h_dist))[:,0].size()[0]
	# rank_t = torch.nonzero(nn.functional.relu(target_dist - cur_t_dist))[:,0].size()[0]

	#################################
	h_test, t_test, r_test = triple_test

	entity_embedding = entity_embedding * weight_in + entity_embedding_ext * weight_ext

	# print(entity_embedding)
	# print(entity_embedding[0])
	h_test_emb = entity_embedding[h_test]
	t_test_emb = entity_embedding[t_test]
	r_test_emb = relation_embedding[r_test]

	# h_test_emb_ext = entity_embedding_ext[h_test]
	# t_test_emb_ext = entity_embedding_ext[t_test]
	# r_test_emb_ext = relation_embedding_ext[r_test]


	# h_test_emb = h_test_emb * 0.5 + h_test_emb_ext * 0.5
	# t_test_emb = t_test_emb * 0.5 + t_test_emb_ext * 0.5
	# r_test_emb = r_test_emb * 0.5 + r_test_emb_ext * 0.5

	#################################
	t_hat_emb = (h_test_emb + r_test_emb).reshape(1,-1)
	h_hat_emb = (t_test_emb - r_test_emb).reshape(1,-1)

	if model.norm == 1.0:
		dist_tail = pairwise_distances(t_hat_emb, entity_embedding, metric='manhattan')
		dist_head = pairwise_distances(h_hat_emb, entity_embedding, metric='manhattan')
	else:
		dist_tail = pairwise_distances(t_hat_emb, entity_embedding, metric='euclidean')
		dist_head = pairwise_distances(h_hat_emb, entity_embedding, metric='euclidean')

	rank_tail_sorted_idx = np.argsort(dist_tail, axis=1)[0]
	rank_head_sorted_idx = np.argsort(dist_head, axis=1)[0]


	if filter_Flag:
		rank_tail = argwhere_tail(triple_test, rank_tail_sorted_idx, triple_dict)
		rank_head = argwhere_head(triple_test, rank_head_sorted_idx, triple_dict)
	else:
		rank_tail = np.argwhere(t_test, rank_tail_sorted_idx)
		rank_head = np.argwhere(h_test, rank_head_sorted_idx)

	#################################

	# print(rank_tail, rank_head)
	# sys.exit('.')
	return rank_tail, rank_head


def print_model_param_info(model):
	print('[Model parameters to train]')
	for name, param in model.named_parameters():
		if param.requires_grad:
			print(name, param.data) # param.data

	# for param in model.parameters():
	# 	if param.requires_grad:
	# 		print(param.name, param.size())
	# 		print(param.data)


	return


if __name__ == '__main__':

	args = parse_args()

	torch.manual_seed(17)
	random.seed(17)
	
	input_data_dir_path = args.input
	output_file_path = args.output

	#################################
	input_train_file_path = os.path.join(input_data_dir_path, 'train2id.txt')
	input_valid_file_path = os.path.join(input_data_dir_path, 'valid2id.txt')
	input_test_file_path = os.path.join(input_data_dir_path, 'test2id.txt')
	input_complete_file_path = os.path.join(input_data_dir_path, 'triple2id.txt')
	input_entity_id_file_path = os.path.join(input_data_dir_path, 'entity2id.txt')
	input_relation_id_file_path = os.path.join(input_data_dir_path, 'relation2id.txt')

	delimiter = find_delimiter(input_train_file_path)

	train_num, train_list, train_dict = load_raw(input_train_file_path, delimiter)
	valid_num, valid_list, valid_dict = load_raw(input_valid_file_path, delimiter)
	test_num, test_list, test_dict = load_raw(input_test_file_path, delimiter)
	complete_num, complete_list, complete_dict = load_raw(input_complete_file_path, delimiter)

	#################################
	config = Config()

	config.entity_num = load_raw_number(input_entity_id_file_path)
	config.relation_num = load_raw_number(input_relation_id_file_path)

	config.train_times = config.train_times
	config.batch_size = train_num // config.batch_num


	##################################################################
	# Model initialization
	#################################
	model = model.TransE(config)
	criterion = nn.MarginRankingLoss(config.margin, False, reduction='sum').to(device)
	optimizer = optim.SGD(model.parameters(), lr = 1e-2)

	if USE_CUDA:
		model.cuda()
		criterion.cuda()

	print_model_param_info(model)

	#################################
	# Back propagation
	#################################
	start_time = time.time()
	train_batch_list = get_batch_list(train_list, config.batch_size, config.batch_num)

	for epoch in range(config.train_times):

		print('[{}]\t{:d}'.format('Current epoch', epoch))
		epoch_loss = floatTensor([0.0])
		random.shuffle(train_batch_list)

		for batch in train_batch_list:

			# print(len(batch))
			# print(len(complete_dict))

			model.zero_grad()

			# print('0')
			if config.filter_Flag:
				h_p_batch, t_p_batch, r_p_batch, h_n_batch, t_n_batch, r_n_batch = get_batch_idx_filter(batch, config.entity_num, complete_dict)
			else:
				h_p_batch, t_p_batch, r_p_batch, h_n_batch, t_n_batch, r_n_batch = get_batch_idx_raw(batch, config.entity_num)

			h_p_batch_t = longTensor(h_p_batch)#.to(device)
			t_p_batch_t = longTensor(t_p_batch)#.to(device)
			r_p_batch_t = longTensor(r_p_batch)#.to(device)
			h_n_batch_t = longTensor(h_n_batch)#.to(device)
			t_n_batch_t = longTensor(t_n_batch)#.to(device)
			r_n_batch_t = longTensor(r_n_batch)#.to(device)

			loss_p_t, loss_n_t = model(h_p_batch_t, t_p_batch_t, r_p_batch_t, h_n_batch_t, t_n_batch_t, r_n_batch_t)
			
			# loss_p_tt = loss_p_t.view(1,-1)
			# loss_n_tt = loss_n_t.view(1,-1)

			neg_one_t = floatTensor([-1])#.to(device)#.to(self.device)
			batch_loss = criterion(loss_p_t, loss_n_t, neg_one_t)

			batch_loss.backward()

			optimizer.step()
			epoch_loss += batch_loss

		model.entity_embedding.weight.data = F.normalize(model.entity_embedding.weight.data, p=2, dim=1)
		model.relation_embedding.weight.data = F.normalize(model.relation_embedding.weight.data, p=2, dim=1)

		if epoch % 10 == 0:
			train_BP_status(epoch, epoch_loss.item(), start_time)


	# print(model.parameters().weight)

	##################################################################
	# Testing
	##################################################################

	print('Start testing')
	print('[{}]\t{:d}'.format('Testing set number', test_num))

	entity_embedding_final = model.entity_embedding.weight.data.cpu().numpy()
	relation_embedding_final = model.relation_embedding.weight.data.cpu().numpy()

	entity_embedding_ext_tmp = model.external_entity_embedding.weight.data.cpu().numpy()
	weight = model.transmit.weight.data.cpu().numpy()
	bias = model.transmit.bias.data.cpu().numpy()

	weight_in = model.weight_in.data.cpu().numpy()
	weight_ext = model.weight_ext.data.cpu().numpy()
	# weight_in_s = 1/(1 + np.exp(-weight_in))
	# weight_ext_s = 1/(1 + np.exp(-weight_ext))

	print(entity_embedding_ext_tmp.shape)
	print(weight.shape)
	print(bias.shape)
	print('-----')
	print(weight_in)
	print(weight_ext)

	entity_embedding_ext_final = np.maximum( (np.matmul(entity_embedding_ext_tmp, weight.T) + bias.T), 0 )

	# hit10Test, meanrankTest = evaluation_transE(test_list, complete_dict, ent_embeddings, rel_embeddings, True, True, head=0)

	MR = 0
	Hit = 0
	# print(test_list)

	for idx, triple_test in enumerate(test_list):
		# print(h_test)

		if idx % 1000 == 0:
			print('[{}]\t{:d}'.format('Currently processing', idx))

		# h_test_t = longTensor([h_test])
		# t_test_t = longTensor([t_test])
		# r_test_t = longTensor([r_test])

		rank_tail, rank_head = eval_KG_comb(entity_embedding_final, relation_embedding_final, entity_embedding_ext_final, config.filter_Flag, complete_dict, triple_test, config.K, weight_in, weight_ext)
		MR += rank_tail + rank_head

		if rank_tail < config.K:
			Hit += 1.0
		if rank_head < config.K:
			Hit += 1.0
	
	print( '[{}]\t{:.4f}'.format('Mean Rank', MR/(2*test_num)) )
	print( '[{}{:d}]\t{:.4f}'.format('HITS@', config.K, Hit/(2*test_num)) )





