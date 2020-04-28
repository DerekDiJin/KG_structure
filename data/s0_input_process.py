import numpy as np
import sys
import random


def write_dict(input_dict, output_file_path):

	input_dict_sorted = {k: v for k, v in sorted(input_dict.items(), key=lambda item: item[1])}

	fOut = open(output_file_path, 'w')

	for key in input_dict_sorted:
		new_line = '{}\t{}\n'.format(key, input_dict_sorted[key])
		fOut.write(new_line)

	fOut.close()

	return 

# The output format is (h, t, l)
def write_triple(input_list, output_file_path):

	fOut = open(output_file_path, 'w')

	for h, l, t in input_list:
		new_line = '{}\t{}\t{}\n'.format(h, t, l)
		fOut.write(new_line)

	fOut.close()

	return



def input_process(argv):
	input_file_path = argv[1]

	prefix = './'

	entity2id_file_path = prefix + 'entity2id.txt'
	relation2id_file_path = prefix + 'relation2id.txt'

	train_file_path = prefix + 'train.txt'
	test_file_path = prefix + 'test.txt'
	valid_file_path = prefix + 'valid.txt'


	entity_set = set([])
	relation_set = set([])

	triple_list = []

	fIn = open(input_file_path, 'r')

	for idx, line in enumerate(fIn.readlines()):
		# print idx

		parts = line.strip().split('\t')

		if len(parts) != 3:
			continue

		h = parts[0]
		l = parts[1]
		t = parts[2]

		entity_set.add(h)
		entity_set.add(t)
		relation_set.add(l)

		triple = (h, l, t)
		triple_list.append(triple)

	fIn.close()
	entity2id_dict = dict( zip(entity_set, range(len(entity_set))) )
	relation2id_dict = dict( zip(relation_set, range(len(relation_set))) )


	write_dict(entity2id_dict, entity2id_file_path)
	write_dict(relation2id_dict, relation2id_file_path)


	random.shuffle(triple_list)

	# train
	train_n = int(len(triple_list) * 0.8)

	# valid
	valid_n = int(len(triple_list) * 0.1)

	write_triple(triple_list[ : train_n], train_file_path)
	write_triple(triple_list[train_n : train_n + valid_n], valid_file_path)
	write_triple(triple_list[train_n + valid_n : ], test_file_path)


	return


if __name__ =="__main__":
	input_process(sys.argv)