import numpy as np, networkx as nx
import scipy.sparse

class RepMethod():
	def __init__(self, bucket_max_value = None, method="hetero", num_buckets = None, use_other_features = False, operators = None,
		use_total = 0, implicit_factorization = True):
		self.method = method
		self.bucket_max_value = bucket_max_value
		self.num_buckets = num_buckets
		self.use_other_features = use_other_features
		self.operators = operators
		self.use_total = use_total


class Graph():
	def __init__(self, adj_matrix = None, num_nodes = None, max_id = None, directed = False, neighbor_list = None,
			num_buckets = None, base_features = None, cat_dict = None, id_cat_dict = None, unique_cat = None, check_eq = True):
		# self.nx_graph = nx_graph
		self.adj_matrix = adj_matrix
		self.num_nodes = num_nodes
		self.max_id = max_id
		self.base_features = base_features
		self.unique_cat = unique_cat
		self.directed = directed
		self.num_buckets = num_buckets

		self. neighbor_list = neighbor_list
		self.cat_dict = cat_dict
		self.id_cat_dict = id_cat_dict
		self.check_eq = check_eq
		

def get_delimiter(input_file_path):
	delimiter = " "
	if ".csv" in input_file_path:
		delimiter = ","
	elif ".tsv" in input_file_path:
		delimiter = "\t"
	else:
		sys.exit('Format not supported.')

	return delimiter


def write_embedding(rep, output_file_path):
	N, K = rep.shape

	fOut = open(output_file_path, 'w')
	fOut.write(str(N) + ' ' + str(K) + '\n')

	for i in range(N):
		cur_line = ' '.join([str(np.round(ii, 6)) for ii in rep[i,:]])
		fOut.write(str(i) + ' ' + cur_line + '\n')

	fOut.close()

	return
