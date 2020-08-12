import argparse, logging
import numpy as np
from . import struc2vec
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from time import time
from . import graph
from semb.methods import BaseMethod

"""
Driver for learning Struc2Vec embedding
"""
class Method(BaseMethod):

	__PARAMS__ = dict(dim=128, walk_length=80, num_walks=10, window_size=10, util_layer=None, \
					  iter=5, worker=1, weighted=False, directed=False, opt1=False, opt2=False, opt3=False)


	def exec_struc2vec(self):
		# execute struc2vec
		if self.params['opt3']:
			until_layer = self.params['until_layer']
		else:
			until_layer = None
		G = struc2vec.Graph(self.graph, self.params['directed'],
		                    self.params['worker'], untilLayer=until_layer)
		if self.params['opt1']:
			G.preprocess_neighbors_with_bfs_compact()
		else:
			G.preprocess_neighbors_with_bfs()
		if self.params['opt2']:
			G.create_vectors()
			G.calc_distances(compactDegree=self.params['opt1'])
		else:
			G.calc_distances_all_vertices(compactDegree=self.params['opt1'])

		G.create_distances_network()
		G.preprocess_parameters_random_walk()
		G.simulate_walks(self.params['num_walks'], self.params['walk_length'])

	def learn_embeddings(self):
		walks = LineSentence('random_walks.txt')
		self.model = Word2Vec(walks, size=self.params['dim'], window=self.params['window_size'],
	                	 min_count=0, hs=1, sg=1, workers=self.params['worker'], iter=self.params['iter'])
		self.embeddings = self.model.wv

	def get_id(self):
		return "struc2vec"

	def train(self):
		self.exec_struc2vec()
		self.learn_embeddings()
