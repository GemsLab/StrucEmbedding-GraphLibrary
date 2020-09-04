import argparse, logging
import numpy as np
from . import struc2vec
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from time import time
from . import graph
from .utils import *
from .graph import *
from semb.methods import BaseMethod
from .utils import *
from .graph import *

"""
Driver for learning Struc2Vec embedding
"""
class Method(BaseMethod):

    __PARAMS__ = dict(dim=128, walk_length=80, num_walks=10, window_size=10, until_layer=None, \
                      iter=5, workers=1, weighted=False, directed=False, opt1=False, opt2=False, opt3=False)


    def exec_struc2vec(self):
        # execute struc2vec
        if self.params['opt3']:
            until_layer = self.params['until_layer']
        else:
            until_layer = None
        graph = from_networkx(self.graph, not self.params['directed'])
        G = struc2vec.Graph(graph, self.params['directed'], self.params['workers'], untilLayer=until_layer)
        
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
        walks = LineSentence(walk_fname)
        self.model = Word2Vec(walks, size=self.params['dim'], window=self.params['window_size'],
                         min_count=0, hs=1, sg=1, workers=self.params['workers'], iter=self.params['iter'])

        self.embeddings = dict()
        for cur_node in self.graph.nodes():
            self.embeddings[cur_node] = self.model.wv.get_vector(str(cur_node)).tolist()

    def get_id(self):
        return "struc2vec"

    def train(self):
        self.exec_struc2vec()
        self.learn_embeddings()
