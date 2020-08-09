import argparse
import numpy as np
import networkx as nx
from . import node2vec
from gensim.models import Word2Vec
from ..method import BaseMethod

"""
Driver for learning Node2Vec embedding
"""


class Method(BaseMethod):

    __PARAMS__ = dict(dim=128, walk_len=80, num_walks=10, window_size=10,
                      iter=1, num_workers=1, p=1, q=1, weighted=False, directed=False)

    def getId(self):
        return "node2vec"

    def train(self):
        # construct and learn
        g = node2vec(
            self.graph, self.params['directed'], self.params['p'], self.params['q'])
        g.preprocess_transition_probs()
        walks = g.simulate_walks(
            self.params['num_walks'], self.params['walk_length'])
        # generate embeddings
        walks = [map(str, walk) for walk in walks]
        model = Word2Vec(walks, size=self.params['dimensions'], window=self.params['window_size'],
                         min_count=0, sg=1, workers=self.params['workers'], iter=self.params['iter'])
        self.model = model
        self.embeddings = self.model.vw
