# -*- coding: utf-8 -*-
"""
Reference implementation of RiWalk.
Author: Xuewei Ma
For more details, refer to the paper:
RiWalk: Fast Structural Node Embedding via Role Identification
ICDM, 2019
"""

import argparse
from . import RiWalkGraph
from .utils import WALK_FILES_DIR
from gensim.models import Word2Vec
from gensim.models.keyedvectors import Word2VecKeyedVectors
import networkx as nx
import os
import glob
from ..method import BaseMethod


class Sentences(object):
    def __init__(self, file_names):
        self.file_names = file_names

    def __iter__(self):
        fs = []
        for file_name in self.file_names:
            fs.append(open(file_name))
        while True:
            flag = 0
            for i, f in enumerate(fs):
                line = f.readline()
                if line != '':
                    flag = 1
                    yield line.split()
            if not flag:
                try:
                    for f in fs:
                        f.close()
                except:
                    pass
                return


class Method(BaseMethod):

    __PARAMS__ = dict(dim=128, walk_length=10, num_walks=80, window_size=10,
                      until_k=4, iter=5, workers=5, flag='sp')

    def getId(self):
        return "riwalk"

    def train(self):
        # remove existing files
        os.system('rm -rf %s' %
                  os.path.join(WALK_FILES_DIR, "__random_walks_*.txt"))
        nx_g, mapping = self.preprocess_graph(self.graph)
        self.embeddings = self.learn(nx_g, mapping)

    def learn_embeddings(self):
        """
        learn embeddings from random walks.
        hs:  0:negative sampling 1:hierarchica softmax
        sg:  0:CBOW              1:skip-gram
        """
        dim = self.params['dimensions']
        window_size = self.params['window_size']
        workers = self.params['workers']
        iter_num = self.params['iter']

        walk_files = glob.glob(os.path.join(
            WALK_FILES_DIR, "__random_walks_*.txt"))
        sentences = Sentences(walk_files)
        model = Word2Vec(sentences, size=dim, window=window_size,
                         min_count=0, sg=1, workers=workers, iter=iter_num)
        return model.wv

    def preprocess_graph(self, nx_g):
        """
        1. relabel nodes with 0,1,2,3,...,N.
        2. convert graph to adjacency representation as a list of tuples.
        """
        mapping = {_: i for i, _ in enumerate(nx_g.nodes())}
        nx_g = nx.relabel_nodes(nx_g, mapping)
        nx_g = [tuple(nx_g.neighbors(_)) for _ in range(len(nx_g))]

        return nx_g, mapping

    def learn(self, nx_g, mapping):
        g = RiWalkGraph.RiGraph(nx_g, self.params)
        g.process_random_walks()

        wv = self.learn_embeddings()

        original_wv = Word2VecKeyedVectors(self.params['dimensions'])
        original_nodes = list(mapping.keys())
        original_vecs = [wv.word_vec(str(mapping[node]))
                         for node in original_nodes]
        original_wv.add(entities=list(map(str, original_nodes)),
                        weights=original_vecs)
        return original_wv
