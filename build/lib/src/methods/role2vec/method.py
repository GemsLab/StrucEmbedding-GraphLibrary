import math
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from gensim.models.doc2vec import Doc2Vec
from .utils import create_documents
from .walkers import FirstOrderRandomWalker, SecondOrderRandomWalker
from .weisfeiler_lehman_labeling import WeisfeilerLehmanMachine
from .motif_count import MotifCounterMachine

from ..method import BaseMethod

"""
Driver for learning Role2Vec embeddings
"""
class Method(BaseMethod):

    __PARAMS__ = dict(window_size=5, walk_number=10, walk_length=80, sampling="first", p=1.0, q=1.0, \
                      dim=128, down_sampling=0.001, alpha=0.025, min_alpha=0.025, min_count=1, workers=1, \
                      epochs=10, features='wl', label_iterations=2, log_base=1.5, graphlet_size=4, \
                      quantiles=5, motif_compression='string', seed=42, factors=8, clusters=50, beta=0.01)

    def getId(self):
        return "role2vec"

    def train(self):
        self.do_walks()
        self.create_structural_features()
        self.pooled_features = self.create_pooled_features()
        self.embedding = self.create_embedding()

    def do_walks(self):
        """
        Doing first/second order random walks.
        """
        if self.args.sampling == "second":
            self.sampler = SecondOrderRandomWalker(self.graph, self.args.P, self.args.Q,  self.args.walk_number, self.args.walk_length)
        else:
            self.sampler = FirstOrderRandomWalker(self.graph, self.args.walk_number, self.args.walk_length)
        self.walks = self.sampler.walks
        del self.sampler

    def create_structural_features(self):
        """
        Extracting structural features.
        """
        if self.args.features == "wl":
            features = {str(node): str(int(math.log(self.graph.degree(node)+1,self.args.log_base))) for node in self.graph.nodes()}
            machine = WeisfeilerLehmanMachine(self.graph, features, self.args.labeling_iterations)
            machine.do_recursions()
            self.features = machine.extracted_features
        elif self.args.features == "degree":
            self.features = {str(node): [str(self.graph.degree(node))] for node in self.graph.nodes()}
        else:
            machine = MotifCounterMachine(self.graph, self.args)
            self.features  = machine.create_string_labels()

    def create_pooled_features(self):
        """
        Pooling the features with the walks
        """
        features = {str(node):[] for node in self.graph.nodes()}
        for walk in self.walks:
            for node_index in range(self.args.walk_length-self.args.window_size):
                for j in range(1,self.args.window_size+1):
                    features[str(walk[node_index])].append(self.features[str(walk[node_index+j])])
                    features[str(walk[node_index+j])].append(self.features[str(walk[node_index])])

        features = {node: [feature for feature_elems in feature_set for feature in feature_elems] for node, feature_set in features.items()}
        return features
   

    def create_embedding(self):
        """
        Fitting an embedding.
        """
        document_collections = create_documents(self.pooled_features)

        model = Doc2Vec(document_collections,
                        vector_size = self.args.dimensions,
                        window = 0, 
                        min_count = self.args.min_count,
                        alpha = self.args.alpha,
                        dm = 0,
                        min_alpha = self.args.min_alpha,
                        sample = self.args.down_sampling,
                        workers = self.args.workers,
                        epochs = self.args.epochs)
        
        embedding = np.array([model.docvecs[str(node)] for node in self.graph.nodes()])
        return embedding
