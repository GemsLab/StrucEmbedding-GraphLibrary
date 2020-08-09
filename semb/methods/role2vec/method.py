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

from semb.methods import BaseMethod

"""
Driver for learning Role2Vec embeddings
"""
class Method(BaseMethod):

    __PARAMS__ = dict(window_size=5, walk_number=10, walk_length=80, sampling="first", p=1.0, q=1.0, \
                      dim=128, down_sampling=0.001, alpha=0.025, min_alpha=0.025, min_count=1, workers=1, \
                      epochs=10, features='wl', label_iterations=2, log_base=1.5, graphlet_size=4, \
                      quantiles=5, motif_compression='string', seed=42, factors=8, clusters=50, beta=0.01)

    def get_id(self):
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
        if self.params['sampling'] == "second":
            self.sampler = SecondOrderRandomWalker(self.graph, self.params['p'], self.params['q'],  self.params['walk_number'], self.params['walk_length'])
        else:
            self.sampler = FirstOrderRandomWalker(self.graph, self.params['walk_number'], self.params['walk_length'])
        self.walks = self.sampler.walks
        del self.sampler

    def create_structural_features(self):
        """
        Extracting structural features.
        """
        if self.params['features'] == "wl":
            features = {str(node): str(int(math.log(self.graph.degree(node)+1,self.params['log_base']))) for node in self.graph.nodes()}
            machine = WeisfeilerLehmanMachine(self.graph, features, self.params['labeling_iterations'])
            machine.do_recursions()
            self.features = machine.extracted_features
        elif self.params['features'] == "degree":
            self.features = {str(node): [str(self.graph.degree(node))] for node in self.graph.nodes()}
        else:
            machine = MotifCounterMachine(self.graph, self.params)
            self.features  = machine.create_string_labels()

    def create_pooled_features(self):
        """
        Pooling the features with the walks
        """
        features = {str(node):[] for node in self.graph.nodes()}
        for walk in self.walks:
            for node_index in range(self.params['walk_length']-self.params['window_size']):
                for j in range(1,self.params['window_size']+1):
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
                        vector_size = self.params['dim'],
                        window = 0, 
                        min_count = self.params['min_count'],
                        alpha = self.params['alpha'],
                        dm = 0,
                        min_alpha = self.params['min_alpha'],
                        sample = self.params['down_sampling'],
                        workers = self.params['workers'],
                        epochs = self.params['epochs'])
        
        embedding = np.array([model.docvecs[str(node)] for node in self.graph.nodes()])
        return embedding
