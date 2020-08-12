from semb.methods import BaseMethod
import numpy as np
import networkx as nx

class Method(BaseMethod):
    __PARAMS__ = {}

    def get_id(self):
        return "degree"

    def train(self):
        self.embeddings = {i: [v] for i, v in dict(self.graph.degree()).items()}
