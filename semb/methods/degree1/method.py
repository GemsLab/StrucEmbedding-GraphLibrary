from semb.methods import BaseMethod
import numpy as np
import networkx as nx

class Method(BaseMethod):
    __PARAMS__ = {}

    def get_id(self):
        return "degree1"

    def train(self):
        dict_degree = dict(self.graph.degree())
        dict_neighbors = {i:[ii for ii in self.graph.neighbors(i)] for i in self.graph.nodes()}
        max_degree = max([i for _,i in dict_degree.items()])
        dict_histogram = dict()
        for cur_node in self.graph.nodes():
            list_histogram = [0.0] * max_degree
            for cur_n in dict_neighbors[cur_node]:
                list_histogram[dict_degree[cur_n] - 1] += 1.0
            dict_histogram[cur_node] = list_histogram

        self.embeddings = dict_histogram
