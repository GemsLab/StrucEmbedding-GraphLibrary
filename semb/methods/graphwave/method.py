from semb.methods import BaseMethod
from .internal.graphwave import *
import networkx as nx
import scipy.sparse as sparse
import scipy.io as sio
import scipy.sparse as sp
import sys


class Method(BaseMethod):

    __PARAMS__ = dict(dim=128, taus='auto', thresh=20000, time_bounds=(0, 100))

    def get_id(self):
        return "graphwave"

    def train(self):
        if self.params['taus'] != 'auto':
            taus_input = [float(self.params['taus'])]
        else:
            taus_input = self.params['taus']
        # learns representations of dimension 4x as many time points?
        time_points = np.linspace(
            self.params['time_bounds'][0], self.params['time_bounds'][1], num=int(self.params['dim'])/4)
        if self.graph.number_of_nodes() > self.params['thresh']:
            print('Using Chebyshev polynomial approximation of heat kernel')
            proc = 'approx'
        else:
            proc = 'exact'
        representations, heat_print, taus = graphwave_alg(
            self.graph, time_points, taus=taus_input, verbose=False, proc=proc)
        self.embeddings = dict()
        list_nodes = list(self.graph.nodes())
        for i in range(0, representations.shape[0]):
            self.embeddings[list_nodes[i]] = representations[i].tolist()
            