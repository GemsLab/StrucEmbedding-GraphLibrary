from semb.methods import BaseMethod
import numpy as np
import math

from scipy.linalg import svd
from grakel import Graph
from grakel.kernels import WeisfeilerLehman, VertexHistogram, ShortestPath
from .utils import extract_egonets


class Method(BaseMethod):

    __PARAMS__ = dict(radius=2, dim=128, kernel='weisfeiler_lehman')

    def get_id(self):
        return "segk"

    def train(self):
        nodes = [i for i in self.graph.nodes()]
        edges = [e for e in self.graph.edges() if e[0] != e[1]]
        reps = self.segk(nodes, edges, self.params['radius'], self.params['dim'], self.params['kernel'])
        self.embeddings = dict()
        for i, cur_node in enumerate(self.graph.nodes):
            self.embeddings[cur_node] = reps[i, :].tolist()



    def segk(self, nodes, edgelist, radius, dim, kernel):
        n = len(nodes)

        if kernel == 'shortest_path':
            gk = [ShortestPath(normalize=True, with_labels=True)
                    for i in range(radius)]
        elif kernel == 'weisfeiler_lehman':
            gk = [WeisfeilerLehman(
                n_iter=4, normalize=True, base_kernel=VertexHistogram) for i in range(radius)]
        else:
            raise ValueError('Use a valid kernel!!')

        #####
        # if number of nodes < 128, set the dim to be number of nodes
        # otherwise, set dim to be 128
        if n < 128:
            dim = 2 ** math.floor(math.log(n - 1, 2))
            print("Warning! Only", str(n),
                    "nodes in the graph. Set dim to be", str(dim))
        #####

        idx = np.random.permutation(n)
        sampled_nodes = [nodes[idx[i]] for i in range(dim)]
        remaining_nodes = [nodes[idx[i]] for i in range(dim, len(nodes))]

        egonet_edges, egonet_node_labels = extract_egonets(edgelist, radius)

        E = np.zeros((n, dim))

        K = np.zeros((dim, dim))
        K_prev = np.ones((dim, dim))
        for i in range(1, radius+1):
            Gs = list()
            for node in sampled_nodes:
                node_labels = {v: egonet_node_labels[node][v]
                                for v in egonet_node_labels[node] if egonet_node_labels[node][v] <= i}
                edges = list()
                for edge in egonet_edges[node]:
                    if edge[0] in node_labels and edge[1] in node_labels:
                        edges.append((edge[0], edge[1]))
                        edges.append((edge[1], edge[0]))
                Gs.append(Graph(edges, node_labels=node_labels))

            K_i = gk[i-1].fit_transform(Gs)
            K_i = np.multiply(K_prev, K_i)
            K += K_i
            K_prev = K_i

        U, S, V = svd(K)
        S = np.maximum(S, 1e-12)
        Norm = np.dot(U * 1. / np.sqrt(S), V)
        E[idx[:dim], :] = np.dot(K, Norm.T)

        K = np.zeros((n-dim, dim))
        K_prev = np.ones((n-dim, dim))
        for i in range(1, radius+1):
            Gs = list()
            for node in remaining_nodes:
                node_labels = {v: egonet_node_labels[node][v]
                                for v in egonet_node_labels[node] if egonet_node_labels[node][v] <= i}
                edges = list()
                for edge in egonet_edges[node]:
                    if edge[0] in node_labels and edge[1] in node_labels:
                        edges.append((edge[0], edge[1]))
                        edges.append((edge[1], edge[0]))
                Gs.append(Graph(edges, node_labels=node_labels))

            K_i = gk[i-1].transform(Gs)
            K_i = np.multiply(K_prev, K_i)
            K += K_i
            K_prev = K_i

        E[idx[dim:], :] = np.dot(K, Norm.T)

        return E
