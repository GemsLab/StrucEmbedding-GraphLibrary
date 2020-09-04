from semb.methods import BaseMethod
from . import network
from .eni import eni
import numpy as np
import time
import os
import tensorflow as tf
import copy
import networkx as nx

folder_eni = os.path.join(os.path.dirname(__file__), "eni/")

class Method(BaseMethod):

    __PARAMS__ = dict(embedding_size=128, epochs_to_train=20, batch_size=256, learning_rate=0.0025,
                      alpha=0.0, lamb=0.5, grad_clip=5.0, k=1, sampling_size=100,
                      seed=1, index_from_0=True, train_device='cpu', save_path=folder_eni, save_suffix='eni')

    def get_id(self):
        return "drne"

    def train(self):
        np.random.seed(int(time.time())
                       if self.params['seed'] == -1 else self.params['seed'])

        self.old_graph = copy.deepcopy(self.graph)
        old_edges = [e for e in self.old_graph.edges()]
        dict_node_o2n = dict()
        list_edges = list()

        count = 1
        for edge in old_edges:
            src = int(edge[0])
            dst = int(edge[1])

            if src not in dict_node_o2n:
                dict_node_o2n[src] = count
                count += 1
            if dst not in dict_node_o2n:
                dict_node_o2n[dst] = count
                count += 1
            list_edges += [(dict_node_o2n[src], dict_node_o2n[dst])]

        graph = [[]]
        for _ in range(len(dict_node_o2n)):
            graph += [[]]

        for cur_edge in list_edges:
            src = cur_edge[0]
            dst = cur_edge[1]
            if dst not in graph[src]:
                graph[src] += [dst]
            if src not in graph[dst]:
                graph[dst] += [src]

        self.graph = graph

        network.sort_graph_by_degree(self.graph)
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Graph().as_default(), tf.Session(config=config) as sess, tf.device(self.params['train_device']):
            alg = eni(self.graph, self.params, sess)
            # print("max degree: {}".format(alg.degree_max))
            alg.train()
            reps = alg.get_embeddings()

        dict_n2o = {val: key for key, val in dict_node_o2n.items()}
        self.embeddings = dict()
        for i in range(0, reps.shape[0]):
            self.embeddings[dict_n2o[i+1]] = reps[i].tolist()
