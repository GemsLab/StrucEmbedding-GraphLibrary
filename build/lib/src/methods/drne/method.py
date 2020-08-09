from ..method import BaseMethod
from . import network
from .eni import eni
import numpy as np
import time
import os
import tensorflow as tf


class Method(BaseMethod):

    __PARAMS__ = dict(embedding_path=16, epochs_to_train=20, batch_size=16, learning_rate=0.0025,
                      undirected=True, alpha=0.0, lamb=0.5, grad_clip=5.0, k=1, sampling_size=100,
                      seed=1, index_from_0=True, train_device='cpu', save_path=os.getcwd())

    def getId(self):
        return "drne"

    def train(self):
        np.random.seed(int(time.time())
                       if self.params['seed'] == -1 else self.params['seed'])
        network.sort_graph_by_degree(self.graph)
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Graph().as_default(), tf.Session(config=config) as sess, tf.device(self.params['train_device']):
            alg = eni(self.graph, self.params, sess)
            print("max degree: {}".format(alg.degree_max))
            alg.train()
            self.embeddings = alg.get_embeddings()
