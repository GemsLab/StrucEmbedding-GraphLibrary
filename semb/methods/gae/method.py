from __future__ import division
from __future__ import print_function

from semb.methods import BaseMethod
import numpy as np
import networkx as nx

from tqdm import tqdm

import time
import os

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import networkx as nx

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from .gae.optimizer import OptimizerAE, OptimizerVAE
from .gae.input_data import load_data
from .gae.model import GCNModelAE, GCNModelVAE
from .gae.preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges

class Method(BaseMethod):
    __PARAMS__ = {
    'learning_rate': 0.01,
    'epochs' : 200,
    'hidden1' : 32,
    'hidden2' : 128,
    'weight_decay' : 0.,
    'dropout' : 0.,
    'model' : 'gcn_vae',
    'features' : 0
    }

    def get_id(self):
        return "vae"

    def train(self):
        tf.compat.v1.flags.DEFINE_string('f','','')

        flags = tf.app.flags
        FLAGS = flags.FLAGS
        flags.DEFINE_float('learning_rate', self.params['learning_rate'], 'Initial learning rate.')
        # flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
        flags.DEFINE_integer('epochs', self.params['epochs'], 'Number of epochs to train.')
        flags.DEFINE_integer('hidden1', self.params['hidden1'], 'Number of units in hidden layer 1.')
        flags.DEFINE_integer('hidden2', self.params['hidden2'], 'Number of units in hidden layer 2.')
        flags.DEFINE_float('weight_decay', self.params['weight_decay'], 'Weight for L2 loss on embedding matrix.')
        flags.DEFINE_float('dropout', self.params['dropout'], 'Dropout rate (1 - keep probability).')
        flags.DEFINE_string('model', self.params['model'], 'Model string.')
        flags.DEFINE_integer('features', self.params['features'], 'Whether to use features (1) or not (0).')

        adj = nx.adjacency_matrix(self.graph)
        adj_orig = adj
        adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
        adj_orig.eliminate_zeros()

        adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
        adj = adj_train

        print('finish making mask_test_edges')

        if FLAGS.features == 0:
            features = sp.identity(adj.shape[0])  # featureless

        # Some preprocessing
        adj_norm = preprocess_graph(adj)

        # Define placeholders
        placeholders = {
            'features': tf.sparse_placeholder(tf.float32),
            'adj': tf.sparse_placeholder(tf.float32),
            'adj_orig': tf.sparse_placeholder(tf.float32),
            'dropout': tf.placeholder_with_default(0., shape=())
        }

        num_nodes = adj.shape[0]

        features = sparse_to_tuple(features.tocoo())
        num_features = features[2][1]
        features_nonzero = features[1].shape[0]

        model_str = FLAGS.model
        # Create model
        model = None
        if model_str == 'gcn_ae':
            model = GCNModelAE(placeholders, num_features, features_nonzero)
        elif model_str == 'gcn_vae':
            model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)

        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

        # Optimizer
        with tf.name_scope('optimizer'):
            if model_str == 'gcn_ae':
                opt = OptimizerAE(preds=model.reconstructions,
                                  labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                              validate_indices=False), [-1]),
                                  pos_weight=pos_weight,
                                  norm=norm)
            elif model_str == 'gcn_vae':
                opt = OptimizerVAE(preds=model.reconstructions,
                                   labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                               validate_indices=False), [-1]),
                                   model=model, num_nodes=num_nodes,
                                   pos_weight=pos_weight,
                                   norm=norm)

        # Initialize session
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        cost_val = []
        acc_val = []

        def get_roc_score(edges_pos, edges_neg, emb=None):
            if emb is None:
                feed_dict.update({placeholders['dropout']: 0})
                emb = sess.run(model.z_mean, feed_dict=feed_dict)

            def sigmoid(x):
                return 1 / (1 + np.exp(-x))

            # Predict on test set of edges
            adj_rec = np.dot(emb, emb.T)
            preds = []
            pos = []
            for e in edges_pos:
                preds.append(sigmoid(adj_rec[e[0], e[1]]))
                pos.append(adj_orig[e[0], e[1]])

            preds_neg = []
            neg = []
            for e in edges_neg:
                preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
                neg.append(adj_orig[e[0], e[1]])

            preds_all = np.hstack([preds, preds_neg])
            labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
            roc_score = roc_auc_score(labels_all, preds_all)
            ap_score = average_precision_score(labels_all, preds_all)

            return roc_score, ap_score

        val_roc_score = []

        adj_label = adj_train + sp.eye(adj_train.shape[0])
        adj_label = sparse_to_tuple(adj_label)

        # Train model
        for epoch in tqdm(range(FLAGS.epochs)):

            t = time.time()
            # Construct feed dictionary
            feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})
            # Run single weight update
            outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)
            

            # Compute average loss
            avg_cost = outs[1]
            avg_accuracy = outs[2]

            roc_curr, ap_curr = get_roc_score(val_edges, val_edges_false)
            val_roc_score.append(roc_curr)

            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
                  "train_acc=", "{:.5f}".format(avg_accuracy), "val_roc=", "{:.5f}".format(val_roc_score[-1]),
                  "val_ap=", "{:.5f}".format(ap_curr),
                  "time=", "{:.5f}".format(time.time() - t))

        print("Optimization Finished!")
        emb = sess.run(model.z_mean, feed_dict=feed_dict)
        print(emb.shape)
        list_nodes = list(self.graph.nodes)
        self.embeddings = dict()
        for i in range(0, emb.shape[0]):
            self.embeddings[list_nodes[i]] = emb[i].tolist()

   