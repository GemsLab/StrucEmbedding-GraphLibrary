from ..exceptions import *

import networkx as nx
import numpy as np

from sklearn.cluster import KMeans

from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn import metrics


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def kmeans_best_result(X, y, n_clusters):
    list_purity = list()
    list_nmi = list()
    kmeans = KMeans(n_clusters=n_clusters, n_init=min(len(y), 1000), init='k-means++').fit(X)
    return {'purity': purity_score(y, kmeans.labels_), 'nmi': normalized_mutual_info_score(y, kmeans.labels_)}


def perform_clustering(dict_embeddings, dict_labels, **kwargs):
    """
    Perform clustering as requested
    The number of clusters is got from the number of number of distinct labels

    Arguments:
    dict_embeddings {dict} -- the output embeddings from the methods
    dict_labels {dict} -- the ground truth labels from the get_label method
                           
    Return:
    dict_performance {dict}

    ***
    Special Notice
    ***
    Currently SEMB only supports single label classification and clustering. Multi-label classification
    and clustering will come out in the following versions.
    """

    list_nodes_embeddings = set([i for i, _ in dict_embeddings.items()])
    list_nodes_labels = set([i for i, _ in dict_labels.items()])
    list_nodes_intersection = sorted(list(list_nodes_labels))

    if list_nodes_embeddings != list_nodes_labels:
        print("Warning: Nodes from the input embeddings and the input labels mismatch!")
        print("Perform classification on the intersection.")
        list_nodes_intersection = sorted(list(list_nodes_embeddings.intersection(list_nodes_labels)))

    X = list()
    y = list()
    for cur_node in list_nodes_intersection:
        X += [dict_embeddings[cur_node]]
        y += [dict_labels[cur_node]]
    X = np.array(X)
    y = np.array(y)

    N_CLUSTERS = len(np.unique(y))

    dict_results = kmeans_best_result(X, y, N_CLUSTERS)

    return {"overall": dict_results}
