from collections import defaultdict
from functools import reduce
import numpy as np
import networkx as nx
import random
import os, sys
import tensorflow as tf
import six
from . import utils

def get_degree(graph):
    degree = [[] for _ in range(len(graph))]
    for i in range(len(graph)):
        degree[i] = len(graph[i])
    return degree

def sort_graph_by_degree(graph):
    degree = get_degree(graph)
    for i in range(len(graph)):
        graph[i] = sorted(graph[i], key=lambda x: degree[x])
    return graph

def get_max_degree(graph):
    return max([len(i) for i in graph])

def neighborhood_sum(*a):
    return list(map(lambda x: reduce(lambda y, z: y+z, x), zip(*a)))

def p_by_degree(x, graph):
    res = np.array([len(graph[i])+1e-2 for i in graph[x]], dtype=float)
    return res/np.sum(res)

def sampling_with_order(x, sampling=False, sampling_size=50, p=None, padding=False):
    if sampling and len(x) > sampling_size:
        #indices = random.sample(range(len(x)), sampling_size)
        indices = np.random.choice(len(x), sampling_size, replace=False, p=p)
        return [x[i] for i in sorted(indices)], sampling_size
    if padding and len(x) < sampling_size:
        return x+[0]*(sampling_size-len(x)), len(x)
    return x, sampling_size

def get_neighborhood_list(node, graph, sampling=True, sampling_size=100):
    p = p_by_degree(node, graph)
    nodes, seqlen = sampling_with_order(graph[node], sampling, sampling_size, p=p, padding=True)
    return node, nodes, seqlen


def degree_to_class(degree, degree_max, nums_class):
    ind = int(np.log2(degree)/np.log2(degree_max+1)*nums_class)
    res = np.zeros(nums_class)
    res[ind] = 1
    return res

def next_batch(graph, degree_max=None, sampling=False, sampling_size=100):
    if degree_max is None:
        degree_max = get_max_degree(graph)
    while True:
        l = np.random.permutation(range(1, len(graph)))
        for i in l:
            #yield (get_neighborhood(i, K, graph, padding=False, sampling=sampling, sampling_size=sampling_size), np.log(len(graph[i])))
            if len(graph[i]) == 0:
                continue
            yield (get_neighborhood_list(i, graph, sampling=sampling, sampling_size=sampling_size), np.log(len(graph[i])+1))

def get_centrality(network, type_='degree', undirected=True):
    if type_ == 'degree':
        if undirected:
            return nx.degree_centrality(network)
        else:
            return nx.in_degree_centrality(network)
    elif type_ == 'closeness':
        return nx.closeness_centrality(network)
    elif type_ == 'betweenness':
        return nx.betweenness_centrality(network)
    elif type_ == 'eigenvector':
        return nx.eigenvector_centrality(network)
    elif type_ == 'kcore':
        return nx.core_number(network)
    else:
        return None

def save_centrality(data, file_path, type_):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    print("\n".join(map(str, data.values())), file=open(os.path.join(file_path, "{}.index".format(type_)), 'w'))

def load_centrality(file_path, type_):
    filename = os.path.join(file_path, "{}.index".format(type_))
    data = np.loadtxt(filename)
    if type_ == 'spread_number':
        return data
    return np.vstack((np.arange(len(data)), data)).T

def save_to_tsv(labels, file_path, degree_max):
    res = []
    for type_ in labels:
        data = load_centrality(file_path, type_)
        if type_ == 'degree':
            data *= degree_max
        res.append(data)
    res = np.vstack(res).astype(int).T.tolist()
    with open(os.path.join(file_path, 'index.tsv'), 'w') as f:
        print("\t".join(labels), file=f)
        for data in res:
            print("\t".join(list(map(str, data))), file=f)
    return res
