from ..exceptions import *
import networkx as nx
import numpy as np
import sklearn
from sklearn import metrics

def get_similarity_matrix(list_emb_sorted_id, list_sorted_id, metric):
    np_emb_sorted_id = np.array(list_emb_sorted_id)
    list_tuple_distances = list()
    # print("Shape for the embedding file:", np_emb_sorted_id.shape)
    # if metric not in ['cosine', 'euclidean', 'dot']:
    #     print("Please choose from [cosine, euclidean, dot]")
    if metric in ['cosine', 'euclidean']:
        # print("Calculating pairwise distance using", metric)
        pairwise_distance = sklearn.metrics.pairwise_distances(np_emb_sorted_id, metric=metric)
    elif metric == 'dot':
        # print("Calculating pairwise distance using dot product")
        pairwise_distance = np.zeros((len(np_emb_sorted_id), len(np_emb_sorted_id)))
        for idx in range(0, len(pairwise_distance)):
            pairwise_distance[idx, :] += np.dot(np_emb_sorted_id, np_emb_sorted_id[idx])
    return pairwise_distance

def get_centrality(graph, centrality='degree', **kwargs):
    """
    Get the correlation of the graph

    Arguments:
    graph {nx.Graph} -- NetworkX graph
    centrality {string} -- chosen from ['degree', 'pagerank', 'betweeness', 'clustering_coeff']

    Return:
    dict_centrality {dict} -- key: node_id, val: centrality
    """

    if not isinstance(graph, nx.classes.graph.Graph):
        raise InputFormatErrorException("Please input graph as NetworkX.graph object")

    if centrality == 'degree':
        dict_centrality = dict(graph.degree())
    elif centrality == 'pagerank':
        dict_centrality = dict(nx.pagerank(graph))
    elif centrality == 'betweeness':
        dict_centrality = dict(nx.algorithms.centrality.betweenness_centrality(graph))
    elif centrality == 'clustering_coeff':
        dict_centrality = dict(nx.clustering(graph))
    else:
        raise MethodKeywordUnAllowedException("Please choose centrality from ['degree', 'pagerank', 'betweeness', 'clustering_coeff']")

    return dict_centrality

def centrality_correlation(graph, dict_embeddings, centrality='degree', similarity='cosine', n_neighbors=5, **kwargs):
    """
    1. For each node $v$ in graph $G$, we calculate a property of interest $p_i(v)$. We consider four properties: degree, PageRank (with damping parameter $\alpha=0.85$), clustering coefficient, and betweenness centrality.
    2. We identify $v$'s $k$-nearest neighbors ($k$-NN) in the embedding space $R^d$ using cosine or Euclidean distance, and compute the average value for each structural property, $\overline{p_{i,kNN}(v)}$​.
    3. Per property $p_i$, we calculate the Pearson correlation between the structural property of a node and its $k$-NN across all nodes.

    Arguments:
    graph {nx.Graph} -- NetworkX graph
    dict_embeddings {dict} -- the embedding file from the embedding method.
                        key: nodeid, val: embedding, list[float]
    centrality {string} -- chosen from ['degree', 'pagerank', 'betweeness', 'clustering_coeff']
    similarity {string} -- chosen from ['dot', 'cosine', 'euclidean'], the similarity metric for 
                           calculating the similarity in the embedding space

    Return:
    float, Pearson correlation
    """

    if not isinstance(graph, nx.classes.graph.Graph):
        raise InputFormatErrorException("Please input graph as NetworkX.graph object")

    if centrality == 'degree':
        dict_centrality = dict(graph.degree())
    elif centrality == 'pagerank':
        dict_centrality = dict(nx.pagerank(graph))
    elif centrality == 'betweeness':
        dict_centrality = dict(nx.algorithms.centrality.betweenness_centrality(graph))
    elif centrality == 'clustering_coeff':
        dict_centrality = dict(nx.clustering(graph))
    else:
        raise MethodKeywordUnAllowedException("Please choose centrality from ['degree', 'pagerank', 'betweeness', 'clustering_coeff']")

    if similarity not in ['dot', 'cosine', 'euclidean']:
        raise MethodKeywordUnAllowedException("Please choose similarity from ['dot', 'euclidean', 'cosine']")

    # Perform checking between the nodes in the graph and the nodes in the embedding files
    list_nodes_embeddings = set([i for i, _ in dict_embeddings.items()])
    list_nodes_labels = set([i for i, _ in dict_centrality.items()])
    list_nodes_intersection = sorted(list(list_nodes_labels))

    if list_nodes_embeddings != list_nodes_labels:
        print("Warning: Nodes from the input embeddings and the input labels mismatch!")
        print("Perform classification on the intersection.")
        list_nodes_intersection = sorted(list(list_nodes_embeddings.intersection(list_nodes_labels)))

    list_centrality_node_id = [float(dict_centrality[i]) for i in list_nodes_intersection]
    
    list_x = list()
    list_y = list()

    list_emb = [dict_embeddings[key] for key in list_nodes_intersection]
    pairwise_distance = get_similarity_matrix(list_emb, list_nodes_intersection, similarity)

    for cur_node in range(0, len(list_nodes_intersection)):
        cur_sim_ranking = sorted(list(range(len(list_nodes_intersection))), key=lambda v: pairwise_distance[cur_node][v])[1:n_neighbors+1]
        list_y += [np.mean([list_centrality_node_id[i] for i in cur_sim_ranking])]
        list_x += [list_centrality_node_id[cur_node]]

    return np.corrcoef(list_x, list_y)[0][1]
