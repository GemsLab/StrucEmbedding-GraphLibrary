from ..exceptions import UnimplementedException, MethodKeywordUnAllowedException
import networkx as nx

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
        raise InputFormatError("Please input graph as NetworkX.graph object")

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

def centrality_correlation(graph, embedding, centrality='degree', **kwargs):
    """
    Get the correlation of the graph

    Arguments:
    graph {nx.Graph} -- NetworkX graph
    embedding {dict} -- the embedding file from the embedding method.
                        key: nodeid, val: embedding, list[float]
    centrality {string} -- chosen from ['degree', 'pagerank', 'betweeness', 'clustering_coeff']

    Return:
    dict_centrality {dict} -- key: node_id, val: centrality
    """
    
