import pandas as pd
import networkx as nx
from texttable import Texttable
from gensim.models.doc2vec import TaggedDocument

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(),args[k]] for k in keys])
    print(t.draw())

def load_graph(graph_path):
    """
    Reading an edge list csv to an NX graph object.
    :param graph_path: Path to the edhe list csv.
    :return graph: NetworkX object.
    """
    graph = nx.from_edgelist(pd.read_csv(graph_path).values.tolist())
    graph.remove_edges_from(graph.selfloop_edges())
    return graph

def load_graph_(graph_path):
    dict_node_o2n = dict()

    list_edges = list()
    with open(graph_path, 'r') as f:
        lines = f.read().splitlines()

    count = 0
    for line in lines:
        src = int(line.split(' ')[0])
        dst = int(line.split(' ')[1])

        if src not in dict_node_o2n:
            dict_node_o2n[src] = count
            count += 1
        if dst not in dict_node_o2n:
            dict_node_o2n[dst] = count
            count += 1
        list_edges += [(dict_node_o2n[src], dict_node_o2n[dst])]

    print("Read in", count, " nodes!")

    G = nx.Graph()
    G.add_edges_from(list_edges)
    return G, dict_node_o2n

def create_documents(features):
    """
    Created tagged documents object from a dictionary.
    :param features: Keys are document ids and values are strings of the document.
    :return docs: List of tagged documents.
    """
    docs = [TaggedDocument(words = v, tags = [str(k)]) for k, v in features.items()]
    return docs
