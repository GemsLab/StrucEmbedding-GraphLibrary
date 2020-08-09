import csv
import numpy as np

def read_edgelist(path, delimiter, nodetype=str, cols=2):
    edges = list()
    with open(path, encoding='utf8') as f:
        for line in f:
            t = line.split(delimiter)
            if cols == 2:
                edges.append((nodetype(t[0]),nodetype(t[1][:-1])))
            elif cols > 2:
                edges.append((nodetype(t[0]),nodetype(t[1])))
    
    edges = [e for e in edges if e[0] != e[1]]

    nodes = set()
    for e in edges:
        if e[0] not in nodes:
            nodes.add(e[0])
        if e[1] not in nodes:
            nodes.add(e[1])
            
    nodes = list(nodes)         
    
    return nodes, edges


def write_to_file(path, nodes, embeddings, delimiter=' '):
    with open(path, 'w') as f:
        f.write(str(len(nodes)) + ' ' + str(len(embeddings[0])) + '\n')
        # writer = csv.writer(f, delimiter=delimiter)
        for i,node in enumerate(nodes):
            lst = [node]
            lst.extend(embeddings[i,:].tolist())
            # writer.writerow(lst)
            lst = [str(i) for i in lst]
            f.write(' '.join(lst) + '\n')


def extract_egonets(edgelist, radius, node_labels=None):
    nodes = list()
    neighbors = dict()
    for e in edgelist:
        if e[0] not in neighbors:
            neighbors[e[0]] = [e[1]]
            nodes.append(e[0])
        else:
            neighbors[e[0]].append(e[1])

        if e[1] not in neighbors:
            neighbors[e[1]] = [e[0]]
            nodes.append(e[1])
        else:
            neighbors[e[1]].append(e[0])

    egonet_edges = dict()
    egonet_node_labels = dict()
    for node in nodes:
        egonet_edges[node] = set()
        egonet_node_labels[node] = {node: 0}

    for i in range(1, radius+1):
        for node in nodes:
            leaves = [v for v in egonet_node_labels[node] if egonet_node_labels[node][v]==(i-1)]
            for leaf in leaves:
                for v in neighbors[leaf]:
                    if v not in egonet_node_labels[node]:
                        egonet_node_labels[node][v] = i
                        egonet_edges[node].add((v,leaf))
                        for v2 in neighbors[v]:
                            if v2 in egonet_node_labels[node]:
                                egonet_edges[node].add((v,v2))

    return egonet_edges, egonet_node_labels


def load_graph_classification_data(ds_name, use_node_labels):
    nodes = list()
    edges = list()
    graph_indicator = dict()
    node_labels = None

    with open("datasets/graph_classification/%s/%s_graph_indicator.txt"%(ds_name,ds_name), "r") as f:
        c = 1
        for line in f:
            graph_indicator[c] = int(line[:-1])
            nodes.append(c)
            c += 1

    with open("datasets/graph_classification/%s/%s_A.txt"%(ds_name,ds_name), "r") as f:
        for line in f:
            edge = line[:-1].split(",")
            edge[1] = edge[1].replace(" ", "")
            edges.append((int(edge[0]), int(edge[1])))

    if use_node_labels:
        node_labels = dict()
        with open("datasets/graph_classification/%s/%s_node_labels.txt"%(ds_name,ds_name), "r") as f:
            c = 1
            for line in f:
                node_labels[c] = int(line[:-1])
                c += 1

    class_labels = list()
    with open("datasets/graph_classification/%s/%s_graph_labels.txt"%(ds_name,ds_name), "r") as f:
        for line in f:
            class_labels.append(int(line[:-1]))

    class_labels  = np.array(class_labels, dtype=np.float32)
    return nodes, edges, graph_indicator, node_labels, class_labels

 
def pyramid_match_kernel(Us, d=20, L=4):
    N = len(Us)
    
    Hs = {}
    for i in range(N):
        n = Us[i].shape[0]
        Hs[i] = []
        for j in range(L):
            l = 2**j
            D = np.zeros((d, l))
            T = np.floor(Us[i]*l)
            T[np.where(T==l)] = l-1
            for p in range(Us[i].shape[0]):
                if p >= n:
                    continue
                for q in range(Us[i].shape[1]):
                    D[q,int(T[p,q])] = D[q,int(T[p,q])] + 1

            Hs[i].append(D)


    K = np.zeros((N,N))

    for i in range(N):
        for j in range(i,N):
            k = 0
            intersec = np.zeros(L)
            for p in range(L):
                intersec[p] = np.sum(np.minimum(Hs[i][p], Hs[j][p]))

            k = k + intersec[L-1]
            for p in range(L-1):
                k = k + (1.0/(2**(L-p-1)))*(intersec[p]-intersec[p+1])

            K[i,j] = k
            K[j,i] = K[i,j]

    return K
