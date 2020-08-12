from ..exceptions import UnimplementedException, MethodKeywordUnAllowedException
import networkx as nx

def get_label(input_dir, delimeter = ' ' ,**kwargs):
    """
    Read in the label file for the downstreaming tasks

    Arguments:
    input_dir {string} -- input directory
    delimeter {char} -- the delimeter between the node_id and node_label. Default is ' '

    Return:
    dict_labels {dict} -- key {int}: node_id, val {int}: node label

    ***
    Special Notice
    ***
    Currently SEMB only supports single label classification and clustering. Multi-label classification
    and clustering will come out in the following versions.
    """

    dict_labels = dict()
    dict_counter = dict()
    with open(input_dir, 'r') as f:
        lines = f.read().splitlines()
    for line in lines:
        dict_labels[int(line.split(' ')[0])] = int(line.split(' ')[1])
        if int(line.split(' ')[1]) not in dict_counter:
            dict_counter[int(line.split(' ')[1])] = 1
        else:
            dict_counter[int(line.split(' ')[1])] += 1
    print("Read in", len(dict_labels), 'node labels.')
    for key, val in dict_counter.items():
        print(">>> Label", key, 'appears', val, 'times')
    return dict_labels
