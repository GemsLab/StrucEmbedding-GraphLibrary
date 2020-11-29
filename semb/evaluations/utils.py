from ..exceptions import *
import networkx as nx
import pandas as pd

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
        cur_node = int(line.split(delimeter)[0])
        cur_label = int(line.split(delimeter)[1])
        dict_labels[cur_node] = cur_label
        if cur_label not in dict_counter:
            dict_counter[cur_label] = 1
        else:
            dict_counter[cur_label] += 1
    print("Read in", len(dict_labels), 'node labels.')
    for key, val in dict_counter.items():
        print(">>> Label", key, 'appears', val, 'times')
    return dict_labels

def concatenate_result_pd(list_results):
    """
    Concatenate the results from the clustering / classifcation test
    Arguments:
    list_results {list of tuples} --  [("method_name", dict_result)], where the dict_result is the returned dict
                                      from the perform_clustering() and perform_classification() functions
    
    Return:
    pd_results -- a pandas table showing the results

    """
    # Perform input checking on the list_results
    if len(list_results) == 0:
        raise InputFormatErrorException("Input length 0!")

    for cur_item in list_results:
        if (len(cur_item) != 2):
            raise InputFormatErrorException("Please input the results as list of tuples, i.e. [(\"method_name\", dict_result)]")

        if (not isinstance(cur_item[0], str)) or (not isinstance(cur_item[1], dict)):
            raise InputFormatErrorException("Please input the results as list of tuples, i.e. [(\"method_name\", dict_result)]")

        if "overall" not in cur_item[1]:
            raise InputFormatErrorException("Invalid input. Please make sure that the input result is generated from perform_classification() or perform_clustering()")

    
    pd_results = pd.DataFrame()
    pd_results['methods'] = [i[0] for i in list_results]


    
    # Peform checking on whether classifcation or clustering is tested
    if 'accuracy' in list_results[0][1]['overall']:
        # Classification
        for cur_item in list_results:
            if 'accuracy' not in cur_item[1]['overall']:
                raise InputFormatErrorException("Invalid input. Please make sure that the input result is generated from perform_classification()")
                
        for metric in ['accuracy', 'f1_macro', 'f1_micro', 'auc_micro', 'auc_macro']:
            for value in ['mean', 'std']:
                pd_results[metric + '_' + value] = [i[1]['overall'][metric][value] for i in list_results]
    else:
        # Clustering
        for cur_item in list_results:
            if 'purity' not in cur_item[1]['overall']:
                raise InputFormatErrorException("Invalid input. Please make sure that the input result is generated from perform_clustering()")
        
        for metric in ['purity', 'nmi']:
            pd_results[metric] = [i[1]['overall'][metric] for i in list_results]
        
    return pd_results

