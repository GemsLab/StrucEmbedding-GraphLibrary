from ..exceptions import UnimplementedException, MethodKeywordUnAllowedException
import networkx as nx
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import sklearn.model_selection
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn import metrics


def perform_classification(dict_embeddings, dict_labels, classifier='LogisticRegression', **kwargs):
    """
    Read in the label file for the downstreaming tasks

    Arguments:
    dict_embeddings {dict} -- the output embeddings from the methods
    dict_labels {dict} -- the labels from the get_label method

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
    list_nodes_intersection = list_nodes_labels

    if list_nodes_embeddings != list_nodes_labels:
        print("Warning: Nodes from the input embeddings and the input labels mismatch!")
        print("Perform classification on the intersection.")
        list_nodes_intersection = list_nodes_embeddings.intersection(list_nodes_labels)

    X = list()
    y = list()
    for cur_node in list_nodes_intersection:
        X += [dict_embeddings[cur_node]]
        y += [dict_labels[cur_node]]
    X = np.array(X)
    y = np.array(y)

    kf = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=0)
    kf.get_n_splits(X)

    dict_performance_ = dict()

    for idx, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # TODO: Add more classifiers here
        clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', penalty='l2', C=1.0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = metrics.accuracy_score(y_test, y_pred)
        f1_micro = metrics.f1_score(y_test, y_pred, average='micro')
        f1_macro = metrics.f1_score(y_test, y_pred, average='macro')
        dict_ = {'accuracy': acc, 'f1_macro': f1_macro, 'f1_micro': f1_micro}
        dict_performance_[idx] = dict_

    dict_overall = dict()
    for cur_metric in ['accuracy', 'f1_macro', 'f1_micro']:
        dict_ = dict()
        dict_['mean'] = np.mean([dict_performance_[i][cur_metric] for i in range(5)])
        dict_['std'] = np.std([dict_performance_[i][cur_metric] for i in range(5)])
        dict_overall[cur_metric] = dict_

    return {"overall": dict_overall, "detailed": dict_performance_}







