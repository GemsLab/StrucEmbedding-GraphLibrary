from ..exceptions import *

import networkx as nx
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer
import sklearn.model_selection

from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

def multiclass_roc_auc_score(truth, pred, average="micro"):
    lb = LabelBinarizer()
    lb.fit(truth)
    truth = lb.transform(truth)
    return roc_auc_score(truth, pred, average=average)

def perform_classification(dict_embeddings, dict_labels, classifier='LogisticRegression', **kwargs):
    """
    Perform classification as per the input request

    Arguments:
    dict_embeddings {dict} -- the output embeddings from the methods
    dict_labels {dict} -- the labels from the get_label method
    classifier {string} -- the classifier to be used for performing classification.
                           Please choose from ['LogisticRegression', 'SVM', 'KNN']
                           
    Return:
    dict_performance {dict}

    ***
    Special Notice
    ***
    Currently SEMB only supports single label classification and clustering. Multi-label classification
    and clustering will come out in the following versions.
    """
    # Error Handling
    if classifier not in ['LogisticRegression', 'SVM', 'KNN']:
        raise MethodKeywordUnAllowedException("Please choose classifier from ['LogisticRegression', 'SVM', 'KNN']!")

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
    X = np.array(X).astype("float")
    y = np.array(y)

    kf = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=0)
    kf.get_n_splits(X)

    dict_performance_ = dict()

    for idx, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # TODO: Add more classifiers here
        clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', penalty='l2', C=1.0)
        if classifier == 'SVM':
            clf = OneVsRestClassifier(
                svm.LinearSVC(
                    penalty='l2', 
                    C=1.0, 
                    multi_class='ovr'), 
                n_jobs=-1)
        elif classifier == 'KNN':
            clf = KNeighborsClassifier(
                n_neighbors=5, 
                metric='euclidean', 
                n_jobs=-1)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = round(metrics.accuracy_score(y_test, y_pred), 4)
        f1_micro = round(metrics.f1_score(y_test, y_pred, average='micro'), 4)
        f1_macro = round(metrics.f1_score(y_test, y_pred, average='macro'), 4)
        dict_ = {'accuracy': acc, 'f1_macro': f1_macro, 'f1_micro': f1_micro}

        if classifier in ['SVM', 'LogisticRegression']:
            y_scores = clf.decision_function(X_test)
            roc_auc_micro = multiclass_roc_auc_score(y_test, y_scores, average='micro')
            roc_auc_macro = multiclass_roc_auc_score(y_test, y_scores, average='macro')
            dict_['auc_micro'] = round(roc_auc_micro, 4)
            dict_['auc_macro'] = round(roc_auc_macro, 4)
        dict_performance_[idx] = dict_

    dict_overall = dict()
    for cur_metric in ['accuracy', 'f1_macro', 'f1_micro']:
        dict_ = dict()
        dict_['mean'] = round(np.mean([dict_performance_[i][cur_metric] for i in range(5)]), 4)
        dict_['std'] = round(np.std([dict_performance_[i][cur_metric] for i in range(5)]), 4)
        dict_overall[cur_metric] = dict_

    if classifier in ['SVM', 'LogisticRegression']:
        for cur_metric in ['auc_micro', 'auc_macro']:
            dict_ = dict()
            dict_['mean'] = round(np.mean([dict_performance_[i][cur_metric] for i in range(5)]), 4)
            dict_['std'] = round(np.std([dict_performance_[i][cur_metric] for i in range(5)]), 4)
            dict_overall[cur_metric] = dict_

    return {"overall": dict_overall, "detailed": dict_performance_}
