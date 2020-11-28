from ..exceptions import *
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

def visualize_classification(list_results, metric="f1_macro", error_bar=True, rotation_deg=45):
    """
    Visualize the result for classification using bar plot.

    Arguments:
    list_results {list of tuples} --  [("method_name", dict_result)], where the dict_result is the returned dict
                                      from the perform_classification() function
    metric -- The metric to use for making the plot. Please choose from ["accuracy", "f1_macro", "f1_micro", "f1_macro", "auc_micro", "auc_macro"]
    error_bar -- bool. Whether to include the error bar in the plot.
    rotation_deg -- int. The rotation degree for the xtick labels
    """

    # Check that metric belongs to the classification results
    if metric not in ["accuracy", "f1_macro", "f1_micro", "f1_macro", "auc_micro", "auc_macro"]:
        raise MethodKeywordUnAllowedException("Please choose metric from [accuracy, f1_macro, f1_micro, f1_macro, auc_micro, auc_macro].")

    if len(list_results) == 0:
        raise InputFormatErrorException("Input length 0!")

    for cur_item in list_results:
        if (len(cur_item) != 2):
            raise InputFormatErrorException("Please input the results as list of tuples, i.e. [(\"method_name\", dict_result)]")

        if (not isinstance(cur_item[0], str)) or (not isinstance(cur_item[1], dict)):
            raise InputFormatErrorException("Please input the results as list of tuples, i.e. [(\"method_name\", dict_result)]")

        if "overall" not in cur_item[1]:
            raise InputFormatErrorException("Invalid input. Please make sure that the input result is generated from perform_classification() or perform_clustering()")

    
    list_methods = [i[0] for i in list_results]
    list_evaluation = [i[1]['overall'][metric]['mean'] for i in list_results]
    list_error = [i[1]['overall'][metric]['std'] for i in list_results]
    plt.close()
    plt.figure(figsize=(8, 2.5), dpi=300)
    plt.style.use('ggplot')
    if error_bar:
        plt.bar(list_methods,
                list_evaluation,
                yerr = list_error,
                capsize=10)
    else:
        plt.bar(list_methods,
                list_evaluation,
                capsize=10)
    plt.ylim(0, 1)
    plt.ylabel(metric)
    plt.xlabel("Methods")
    plt.xticks(rotation=rotation_deg)
    plt.show()


def visualize_clustering(list_results, metric="purity", rotation_deg=45):
    """
    Visualize the result for clustering using bar plot.

    Arguments:
    list_results {list of tuples} --  [("method_name", dict_result)], where the dict_result is the returned dict
                                      from the perform_clustering() function
    metric -- The metric to use for making the plot. Please choose from ["purity", "nmi"]
    rotation_deg -- int. The rotation degree for the xtick labels
    """

    # Check that metric belongs to the classification results
    if metric not in ["purity", "nmi"]:
        raise MethodKeywordUnAllowedException("Please choose metric from [purity, nmi].")

    if len(list_results) == 0:
        raise InputFormatErrorException("Input length 0!")

    for cur_item in list_results:
        if (len(cur_item) != 2):
            raise InputFormatErrorException("Please input the results as list of tuples, i.e. [(\"method_name\", dict_result)]")

        if (not isinstance(cur_item[0], str)) or (not isinstance(cur_item[1], dict)):
            raise InputFormatErrorException("Please input the results as list of tuples, i.e. [(\"method_name\", dict_result)]")

        if "overall" not in cur_item[1]:
            raise InputFormatErrorException("Invalid input. Please make sure that the input result is generated from perform_classification() or perform_clustering()")

    
    list_methods = [i[0] for i in list_results]
    list_evaluation = [i[1]['overall'][metric] for i in list_results]
    plt.close()
    plt.figure(figsize=(8, 2.5), dpi=300)
    plt.style.use('ggplot')
    
    plt.bar(list_methods,
            list_evaluation,
            capsize=10)
    plt.ylim(0, 1)
    plt.ylabel(metric)
    plt.xlabel("Methods")
    plt.xticks(rotation=rotation_deg)
    plt.show()

