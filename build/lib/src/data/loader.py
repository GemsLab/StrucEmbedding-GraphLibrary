import networkx as nx
import requests
from requests_file import FileAdapter
from os import path

from ..exceptions import DatasetNotExistException

# repository for all data sets
DATASETS = {}

# dynamically export all included datasets 
def _find_builtin_datasets(methods):
    pass

# TODO: add support to register 3rd party extension methods automatically by searching the installed packages
def _find_external_datasets(methods):
    pass

_find_builtin_datasets(DATASETS)
_find_external_datasets(DATASETS)

def get_dataset_ids():
    global DATASETS
    return list(DATASETS.keys())

def load(dataset_id):
    """
    For a data item, loads from its source and convert to a graph

    Args:
        dataset_id (str): unique id to load the dataset from
    """
    # TODO
    pass
