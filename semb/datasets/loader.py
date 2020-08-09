import os, pkg_resources, importlib, re

from ..exceptions import DatasetNotExistException

# repository for all data sets
DATASETS = {}

# dynamically export all included datasets 
def _find_builtin_datasets(datasets):
    d = os.path.dirname(__file__)
    pkg = '.'.join(__name__.split('.')[:-1])
    for mod_name in os.listdir(d):
        if os.path.isdir(os.path.join(d, mod_name)) and '__' not in mod_name:
            datasets[mod_name] = f'{pkg}.{mod_name}.dataset'

def _find_external_datasets(datasets):
    regex = r"semb-dataset\[.+\]"
    for pkg in pkg_resources.working_set:
        matches = re.findall(regex, pkg.key)
        if len(matches) > 0:
            match = matches[0]
            datasets[match.split('[')[1][:-1]] = match

_find_builtin_datasets(DATASETS)
_find_external_datasets(DATASETS)

def get_dataset_ids():
    global DATASETS
    return list(DATASETS.keys())

def load(dataset_id):
    global DATASETS
    if dataset_id not in DATASETS:
        raise DatasetNotExistException()
    return importlib.import_module(DATASETS[dataset_id]).Dataset
