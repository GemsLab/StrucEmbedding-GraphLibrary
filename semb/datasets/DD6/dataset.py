from semb.datasets import BaseDataset, DatasetInfo

import os
import networkx as nx
from typing import List

# TODO: Make this a remote URL in the future
SAMPLE_DATA_DIR = os.path.join(os.path.dirname(__file__), "../../../sample-data/DD6")

class Dataset(BaseDataset):
    
    def get_id(self) -> str:
        return 'DD6'

    def get_datasets(self) -> List[DatasetInfo]:
        return [
            DatasetInfo(name="DD6", description="DD6 dataset", \
                    src_url=f'{SAMPLE_DATA_DIR}/DD6.edgelist', \
                    label_url=f'{SAMPLE_DATA_DIR}/DD6_label.txt')]

    def load_dataset(self, dataset: DatasetInfo, directed=False, weighted=False) -> nx.Graph:
        if weighted:
            graph = nx.read_edgelist(
                dataset.src_url, 
                nodetype=int, 
                data=(('weight', 'data')), 
                create_using=(nx.Graph() if not directed else nx.DiGraph()))
        else:
            graph = nx.read_edgelist(
                dataset.src_url,
                nodetype=int,
                create_using=(nx.Graph() if not directed else nx.DiGraph()))
            for edge in graph.edges():
                graph[edge[0]][edge[1]]['weight'] = 1

        return graph

    def load_label(self, dataset: DatasetInfo) -> dict:
        dict_labels = dict()
        dict_counter = dict()
        with open(dataset.label_url, 'r') as f:
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