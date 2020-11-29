from semb.datasets import BaseDataset, DatasetInfo

import os
import networkx as nx
from typing import List

# TODO: Make this a remote URL in the future
SAMPLE_DATA_DIR = os.path.join(os.path.dirname(__file__), "../../../sample-data/Synthetic_Role/Large")

class Dataset(BaseDataset):
    
    def get_id(self) -> str:
        return 'Synthetic_Role'

    def get_datasets(self) -> List[DatasetInfo]:
        return [
            DatasetInfo(name="0 - Barbell Large A", description="Connecting the out-most nodes on the chain of B5 into a circle.", \
                    src_url=f'{SAMPLE_DATA_DIR}/B5_L_A.edgelist', \
                    label_url=f'{SAMPLE_DATA_DIR}/B5_L_A_label.txt'),
            DatasetInfo(name="1 - Barbell Large B", description="Connecting the out-most nodes on the chain of B5 into a circle. Additional 5-clique at each connector.", \
                    src_url=f'{SAMPLE_DATA_DIR}/B5_L_B.edgelist', \
                    label_url=f'{SAMPLE_DATA_DIR}/B5_L_B_label.txt'),
            DatasetInfo(name="2 - Ferris Wheel", description="Enlarged version of C8 with similar perturbation.", \
                    src_url=f'{SAMPLE_DATA_DIR}/C8_L.edgelist', \
                    label_url=f'{SAMPLE_DATA_DIR}/C8_L_label.txt'),
            DatasetInfo(name="3 - H10_S_L", description="10 H5 on a circle with 2 circular nodes between each connecting circular node with house’s side.", \
                    src_url=f'{SAMPLE_DATA_DIR}/H10_S_L.edgelist', \
                    label_url=f'{SAMPLE_DATA_DIR}/H10_S_L_label.txt'),
            DatasetInfo(name="4 - H10_T_L", description="10 H5 on a circle with 2 circular nodes between each connecting circular node with house’s roof.", \
                    src_url=f'{SAMPLE_DATA_DIR}/H10_T_L.edgelist', \
                    label_url=f'{SAMPLE_DATA_DIR}/H10_T_L_label.txt'),
            DatasetInfo(name="5 - PB-L", description="10 half-sided PB5 connected to each node of a 10- node circular graph. All the node degrees are 3.", \
                    src_url=f'{SAMPLE_DATA_DIR}/R10_L.edgelist', \
                    label_url=f'{SAMPLE_DATA_DIR}/R10_L_label.txt'),
            DatasetInfo(name="6 - City of Stars", description="10 normal stars and 5 binary stars as in S5.", \
                    src_url=f'{SAMPLE_DATA_DIR}/S5_10_L.edgelist', \
                    label_url=f'{SAMPLE_DATA_DIR}/S5_10_L_label.txt')]

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
        if dataset.label_url is None:
            print("Error: Current version of SEMB doesn't support multiclass classification or" \
                   + " there is currently no label file for this dataset")
            return

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
            
