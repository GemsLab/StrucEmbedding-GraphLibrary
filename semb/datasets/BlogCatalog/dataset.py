from semb.datasets import BaseDataset, DatasetInfo

import os
import networkx as nx
from typing import List

# TODO: Make this a remote URL in the future
SAMPLE_DATA_DIR = os.path.join(os.path.dirname(__file__), "../../../sample-data/BlogCatalog")

class Dataset(BaseDataset):
    
    def get_id(self) -> str:
        return 'BlogCatalog'

    def get_datasets(self) -> List[DatasetInfo]:
        return [
            DatasetInfo(name="BlogCatalog", description="BlogCatalog data", \
                    src_url=f'{SAMPLE_DATA_DIR}/BlogCatalog.edgelist')]

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
            
