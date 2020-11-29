from collections import namedtuple
from typing import List
from os import path
import networkx as nx

import requests
from requests_file import FileAdapter

from ..exceptions import UnimplementedException

# the Data type for all supported data sets
DatasetInfo = namedtuple('DatasetInfo', ['name', 'description', 'src_url', 'label_url'])

# the base class for all new data providers
class BaseDataset(object):

    def get_id(self) -> str:
        raise UnimplementedException(
            "Please implement the get_id() method to register the unique id for your datasets")

    def get_datasets(self) -> List[DatasetInfo]:
        raise UnimplementedException(
            "Please implement the get_datasets() method for registering dataset details")

    def load_dataset(self, dataset: DatasetInfo, **kwargs) -> nx.Graph:
        raise UnimplementedException(
            "Please implement the load_dataset() method for convert the dataset into graph")

    def load_label(self, dataset: DatasetInfo) -> dict:
        raise UnimplementedException(
            "Please implement the load_label() method to load the corresponding label file")


    def _fetch_url(self, url: str):
        """
        Convenient method to load data from url (including file-base)

        Args:
            url (str): HTTP Url or File Url

        Returns:
            str: Raw data
        """
        s = requests.Session()
        s.mount('file://', FileAdapter())
        return requests.get(url).text
