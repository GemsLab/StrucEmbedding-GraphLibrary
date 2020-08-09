from collections import namedtuple
from typing import List
from os import path
from ..exceptions import UnimplementedException

# TODO: add more buildtin datasets below
# FIXME: fix the current relative path into remote file url
SAMPLE_DATA_PATH = path.join(path.dirname(__file__), '../../sample-data')

# the Data type for all supported data sets
Dataset = namedtuple('Dataset', ['name', 'description', 'format', 'src_url'])

# the base class for all new data providers
class BaseDataProvider(object):
    
    def provideId(self) -> str:
        raise UnimplementedException(
            "Please implement the provideId() method to register the unique id for your datasets")

    def provideDatasets(self) -> List[Dataset]:
        raise UnimplementedException(
            "Please implement the provideDatasets() method for registering dataset details")
