# The Structural EMBedding library (SEMB)

**Authors: GEMS Lab Team @ University of Michigan**

This SEMB library allows fast onboarding to explore structural embedding of graph data using hetereogenous methods, with a unified API interface and a modular codebase enabling easy intergration of 3rd party methods and datasets.

The library itself has already included a set of popular methods and datasets ready for use immediately.

The library requires *Python 3.7+*.

## Getting started

Make sure you are using *Python 3.7+* for all below!

### Installation
`python setup.py install` (TODO: Pip support will be added soon)

### Import and load a dataset
```py
from semb.datasets import load, get_dataset_ids
# explore all datasets (both built in and extended by 3rd party)
ids = get_dataset_ids()
# load a dataset
graph = load(ids[0])
```

### Import and load a method
```py
from semb.methods import load, get_method_ids
# explore all methods (both built in and extended by 3rd party)
ids = ge_method_ids()
# load a method, returns a constructor for a method's base class
Method = load(ids[0])
# create and run a method.
# NOTE: except for the first "graph" arg, everything other argument MUST be in keyword form!
method = Method(graph, a=1, b=2, c=3, ...)
method.train()
embeddings = method.get_embeddings()
```

## Extending SEMB

First make sure the `semb` library is installed.

### Developing 3rd party Dataset extension

- Create a Python 3.7+ [package](https://packaging.python.org/tutorials/packaging-projects/) with a name in form of `semb-dataset[$YOUR_CHOSEN_DATASET_ID]`
- Within the package root directory, make sure `__init__.py` is present
- Create a `dataset.py` and make a `Method` class that inherits from `from semb.datasets import BaseDataset` and implement the required methods. See `semb/datasets/airports/dataset.py` for more details.
- Install the package via `setup.py` or pip.
- Now the dataset is loadable by the main client program that uses `semb`!

### Developing 3rd party Method extension

- Create a Python 3.7+ [package](https://packaging.python.org/tutorials/packaging-projects/) with a name in form of `semb-method[$YOUR_CHOSEN_METHOD_ID]`
- Within the package root directory, make sure `__init__.py` is present
- Create a `dataset.py` and make a `Dataset` class that inherits from `from semb.methods import BaseMethod` and implement the required methods. See `semb/methods/node2vec/method.py` for more details.
- Install the package via `setup.py` or pip.
- Now the method is load-able by the main client program that uses `semb`!

### Note
For both `dataset` and `method` extensions, make sure the `get_id()` to be overridden and returns the same id as your chosen id in your package name.
