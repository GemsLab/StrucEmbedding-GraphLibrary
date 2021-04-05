# The Structural EMBedding graph library (SEMB)

**Authors: GEMS Lab Team @ University of Michigan** ([Mark Jin](https://mark-jin.com), Ruowang Zhang, Mark Heimann)

This SEMB library allows fast onboarding to get and evaluate structural node embeddings. With the unified API interface and the modular codebase, SEMB library enables easy intergration of 3rd-party methods and datasets.

The library itself has already included a set of popular methods and datasets ready for immediate use.

- Built-in methods: [node2vec](https://github.com/aditya-grover/node2vec), [struc2vec](https://github.com/leoribeiro/struc2vec), [GraphWave](https://github.com/snap-stanford/graphwave), [xNetMF](https://github.com/GemsLab/REGAL), [role2vec](https://github.com/benedekrozemberczki/role2vec), [DRNE](https://github.com/tadpole/DRNE), [MultiLENS](https://github.com/GemsLab/MultiLENS), [RiWalk](github.com/maxuewei2/RiWalk), [SEGK](https://github.com/giannisnik/segk), (more methods to add in the near future)

- Built-in datasets: 

  | Dataset                                                      | # Nodes | # Edges |
  | ------------------------------------------------------------ | ------- | ------- |
  | [BlogCatalog](http://snap.stanford.edu/node2vec/)            | 10,312  | 333,983 |
  | [Facebook](http://snap.stanford.edu/data/egonets-Facebook.html) | 4,039   | 88,234  |
  | [ICEWS](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/QI2T9A) | 1,255   | 1,414   |
  | [PPI](snap.stanford.edu/graphsage/)                          | 56,944  | 818,786 |
  | [BR air-traffic](https://github.com/leoribeiro/struc2vec/tree/master/graph) | 131     | 1,038   |
  | [EU air-traffic](https://github.com/leoribeiro/struc2vec/tree/master/graph) | 399     | 5,995   |
  | [US air-traffic](https://github.com/leoribeiro/struc2vec/tree/master/graph) | 1,190   | 13,599  |
  | [DD6](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets) | 4,152   | 20,640  |
  | Synthetic Datasets                                           |         |         |

The library requires *Python 3.7* for best usage. In *Python 3.8*, the Tensorflow 1.14.0 used in DRNE might not be successfully installed.

## Installation and Usage

Make sure you are using *Python 3.6+* for all below!

1. First, creat a virtual environment and activate the virtual environment.

   ```bash
   python3 -m venv <VENV_NAME>
   source <VENV_NAME>/bin/activate
   ```

2. Change directory to the `StrucEmbeddingLibrary` and install the dependencies

   ```bash
   (<VENV_NAME>) cd StrucEmbeddingLibrary
   (<VENV_NAME>) python3 -m pip install -r requirements.txt --no-cache-dir
   ```

3. Install the `SEMB` package

   ```bash
   (<VENV_NAME>) cd StrucEmbeddingLibrary
   (<VENV_NAME>) python3 setup.py install
   ```

   After installation, we  highly recommend you go through our [Tutorial](https://github.com/GemsLab/StrucEmbeddingLibrary/blob/master/Tutorial.ipynb) to see how SEMB library works.

4. To enable using the jupyter notebook, do the following,

   ```bash
   (<VENV_NAME>) python3 -m pip install ipykernel --no-cache-dir
   (<VENV_NAME>) python3 -m ipykernel install --name=<VENV_NAME>
   (<VENV_NAME>) jupyter notebook
   ```

   Choose `<VENV_NAME>` at the top right corner of the page when creating a new jupyter notebook / running the tutorial notebook.



## Extending SEMB

First make sure the `semb` library is installed.

### Developing 3rd party Dataset extension

Currently, SEMB only supports embedding and evaluation on *undirected* and *unweighted* graphs.

- Create a Python 3.7+ [package](https://packaging.python.org/tutorials/packaging-projects/) with a name in form at `semb/datasets/[$YOUR_CHOSEN_DATASET_ID]`
- Within the package root directory, make sure `__init__.py` is present
- Create a `dataset.py` and make a `Method` class that inherits from `from semb.datasets import BaseDataset` and implement the required methods. See `semb/datasets/airports/dataset.py` for more details.
  - To use the built-in `load_dataset()`method, we accept the graph edgelist with the following format
    - `<Node1_id (int)> <Blank> <Node2_id (int)> <\n>`
    - Otherwise, you can overload and implement your own `load_dataset()` function. Please make sure that the returned graph is of `networkx.classes.graph.Graph` datatype. 
  - If the dataset is accompanied by the label file, to use the built-in `load_label()` function, we accept the label file with the following format
    - `<Node_id (int)> <delimeter> <Node_label (int)>`
    - Otherwise, you can overload and implement your own `load_label()` function. Please make sure that the returned type is python built-in `dict()` with the key as `<Node_id (int)>` and value as `<Node_label (int)>`
- Install the package via `setup.py` or pip.
- Now the dataset is loadable by the main client program that uses `semb`!

### Developing 3rd party Method extension

- Create a Python 3.7+ [package](https://packaging.python.org/tutorials/packaging-projects/) with a name in form of `semb/methods/[$YOUR_CHOSEN_METHOD_ID]`
- Within the package root directory, make sure `__init__.py` is present
- Create a ` method.py` and make a `Method` class that inherits from `from semb.methods import BaseMethod` and implement the required methods. See `semb/methods/node2vec/method.py` for more details.
  - Please make sure that your implemented method accepts `networkx.classes.graph.Graph` as input.
  - Please make sure that when `train()` is called, the `self.embeddings` should be a Python built-in `dict()` with key as `<Node_id (int)>` and value(embedding) as `<List (float)>`.
- Install the package via `setup.py` or pip.
- Now the method is load-able by the main client program that uses `semb`!

### Note
For both `dataset` and `method` extensions, make sure the `get_id()` to be overridden and returns the same id as your chosen id in your package name.

### Contact

If you encounter any question using our SEMB library, feel free to raise an issue or send an email to [kinmark@umich.edu](kinmark@umich.edu). Go Blue!
