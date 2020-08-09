from ..exceptions import UnimplementedException, MethodKeywordUnAllowedException

class BaseMethod(object):
    """
    Base driver for all embedding methods
    """
    # allowed keyword arguments and their default value
    __PARAMS__ = dict()
    # custom defined constraint functions for params
    __CONSTRAINTS__ = dict()

    def __init__(self, graph, *kwargs):
        """
        Initialize a Method

        Arguments:
            graph {nx.Graph} -- NetworkX graph
        """
        self.graph = graph
        self.params = {}
        self.embeddings = None

        # check for allowed args
        for key, default_value in __PARAMS__.items():
            self.params[key] = default_value
        for key, value in kwargs.items():
            if key not in __PARAMS__:
                raise MethodKeywordUnAllowedException("Disallowed keyword argument %s" % key)
            if key in __CONSTRAINTS__ and not __CONSTRAINTS__[key](value):
                raise MethodKeywordUnAllowedException("Disallowed value %s for keyword argument %s" % (value, key))
            self.params[key] = value
    
    def getId(self) -> str:
        """
        Return an unique id for this method
        """
        raise UnimplementedException("Please implement the getId() method for indexing your method")

    def train(self):
        """
        Train the model to generate embedding data.
        Should be overridden.
        """
        raise UnimplementedException(
            "Please implement the train() method and save data into self.embeddings!")

    def get_embeddings(self):
        """
        Get the embeddings
        """
        assert(self.embeddings is not None, "Please train() first")
        return self.embeddings
