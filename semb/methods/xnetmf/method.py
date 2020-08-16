from semb.methods import BaseMethod
from .internal.xnetmf import *
from .internal.config import *

class Method(BaseMethod):
    
    __PARAMS__ = dict(dim=128, max_layer=2, discount=0.1, gamma=1)

    def get_id(self):
        return "xnetmf"

    def train(self):
        # learn representations with xNetMF.  Can adjust parameters (e.g. as in REGAL)
        rep_method = RepMethod(max_layer=self.params['max_layer'], p=self.params['dim'], 
                              alpha=self.params['discount'], gammastruc=self.params['gamma'])
        
        graph = Graph(nx.adjacency_matrix(self.graph))
        # FIXME: best way to convert/represent it in the format of the original code?
        self.embeddings = get_representations(graph, rep_method)
