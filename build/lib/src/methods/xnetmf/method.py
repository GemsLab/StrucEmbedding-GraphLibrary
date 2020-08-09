from ..method import BaseMethod
from .internal.xnetmf import *
from .internal.config import *

class Method(BaseMethod):
    
    __PARAMS__ = dict(dim=128, max_layer=2, discount=0.1, gamma=1)

    def getId(self):
        return "xnetmf"

    def train(self):
        # learn representations with xNetMF.  Can adjust parameters (e.g. as in REGAL)
        rep_method = RepMethod(max_layer=self.params['max_layer'], p=self.params['dim'], 
                              alpha=self.params['discount'], gammastruc=self.params['gamma']) 
        # FIXME: this doesnt look like a standard embeddings format
        self.embeddings = get_representations(self.graph, rep_method)

