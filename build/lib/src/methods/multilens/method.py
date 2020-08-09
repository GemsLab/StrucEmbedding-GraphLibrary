from ..method import BaseMethod
from .util import *
from .main import *


class Method(BaseMethod):

    __PARAMS__ = dict(dim=128, L=2, base=4, operators=[
                      'mean', 'var', 'sum', 'max', 'min', 'L1', 'L2'])

    def getId(self):
        return "multilens"

    def train(self):
        dim = self.params['dim']
        L = self.params['L']
        num_buckets = self.params['base']
        op = self.params['operators']
        ######################################################
        # Multi-Lens starts.
        ######################################################
        g_sums = []
        rep_method = RepMethod(method="hetero", bucket_max_value=30,
                               num_buckets=num_buckets, operators=op, use_total=len(op))

        ########################################
        # Step 1: get base features
        ########################################
        init_feature_matrix = get_init_features(
            self.graph, base_features, nodes_to_embed)
        init_feature_matrix_seq = get_seq_features(
            self.graph, rep_method, input_dense_matrix=init_feature_matrix, nodes_to_embed=nodes_to_embed)

        Kis = get_Kis(init_feature_matrix_seq, dim, L)
        print(Kis)

        feature_matrix_emb, g_sum = feature_layer_evaluation_embedding(
            self.graph, rep_method, feature_matrix=init_feature_matrix_seq, k=Kis[0])

        g_sums.append(g_sum)

        ########################################
        # Step 2: feature proliferation.
        # layer 0 is the base feature matrix
        # layer 1+: are the layers of higher order
        ########################################

        rep = feature_matrix_emb
        feature_matrix = init_feature_matrix

        for i in range(L):
            print('[Current layer] ' + str(i))
            print('[feature_matrix shape] ' + str(feature_matrix.shape))

            feature_matrix_new = search_feature_layer(
                self.graph, rep_method, base_feature_matrix=feature_matrix)
            feature_matrix_new_seq = get_seq_features(
                self.graph, rep_method, input_dense_matrix=feature_matrix_new, nodes_to_embed=nodes_to_embed)
            feature_matrix_new_emb, g_new_sum = feature_layer_evaluation_embedding(
                self.graph, rep_method, feature_matrix=feature_matrix_new_seq, k=Kis[i+1])

            feature_matrix = feature_matrix_new
            rep_new = feature_matrix_new_emb
            rep = np.concatenate((rep, rep_new), axis=1)

            g_sums.append(g_new_sum)

        self.embeddings = g_sum
