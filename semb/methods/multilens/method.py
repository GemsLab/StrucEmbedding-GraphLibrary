from semb.methods import BaseMethod
from .util import *
from .main import *


class Method(BaseMethod):

    __PARAMS__ = dict(dim=128, L=2, base=4, operators=[
                      'mean', 'var', 'sum', 'max', 'min', 'L1', 'L2'])

    def get_id(self):
        return "multilens"

    def train(self):
        directed = True
        base_features = ['row', 'col', 'row_col']
        dim = self.params['dim']
        L = self.params['L']
        num_buckets = self.params['base']
        op = self.params['operators']

        dict_id_idx = dict()
        dict_idx_id = dict()

        raw_ = list()
        cur_count_ = 0
        for cur_edge in self.graph.edges():
            src = cur_edge[0]
            dst = cur_edge[1]
            if src not in dict_id_idx:
                dict_id_idx[src] = cur_count_
                dict_idx_id[cur_count_] = src
                cur_count_ += 1
            if dst not in dict_id_idx:
                dict_id_idx[dst] = cur_count_
                dict_idx_id[cur_count_] = dst
                cur_count_ += 1
            raw_ += [[dict_id_idx[src], dict_id_idx[dst]]]

        raw = np.array(raw_)
        COL = raw.shape[1]

        if COL < 2:
            sys.exit('[Input format error.]')
        elif COL == 2:
            print('[unweighted graph detected.]')
            rows = raw[:,0]
            cols = raw[:,1]
            weis = np.ones(len(rows))

        elif COL == 3:
            print('[weighted graph detected.]')
            rows = raw[:,0]
            cols = raw[:,1]
            weis = raw[:,2]

        check_eq = True
        max_id = int(max(max(rows), max(cols)))
        num_nodes = max_id + 1

        nodes_to_embed = range(int(max_id)+1)

        if max(rows) != max(cols):
            rows = np.append(rows,max(max(rows), max(cols)))
            cols = np.append(cols,max(max(rows), max(cols)))
            weis = np.append(weis, 0)
            check_eq = False

        adj_matrix = sps.lil_matrix( sps.csc_matrix((weis, (rows, cols))))

        CAT_DICT = defaultdict(set)
        ID_CAT_DICT = dict()
        for i in range(num_nodes):
            CAT_DICT[1].add(i)
            ID_CAT_DICT[i] = 1
        unique_cat = [1]

        ######################################################
        # Multi-Lens starts.
        ######################################################

        g_sums = []

        neighbor_list = construct_neighbor_list(adj_matrix, nodes_to_embed)
        neighbor_list_r = construct_neighbor_list(adj_matrix.T, nodes_to_embed)

        graph = Graph(adj_matrix = adj_matrix, max_id = max_id, num_nodes = num_nodes, base_features = base_features,
            neighbor_list = neighbor_list, directed = directed, cat_dict = CAT_DICT, id_cat_dict = ID_CAT_DICT, unique_cat = unique_cat, check_eq = check_eq)

        rep_method = RepMethod(method = "hetero", bucket_max_value = 30, num_buckets = num_buckets, operators = op, use_total = len(op))

        ########################################
        # Step 1: get base features
        ########################################
        init_feature_matrix = get_init_features(graph, base_features, nodes_to_embed)
        init_feature_matrix_seq = get_seq_features(graph, rep_method, input_dense_matrix = init_feature_matrix, nodes_to_embed = nodes_to_embed)


        Kis = get_Kis(init_feature_matrix_seq, dim, L)
        # print Kis
        
        feature_matrix_emb, g_sum = feature_layer_evaluation_embedding(graph, rep_method, feature_matrix = init_feature_matrix_seq, k = Kis[0])

        g_sums.append(g_sum)

        ########################################
        # Step 2: feature proliferation.
        # layer 0 is the base feature matrix
        # layer 1+: are the layers of higher order
        ########################################

        rep = feature_matrix_emb
        feature_matrix = init_feature_matrix

        for i in range(L):
            print('[Current layer]', str(i))
            print('[feature_matrix shape]', str(feature_matrix.shape))

            feature_matrix_new = search_feature_layer(graph, rep_method, base_feature_matrix = feature_matrix)
            feature_matrix_new_seq = get_seq_features(graph, rep_method, input_dense_matrix = feature_matrix_new, nodes_to_embed = nodes_to_embed)
            feature_matrix_new_emb, g_new_sum = feature_layer_evaluation_embedding(graph, rep_method, feature_matrix = feature_matrix_new_seq, k = Kis[i+1])

            feature_matrix = feature_matrix_new
            rep_new = feature_matrix_new_emb
            rep = np.concatenate((rep, rep_new), axis=1)

            g_sums.append(g_new_sum)

        N, K = rep.shape
        self.embeddings = dict()
        for i in range(N):
            self.embeddings[dict_idx_id[i]] = rep[i, :].tolist()