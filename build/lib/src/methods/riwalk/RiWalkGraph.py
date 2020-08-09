# -*- coding: utf-8 -*-
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import random
from .utils import WALK_FILES_DIR

class RiGraph:
    def __init__(self, nx_g, params):
        self.g = nx_g
        self.until = params['until_k']
        self.num_walks = params['num_walks']
        self.walk_length = params['walk_length']
        self.workers = params['workers']
        self.flag = params['flag']

        self.num_nodes = len(self.g)
        self.degrees_ = tuple([len(self.g[_]) for _ in range(len(self.g))])

        self.rand = random.Random()

    def get_bfs_dict(self, root, until):
        last_layer = {root}
        node_layer_dict = {0: last_layer}
        visited = {root}
        current_layer = 1
        while True:
            next_set = set().union(*[self.g[node] for node in last_layer])
            next_set.difference_update(visited)
            if not next_set:
                return node_layer_dict, visited
            next_set = tuple(next_set)
            node_layer_dict[current_layer] = next_set
            visited.update(next_set)
            if current_layer == until:
                return node_layer_dict, visited
            last_layer = next_set
            current_layer += 1

    def get_wl_dict(self, until, node_layer_dict, nei_nodes):
        wl_list = [[0] * (until + 1) for _ in range(self.num_nodes)]
        layer_list = [0] * self.num_nodes
        for layer, js in node_layer_dict.items():
            for j in js:
                layer_list[j] = layer
                for l in self.g[j]:  # time consuming
                    wl_list[l][layer] += 1
        return [layer_list[node] for node in nei_nodes], [wl_list[node] for node in nei_nodes]

    def get_sp_dict(self, node_layer_dict, nei_nodes):
        layer_list = [0] * self.num_nodes
        for layer, ks in node_layer_dict.items():
            for k in ks:
                layer_list[k] = layer
        return [layer_list[node] for node in nei_nodes], [self.degrees_[node] for node in nei_nodes]

    def simulate_walk(self, walk_length, start_node, nei_nodes_set, rand):
        walk = [start_node]
        g = self.g
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = g[cur]  # tuple(set(g[cur])&nei_nodes_set)
            if cur_nbrs:
                while True:
                    next_node = rand.choice(cur_nbrs)
                    if next_node in nei_nodes_set:
                        break
                walk.append(next_node)
            else:
                break
        return walk

    def simulate_walks_for_node(self, node, num_walks, walk_length, nei_nodes_set, rand):
        walks = []
        for walk_iter in range(num_walks):
            walk = self.simulate_walk(walk_length=walk_length, start_node=node,
                                      nei_nodes_set=nei_nodes_set, rand=rand)
            walks.append(walk)
        return walks

    def process_random_walks(self):
        until, num_walks, walk_length = self.until, self.num_walks, self.walk_length
        vertices = np.random.permutation(self.num_nodes).tolist()
        parts = self.workers
        chunks = partition(list(vertices), parts)
        futures = {}
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            part = 1
            for c in chunks:
                job = executor.submit(process_random_walks_chunk, self, c, part,
                                      until, num_walks, walk_length)
                futures[job] = part
                part += 1
            for job in as_completed(futures):
                job.result()
        return


def get_ri_walks(walks, start_node, ri_dict):
    ri_walks = []
    ri_dict[start_node] = start_node
    for walk in walks:
        ri_walk = [ri_dict[x] for x in walk]
        ri_walks.append(ri_walk)
    return ri_walks


def simple_log2(x):
    return x.bit_length() - 1


def save_random_walks(walks, part, i):
    indexes = np.random.permutation(len(walks)).tolist()
    with open('lib/RiWalk/walks/__random_walks_{}_{}.txt'.format(part, i), 'w') as f:
        for i in indexes:
            walk = walks[i]
            f.write(u"{}\n".format(u" ".join(str(v) for v in walk)))


def process_random_walks_chunk(rigraph, vertices, part_id, until, num_walks, walk_length):
    walks_all = []
    i = 0
    rand = rigraph.rand
    for count, v in enumerate(vertices):
        node_layer_dict, nei_nodes_set = rigraph.get_bfs_dict(v, until)
        nei_nodes = tuple(nei_nodes_set)
        walks = rigraph.simulate_walks_for_node(v, num_walks, walk_length, nei_nodes_set, rand)

        if 'sp' == rigraph.flag:
            layer_list, degree_list = rigraph.get_sp_dict(node_layer_dict, nei_nodes)
            root_degree = simple_log2(rigraph.degrees_[v] + 1)
            degree_list = np.log2(np.asarray(degree_list) + 1).astype(np.int32).tolist()
            sp_dict = {node_: hash((root_degree, layer_, degree_)) for node_, layer_, degree_ in
                       zip(nei_nodes, layer_list, degree_list)}
            sp_walks = get_ri_walks(walks, v, sp_dict)
            walks_all.extend(sp_walks)

        if 'wl' == rigraph.flag:
            layer_list, wl_lists = rigraph.get_wl_dict(until, node_layer_dict, nei_nodes)
            wl_lists = np.log2(np.asarray(wl_lists) + 1).astype(np.int32).tolist()
            v_wl_list = tuple(wl_lists[nei_nodes.index(v)])
            wl_dict = {node_: hash((v_wl_list, layer_, tuple(wl_))) for node_, layer_, wl_ in
                       zip(nei_nodes, layer_list, wl_lists)}
            wl_walks = get_ri_walks(walks, v, wl_dict)
            walks_all.extend(wl_walks)

        if len(walks_all) > 100000:
            save_random_walks(walks_all, part_id, i)
            i += 1
            walks_all = []
    save_random_walks(walks_all, part_id, i)


# https://github.com/leoribeiro/struc2vec
def partition(lst, n):
    division = len(lst) / float(n)
    return [lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n)]
