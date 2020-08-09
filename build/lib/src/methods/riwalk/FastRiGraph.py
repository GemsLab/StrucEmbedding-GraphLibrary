# -*- coding: utf-8 -*-
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import random
import logging
from os import path
from .utils import WALK_FILES_DIR

class RiGraph:
    def __init__(self, nx_g, args):
        self.g = nx_g
        self.num_walks = args.num_walks
        self.walk_length = args.walk_length
        self.workers = args.workers
        self.flag = args.flag

        self.num_nodes = len(self.g)
        self.degrees_ = tuple([len(self.g[_]) for _ in range(len(self.g))])

        self.rand = random.Random()

    def get_dis(self, rws, start):
        dis = {start: 0}
        for walk in rws:
            for i,j in zip(walk[:-1],walk[1:]):
                nd = dis[i] + 1
                dis[j] = min(dis.get(j,999999), nd)
        return dis

    def get_sp_dict(self, node_layer_dict, nei_nodes):
        return [node_layer_dict[node] for node in nei_nodes], [self.degrees_[node] for node in nei_nodes]

    def simulate_walk(self, walk_length, start_node, rand):
        walk = [start_node]
        g = self.g
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = g[cur]  # tuple(set(g[cur])&nei_nodes_set)
            if cur_nbrs:
                next_node = rand.choice(cur_nbrs)
                walk.append(next_node)
            else:
                break
        return walk

    def simulate_walks_for_node(self, node, num_walks, walk_length, rand):
        walks = []
        for walk_iter in range(num_walks):
            walk = self.simulate_walk(walk_length=walk_length, start_node=node, rand=rand)
            walks.append(walk)
        return walks

    def process_random_walks(self):
        num_walks, walk_length = self.num_walks, self.walk_length
        vertices = np.random.permutation(self.num_nodes).tolist()
        parts = self.workers
        chunks = partition(list(vertices), parts)
        futures = {}
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            part = 1
            for c in chunks:
                job = executor.submit(process_random_walks_chunk, self, c, part,
                                      num_walks, walk_length)
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
    with open(path.join(WALK_FILES_DIR, '__random_walks_{}_{}.txt'.format(part, i)), 'w') as f:
        for i in indexes:
            walk = walks[i]
            f.write(u"{}\n".format(u" ".join(str(v) for v in walk)))


def process_random_walks_chunk(rigraph, vertices, part_id, num_walks, walk_length):
    walks_all = []
    i = 0
    rand = rigraph.rand
    for count, v in enumerate(vertices):
        walks = rigraph.simulate_walks_for_node(v, num_walks, walk_length, rand)
        node_layer_dict=rigraph.get_dis(walks,v)
        nei_nodes=list(node_layer_dict.keys())

        if 'sp' == rigraph.flag:
            layer_list, degree_list = rigraph.get_sp_dict(node_layer_dict, nei_nodes)
            root_degree = simple_log2(rigraph.degrees_[v] + 1)
            degree_list = np.log2(np.asarray(degree_list) + 1).astype(np.int32).tolist()
            sp_dict = {node_: hash((root_degree, layer_, degree_)) for node_, layer_, degree_ in
                       zip(nei_nodes, layer_list, degree_list)}
            sp_walks = get_ri_walks(walks, v, sp_dict)
            walks_all.extend(sp_walks)
        if count%10==0:
            logging.debug('worker {} process {} nodes.'.format(part_id, count))
        if len(walks_all) > 100000:
            save_random_walks(walks_all, part_id, i)
            i += 1
            walks_all = []
    save_random_walks(walks_all, part_id, i)


# https://github.com/leoribeiro/struc2vec
def partition(lst, n):
    division = len(lst) / float(n)
    return [lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n)]
