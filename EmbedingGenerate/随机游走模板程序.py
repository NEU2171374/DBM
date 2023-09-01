import numpy as np
import networkx as nx
import itertools
import math
import random
from joblib import Parallel, delayed


def partition_num(num, workers):
    if num % workers == 0:
        return [num // workers] * workers
    else:
        return [num // workers] * workers + [num % workers]


class DeepWalk:
    def __init__(self, graph, walk_length, num_walks, workers=1):
        self.graph = graph

        self.walker = RandomWalker(graph)

        self.sentences = self.walker.simulate_walks(
            num_walks=num_walks, walk_length=walk_length, workers=workers, verbose=1)

        print(self.sentences)


class RandomWalker:
    def __init__(self, G, ):
        self.G = G

    def simulate_walks(self, num_walks, walk_length, workers=1, verbose=0):

        G = self.G

        nodes = list(G.nodes())

        results = Parallel(n_jobs=workers, verbose=verbose, )(
            delayed(self._simulate_walks)(nodes, num, walk_length) for num in
            partition_num(num_walks, workers))

        walks = list(itertools.chain(*results))

        return walks

    def _simulate_walks(self, nodes, num_walks, walk_length, ):
        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            for v in nodes:
                walks.append(self.deepwalk_walk(walk_length=walk_length, start_node=v))

        return walks

    def deepwalk_walk(self, walk_length, start_node):
        # 如何走改这里

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(self.G.neighbors(cur))
            if len(cur_nbrs) > 1:
                walk.append(random.choice(cur_nbrs))
            else:
                break
        return walk


if __name__ == "__main__":
    G = nx.read_edgelist('../数据集/数据清洗MovieLens数据集/train.csv', delimiter=',', create_using=nx.Graph(),
                         nodetype=None, data=[('weight', int)])

    model = DeepWalk(G, walk_length=5, num_walks=1, workers=5)

    #  num_walks=80
