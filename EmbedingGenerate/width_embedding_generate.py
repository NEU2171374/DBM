import numpy as np
import networkx as nx
import itertools
import math
import random
from joblib import Parallel, delayed
import time
from gensim.models import Word2Vec
import para
import os.path


def partition_num(num, workers):
    if num % workers == 0:
        return [num // workers] * workers
    else:
        return [num // workers] * workers + [num % workers]


class DeepWalk:

    def __init__(self, graph, walk_length, num_walks, s, node_type, workers=1):

        print('[DeepWalk_init] ' + 'walk_length=' + str(walk_length) + ', num_walks=' + str(num_walks) + ', set=' + str(
            s))
        self.graph = graph

        self.walker = RandomWalker(graph)

        self.sentences = self.walker.simulate_walks(
            num_walks=num_walks, walk_length=walk_length, s=s, node_type=node_type, workers=workers, verbose=1)

        # print(self.sentences)

    def train(self, embed_size=128, window_size=5, workers=3, iter=5, **kwargs):
        # Word2Vec 参数说明 http://www.360doc.com/content/17/0810/17/17572791_678202559.shtml
        # 嵌入维度默认200 window_size=40

        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["vector_size"] = embed_size
        kwargs["sg"] = 1  # skip gram  默认为0，对应CBOW算法
        kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax 0表示不使用 1表示使用 默认为0
        # Skip-gram 和 Hierarchical Softmax 模型常常一起用
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["epochs"] = iter

        print("Learning embedding vectors...")
        model = Word2Vec(**kwargs)
        print("Learning embedding vectors done!")

        self.w2v_model = model
        return model

    def get_embeddings(self, node_type, set):
        if self.w2v_model is None:
            print("model not train")
            return {}

        self._embeddings = {}

        if node_type == 1:  # film_node
            for word in self.graph.nodes():
                if int(word) > set:
                    # 注意 MovieLens int(word)<10000 ; 如果是 MovieTweeting int(word)<100000 就是人
                    self._embeddings[word] = self.w2v_model.wv[word]
        elif node_type == 0:  # people_node
            for word in self.graph.nodes():
                if int(word) < set:
                    # 注意 MovieLens int(word)<10000 ; 如果是 MovieTweeting int(word)<100000 就是人
                    self._embeddings[word] = self.w2v_model.wv[word]
        return self._embeddings


class RandomWalker:
    def __init__(self, G, ):
        self.G = G

    def simulate_walks(self, num_walks, walk_length, s, node_type, workers=1, verbose=0):

        G = self.G

        nodes = list(G.nodes())

        results = Parallel(n_jobs=workers, verbose=verbose, )(
            delayed(self._simulate_walks)(nodes, num, walk_length, s, node_type) for num in
            partition_num(num_walks, workers))

        walks = list(itertools.chain(*results))

        return walks

    def _simulate_walks(self, nodes, num_walks, walk_length, s, node_type):
        walks = []
        if node_type == 0:  # people_node
            for _ in range(num_walks):
                random.shuffle(nodes)
                for v in nodes:
                    if int(v) > s:
                        # 人员还是序列还是电影序列  int(v) > 9999 电影为中心，学人员节点
                        # int(v) < 10000 以人员为中心，学习周边电影
                        # 注意 MovieLens int(word)<10000 ; 如果是 MovieTweeting int(word)<100000 就是人
                        walks.append(self.deepwalk_walk(walk_length=walk_length, start_node=v))
        elif node_type == 1:  # film_node
            for _ in range(num_walks):
                random.shuffle(nodes)
                for v in nodes:
                    if int(v) < s:
                        # 人员还是序列还是电影序列  int(v) > 9999 电影为中心，学人员节点
                        # int(v) < 10000 以人员为中心，学习周边电影
                        # 注意 MovieLens int(word)<10000 ; 如果是 MovieTweeting int(word)<100000 就是人
                        walks.append(self.deepwalk_walk(walk_length=walk_length, start_node=v))

        return walks

    def deepwalk_walk(self, walk_length, start_node):
        # 如何走改这里

        cur = start_node
        cur_nbrs = list(self.G.neighbors(cur))
        random.shuffle(cur_nbrs)
        return cur_nbrs[:walk_length]


def run(node_type, s, walk_length, embed_size=128, num_walks=80, window_size=10):
    # try:
    start = time.perf_counter()
    if node_type == 1:
        nodeType = 'film'
    elif node_type == 0:
        nodeType = 'people'

    if s == 100000:
        dataset = '推特'
    elif s == 10000:
        dataset = 'MovieLens'

    tag = dataset + '数据集/广度' + nodeType + '_node_' + str(walk_length) + '_' + str(num_walks) + '_' + str(
        embed_size) + '_' + str(window_size)
    filename = '../数据集/嵌入向量/主算法_230826/' + tag

    if os.path.isfile(filename + '.npy') == True:
        print('[run] ' + tag + '已经存在')
        return

    print('[run] ' + tag)

    G = nx.read_edgelist('../数据集/数据清洗' + dataset + '数据集/train.csv', delimiter=',', create_using=nx.Graph(),
                         nodetype=None, data=[('weight', int)])

    model = DeepWalk(G, walk_length, num_walks, s, node_type, workers=5)
    model.train(embed_size, window_size, iter=4)  # iter=4

    embeddings = model.get_embeddings(node_type, s)

    np.save(filename + '.npy', embeddings)

    test02 = np.load(filename + '.npy', encoding="latin1")
    doc = open(filename + '.txt', 'a')  # 打开一个存储文件，并依次写入

    end = time.clock()
    print("总共耗费时间： %f s" % (end - start))
    """except Exception:
        print('错误: ' + str(Exception) + '; 失败参数：' + dataset + '数据集/广度' + nodeType + '_node__' + str(
            walk_length) + '_' + str(num_walks) + '_' + str(embed_size) + '_' + str(window_size))
        end = time.clock()
        print("总共耗费时间： %f s" % (end - start))"""


def runAll():
    for s in para.set_:
        for node_type in para.type_:

            """for walk_length in para.breadth_walk_length_:
                print('[runAll] walk_length'+str(walk_length)+', set='+str(s)+', node_type='+str(node_type))
                run(node_type, s, walk_length)"""

            for embed_size in para.embed_size_:
                if s == 100000:
                    b_w_l = 150
                if s == 10000:
                    b_w_l = 75
                print('[runAll] embed_size' + str(embed_size) + ', set=' + str(s) + ', node_type=' + str(node_type))
                run(node_type, s, b_w_l, embed_size)


def runSingle():
    walk_length = para.breadth_walk_length
    num_walks = para.breadth_num_walks
    embed_size = para.breadth_embed_size
    window_size = para.breadth_window_size
    node_type = para.breadth_type
    s = para.breadth_set

    print('[runSingle] walk_length=' + str(walk_length) + ', num_walks=' + str(num_walks) + ', embed_size=' + str(
        embed_size) + ', window_size=' + str(window_size) + ', set=' + str(s))
    run(node_type, s, walk_length, embed_size, num_walks, window_size)


if __name__ == "__main__":
    """
    本算法是基于元路径【广度】，跳跃取同质点，可以实现电影同质和人员同质。
    敏感性参数：嵌入维度、游走长度、窗口大小、alpha[人员系数]、beta[电影系数]
    注意修改参数：int(v) < 10000:人员节点，int(v) >= 10000:电影节点
    """

    runSingle()
