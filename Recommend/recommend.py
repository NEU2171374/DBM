import numpy as np
import pandas as pd
import time
import para
import sys
#sys.stdout = open('log.txt', 'a')

global recordPath
recordPath = "实验记录.txt"

global endStr
endStr = '--------------------------------------end--------------------------------------'

class recommend:

    def __init__(self, top_N=10, top_people=10, alpha=0.5, beta=0.5):
        # alpha 人员深度广度修正系数 , beta=0.5 电影深度广度修正系数

        # print('[recommend init] top_n = '+str(top_N)+', top_people = '+str(top_people))
        self.top_N = top_N
        self.top_people = top_people
        # 每个人推荐的人员数量
        self.alpha = alpha
        # 人员loss= alpha*深度+（1-alpha）*广度
        self.beta = beta
        # 电影loss= beta*深度+（1-beta）*广度

    def generatenum(self, path, strategy, set):

        if strategy == 2:
            # 表明这是一个敏感性测试工具
            if set == 10000 :
                self.goal_peoplelist = ['692', '235', '2185', '3462', '3255',
                                        '1800', '1015', '53', '3808', '2015',
                                        '5072', '1337', '3914', '5518', '5565',
                                        '4784', '877', '2967', '4298', '5345',
                                        '5845', '4790', '1182', '5809', '4084',
                                        '1142', '3970', '1396', '4747', '4078',
                                        '2575', '5691', '462', '563', '78',
                                        '6030', '4323', '4685', '3898', '3482', ]
                # MovieLens 测试人员列表

            if set == 100000 :
                self.goal_peoplelist = ['38239', '14990', '55231', '18191', '68835',
                                        '64425', '3213', '49683', '67320', '12690',
                                        '68930', '12168', '58840', '2511', '3668',
                                        '44476', '1823', '62583', '54599', '15935',
                                        '25579', '68468', '16557', '53654', '41428',
                                        '11783', '60690', '37058', '45781', '41455',
                                        '18349', '68999', '17877', '35033', '29780',
                                        '11353', '25710', '35023', '12730', '5599',
                                        ]
                # MovieTweetings 测试人员列表


        else:
            list = []
            df = pd.read_csv(path, header=None, index_col=None, encoding='utf-8', delimiter=",", engine='python')

            peoplelist = df[0].unique()
            print('测试长度：', len(peoplelist))

            for i in range(len(peoplelist)):
                a = peoplelist[i]
                list.append(str(a))

            self.goal_peoplelist = list

    def generatenum02(self, ):
        # 去掉了评分十次以下的数据
        list = []

        df = pd.read_csv('../data/splitdata/testset.csv', header=None, index_col=None, encoding='utf-8',
                         delimiter=",", engine='python')

        df02 = pd.read_csv('../data/splitdata/occurrences.csv', header=None, index_col=0, encoding='utf-8',
                           delimiter=",", engine='python')

        peoplelist = df[0].unique()

        for i in range(len(peoplelist)):
            a = peoplelist[i]
            if df02.loc[a, 1] < 11:
                continue
            list.append(str(a))

        print('测试长度：', len(list))
        print(list)

        return list

    def readdic(self, pathlist):

        start = time.clock()
        print("读取数据开始！！！")

        self.people_deep_vector = np.load(pathlist[0]).item()
        self.people_scope_vector = np.load(pathlist[1]).item()
        self.film_deep_vector = np.load(pathlist[2]).item()
        self.film_scope_vector = np.load(pathlist[3]).item()

        train = pd.read_csv(pathlist[4], header=None, encoding='utf-8', delimiter=",", engine='python')

        train_people_film = {}

        train_people = train[0].unique()

        for people in train_people:

            if people in train_people_film:
                continue
            else:
                pmlist = train[train[0] == people]
                film_temp = []
                for ind, row in pmlist.iterrows():
                    film_temp.append(row[1])

                train_people_film[people] = film_temp

        self.train = train_people_film
        # self.train {people:[film ...]}

        test = pd.read_csv(pathlist[5], header=None, encoding='utf-8', delimiter=",", engine='python')
        test_people_film = {}

        test_people = test[0].unique()

        for people in test_people:

            if people in test_people_film:
                continue
            else:
                pmlist = test[test[0] == people]
                film_temp = []
                for ind, row in pmlist.iterrows():
                    film_temp.append(row[1])

                test_people_film[people] = film_temp

        self.test = test_people_film
        # self.test {people:[film ...]}

        end = time.clock()

        print("数据读取完成，生成训练集合和测试集合，用时： %f s" % (end - start))

    def rec1(self, set):

        start = time.clock()
        count = 1
        sumhit = 0
        sumtestlen = 0
        sumrecommend_len = 0

        list = self.goal_peoplelist

        #fo = open(recordPath, "a")
        for person in list:

            peoplelist = self.loss_topN(person, set)
            # 返回推荐top人表 [ ] 转换完了int

            people = int(person)
            refer_movie_set = self.people_refer_film(people, peoplelist)
            # 推荐相似用户的关联电影

            topNfilm = self.rec_movie_topN(people, refer_movie_set)

            hit, testlen, recommend_len = self.evaluate(people, topNfilm)

            sumhit += hit
            sumtestlen += testlen
            sumrecommend_len += recommend_len
            count += 1

            if count % 5 == 0:
                print('完成%d人' % count)
                print('hit:%d,testlen:%d,recommendlen:%d' % (sumhit, sumtestlen, sumrecommend_len))
                precision = sumhit / sumrecommend_len
                recall = sumhit / sumtestlen
                F1 = 2 * precision * recall / (precision + recall)
                print('precision:%8.7f,recall:%8.7f,F1:%8.7f' % (precision, recall, F1))

        print('-------------总计-------------')
        print('交集:%d,测试集：%d,推荐数量：%d' % (sumhit, sumtestlen, sumrecommend_len))

        endall = time.clock()
        print("目标人群生成相似人员完成，总用时： %f s" % (endall - start))

    def people_refer_film(self, goal_person, people_list):

        goal_trainfilm = self.train[goal_person]

        recommend_trainfilm = []

        for sim_people in people_list:

            for movie in self.train[sim_people]:

                if movie in goal_trainfilm:
                    continue
                elif movie in recommend_trainfilm:
                    continue
                else:
                    recommend_trainfilm.append(movie)

        return recommend_trainfilm

    def rec_movie_topN(self, people, refer_movie_set):
        # 涉及到p2p计算，都需要转换为str
        goal_film_set = self.train[people]

        goal_film_set = list(map(str, goal_film_set))
        rec_movie_set = list(map(str, refer_movie_set))

        film_deep_vector = self.film_deep_vector
        film_scope_vector = self.film_scope_vector

        maxloss = []
        beta = self.beta

        for mv in rec_movie_set:
            # mv 拟推荐电影
            loss = []
            for train_film in goal_film_set:
                temp = self.similar_ratio_loss(film_deep_vector[mv], film_deep_vector[train_film],
                                               film_scope_vector[mv], film_scope_vector[train_film], beta)
                # 第一参数是深度，第二参数是目标深度，第三参数是广度，第四参数是目标广度
                loss.append(temp)

            maxloss.append(max(loss))

        topNfilm = self.selectTopN(top=self.top_N, loss=maxloss, people_num=rec_movie_set)
        # 返回的内容已转为int 电影

        # print(topNfilm)

        return topNfilm

    def loss_topN(self, goal_person, set):
        """
        获取损失最小的相似人员Top
        :param people: 被测试人员 String
        :return: 人员列表，int类型[ ]
        """

        people_deep_vector = self.people_deep_vector
        people_scope_vector = self.people_scope_vector
        alpha = self.alpha

        loss = []
        people_number = []

        for i, idx in enumerate(people_scope_vector):

            if idx == goal_person:
                continue
            if set == 100000:
                if int(idx) > 100000:
                    print('[loss_topN] set == 100000')
                    continue
            elif set == 10000:
                if int(idx) > 10000:
                    print('[loss_topN] set == 10000')
                    continue

            temp = self.similar_ratio_loss(people_deep_vector[idx], people_deep_vector[goal_person],
                                           people_scope_vector[idx], people_scope_vector[goal_person],
                                           alpha)
            # 第一参数是深度，第二参数是目标深度，第三参数是广度，第四参数是目标广度

            loss.append(temp)
            people_number.append(idx)
            # 返回的loss是int ,people_number 元素是 string

        topNpeople = self.selectTopN(top=self.top_people, loss=loss, people_num=people_number)
        # int类型

        return topNpeople

    def cos_sim(self, vector_a, vector_b):
        """
        计算两个向量的余弦相似度
        :param vector_a:
        :param vector_b:
        :return:
        """
        vector_a = np.mat(vector_a)
        vector_b = np.mat(vector_b)
        num = float(vector_a * vector_b.T)
        denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        sim = num / denom
        return sim

    def similar_ratio_loss(self, deep_a, goal_deep_b, scope_c, goal_scope_d, ratio):
        # 第一参数是深度，第二参数是目标深度，第三参数是广度，第四参数是目标广度

        vector_a = np.mat(deep_a)
        vector_b = np.mat(goal_deep_b)
        num1 = float(vector_a * vector_b.T)
        denom1 = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        sim1 = num1 / denom1

        vector_a = np.mat(scope_c)
        vector_b = np.mat(goal_scope_d)
        num2 = float(vector_a * vector_b.T)
        denom2 = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        sim2 = num2 / denom2

        sim = ratio * sim1 + (1 - ratio) * sim2
        return sim

    def selectTopN(self, top, loss, people_num, ):
        """
        传入全部人员的损失，和人员列表
        :param loss:
        :param people_num:
        :return: 损失最小的topN人员，int
        """

        t = np.argsort(np.array(loss))
        # 返回的是人员索引,损失越大越好
        t = t[-top:]
        selectToppeople = []
        for i in t:
            selectToppeople.append(people_num[i])

        selectToppeople = list(map(int, selectToppeople))

        return selectToppeople

    def evaluate(self, people, film):
        """
        评估参数
        :param people: 目标人员int
        :param film: 电影推荐列表int
        :return:
        """

        test = self.test
        # 获取test中的人看过的电影{}

        test_film_list = test[people]
        # test中看过的电影列表

        testlenth = len(test_film_list)
        # 测试集中看过电影的长度

        hit = 0

        for movie in film:
            if movie in test_film_list:
                hit += 1

        # print("命中个数： %d,  测试集中看过的电影： %d,  推荐电影数量： %d " %
        #       (hit, testlenth, len(film)))

        # s2 = time.clock()
        # print("耗费时间： %f s" % (s2 - s1))
        # print()
        return hit, testlenth, len(film)

def run(top_n, top_p, s, breadth_walk_length, depth_walk_length=10, embed_size=128):
    #try:
    top_N = top_n
    top_people = top_p
    t = recommend(top_N, top_people, alpha=0.7, beta=0.7) # 0.5 0.5
    strr = '[run] 推荐参数：top_N = '+str(top_n)+", top_people = "+str(top_p)+\
          ', s = '+str(s)+', depth_walk_length = '+str(depth_walk_length)+\
          ', breadth_walk_length = '+str(breadth_walk_length)+', embed_size = '+str(embed_size)
    print(strr)
    # 人员loss= alpha*深度+（1-alpha）*广度 ，电影loss= beta*深度+（1-beta）*广度

    if s == 100000:
        dataset = '推特'
        """if breadth_walk_length < 150:
            print(endStr)
            return"""
    elif s == 10000:
        dataset = 'MovieLens'
        """if breadth_walk_length < 75:
            print(endStr)
            return"""

    t.generatenum('../数据集/数据清洗'+dataset+'数据集/test.csv', 2, s)
    # 后面系数为2表明测试程序

    p01 = '../数据集/嵌入向量/主算法/MovieLens数据集/'
    pathlist01 = [p01+'深度people_node_'+str(depth_walk_length)+'_'+str(para.depth_num_walks_default)+'_'+str(embed_size)+'_'+str(para.depth_window_size_default)+'.npy',
                  p01+'广度people_node_'+str(breadth_walk_length)+'_'+str(para.breadth_num_walks_default)+'_'+str(embed_size)+'_'+str(para.breadth_window_size_default)+'.npy',
                  p01+'深度film_node_'+str(depth_walk_length)+'_'+str(para.depth_num_walks_default)+'_'+str(embed_size)+'_'+str(para.depth_window_size_default)+'.npy',
                  p01+'广度film_node_'+str(breadth_walk_length)+'_'+str(para.breadth_num_walks_default)+'_'+str(embed_size)+'_'+str(para.breadth_window_size_default)+'.npy',
                  '../数据集/数据清洗MovieLens数据集/train.csv',
                  '../数据集/数据清洗MovieLens数据集/test.csv']

    p02 = '../数据集/嵌入向量/主算法/推特数据集/'
    pathlist02 = [p02 + '深度people_node_'+str(depth_walk_length)+'_'+str(para.depth_num_walks_default)+'_'+str(embed_size)+'_'+str(para.depth_window_size_default)+'.npy',
                  p02+'广度people_node_'+str(breadth_walk_length)+'_'+str(para.breadth_num_walks_default)+'_'+str(embed_size)+'_'+str(para.breadth_window_size_default)+'.npy',
                  p02+'深度film_node_'+str(depth_walk_length)+'_'+str(para.depth_num_walks_default)+'_'+str(embed_size)+'_'+str(para.depth_window_size_default)+'.npy',
                  p02+'广度film_node_'+str(breadth_walk_length)+'_'+str(para.breadth_num_walks_default)+'_'+str(embed_size)+'_'+str(para.breadth_window_size_default)+'.npy',
                  '../数据集/数据清洗推特数据集/train.csv',
                  '../数据集/数据清洗推特数据集/test.csv']
    # MovieTweetings 数据集的推荐

    if s == 100000:
        t.readdic(pathlist02)
    elif s == 10000:
        t.readdic(pathlist01)
    # t.readdic(pathlist01)

    t.rec1(s)

    print(endStr)
    """except Exception:
        strr = '错误: ' + str(Exception)
        print(strr)
        print(endStr)"""

def runSingle():
    top_N = para.top_N
    top_people = para.top_people
    s = para.set
    depth_walk_length = para.depth_walk_length
    breadth_walk_length = para.breadth_walk_length
    embed_size = para.embed_size
    print('[runSingle] top_n = ' + str(top_N) + ', top_people = ' + str(top_people) + ', s = ' + str(s))
    run(top_N, top_people, s,breadth_walk_length , depth_walk_length, embed_size)

def runAll():

    for s in para.set_:
        if s == 100000:
            b_w_l_default = 150
        elif s == 10000:
            b_w_l_default = 75

        """for d_w_l in para.depth_walk_length_:
            if s == 100000:
                break
            strr = '[runAll] 变量: depth_walk_length = '+ str(d_w_l)+', s = '+str(s)
            print(strr)
            run(para.top_N_default, para.top_people_default, s, b_w_l_default, d_w_l)

        for b_w_l in para.breadth_walk_length_:
            if s == 100000:
                break
            strr = '[runAll] 变量: breath_walk_length = ' + str(b_w_l) + ', s = ' + str(s)
            print(strr)
            run(para.top_N_default, para.top_people_default, s, b_w_l, para.depth_walk_length_default, para.embed_size_default)"""

        for e_s in para.embed_size_:
            if s == 100000:
                break
            strr = '[runAll] 变量: embed_size = ' + str(e_s) + ', s = ' + str(s)
            print(strr)
            run(para.top_N_default, para.top_people_default, s, b_w_l_default, para.depth_walk_length_default, e_s)

"""        for people in para.top_people_:
            strr = '[runAll] 变量: top_people = ' + str(people) + ', s = ' + str(s)
            print(strr)
            run(para.top_N_default, people, s, b_w_l_default)

        for n in para.top_N_:
            strr = '[runAll] 变量: top_n = ' + str(n) + ', s = ' + str(s)
            print(strr)
            run(n, para.top_people_default, s, b_w_l_default)"""

if __name__ == '__main__':
    """
    说明：推荐有两个方法：
    方法一（methon = 1）：产生相似性最高的top_people个用户，默认为10。用户关联的电影为备选推荐电影集合
    备选推荐电影集合与测试集中用户看过的电影，做余弦差，最大得分为最终得分，将最终得分从大到小排列，取前top_N=10推荐
    """
    runSingle()
    #runAll()
