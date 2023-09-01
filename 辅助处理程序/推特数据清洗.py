import pandas as pd
from sklearn.model_selection import train_test_split


def generateocc():
    df = pd.read_csv('../数据集/推特电影原数据/ratings.dat', header=None, encoding='utf-8',
                     delimiter="::", engine='python')

    df[1] = df[1] + 100000
    # 最小电影编号100008 ,只要小于100000就是人

    del df[3]
    # 删除时间戳

    df = df.drop(df[df[2] < 5].index)
    print("删除评分小于5分，表尺寸：", df.shape)  # (826685, 3)

    df.to_csv('../数据集/数据清洗推特数据集/全部评分分数大于4分.csv', index=None, header=None)

    df = pd.read_csv('../数据集/数据清洗推特数据集/全部评分分数大于4分.csv', index_col=None, header=None)

    # df1 = df[df[2] == 3] # 筛选出3分的电影
    df1 = df[0].value_counts()
    # 数据透视，电影评分3分的电影-人数

    df1.to_csv('../数据集/数据清洗推特数据集/评分大于4分出现次数.csv', index=True, header=None)


def splitdata():
    df = pd.read_csv('../数据集/数据清洗推特数据集/全部评分分数大于4分.csv', index_col=None, header=None)
    # 最小电影编号100008 ,只要小于100000就是人

    ratingnum = pd.read_csv('../数据集/数据清洗推特数据集/评分大于4分出现次数.csv', header=None, index_col=0, encoding='utf-8',
                            delimiter=",", engine='python')

    deletelist = []
    count = 0
    for ind, row in df.iterrows():
        # ind和row分别代表了每一行的index和内容。
        if ratingnum.loc[row[0], 1] < 50:
            deletelist.append(ind)
            count += 1

    print("删除评分次数在50次以下的记录:", count)
    # 删除评分次数在50次以下的记录: 354589
    data = df.drop(deletelist)

    # data:需要进行分割的数据集
    train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

    data.to_csv('../数据集/数据清洗推特数据集/全部评分.csv', index=None, header=None)
    print('data.shape', data.shape)
    # data.shape (472096, 3)
    # 全部评分，去掉不合适的人和电影 =train_set+test_set

    train_set.to_csv('../数据集/数据清洗推特数据集/train.csv', index=None, header=None)
    print('train_set.shape', train_set.shape)
    # train_set.shape (377676, 3)

    test_set.to_csv('../数据集/数据清洗推特数据集/test.csv', index=None, header=None)
    print('test_set.shape', test_set.shape)
    # test_set.shape (94420, 3)


def pe_mv_count():
    df = pd.read_csv('../数据集/数据清洗推特数据集/全部评分.csv', index_col=None, header=None)

    print("人员数量", len(df[0].unique()))
    print("电影数量", len(df[1].unique()))


if __name__ == '__main__':
    # 选择评分大于等于5分，表示有过观影记录，小于5分，认为没看过，剩余数据2-8拆分，交互记录大于40次的用户进行2-8拆分

    # generateocc()

    # 2.分割数据
    splitdata()

    pe_mv_count()
