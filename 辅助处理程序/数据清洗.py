from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


def handle_data(droprate=4, appear_counts=5):
    # droprate=4 评分小于4分的记录会被删除； appear_counts=5 删除后的交互记录小于5次也会被删除

    df = pd.read_csv('../数据集/MovieLens1M/ratings.dat', header=None, encoding='utf-8',
                     delimiter="::", engine='python')
    #  UserIDs range between 1 and 6040
    #  MovieIDs range between 10001 and 13952
    df[1] = df[1] + 10000
    del df[3]

    df = df.drop(df[df[2] < droprate].index)
    print("未删减数据表大小：", df.shape)  # (575281, 3)
    df.to_csv('../数据集/数据清洗MovieLens数据集/全部评分.csv', index=None, header=None)

    df_appear_count = df[0].value_counts()
    df_appear_count.to_csv('../数据集/数据清洗MovieLens数据集/出现次数.csv', index=True, header=None)

    drop = droplist(appear_count=appear_counts)

    df = washset(df, drop)
    print("删减出现次数较少的记录，最后表大小", df.shape)  # (575242, 3)
    print("用户数量：", len(df[0].unique()))  # 用户数量： 6028
    print("电影数量：", len(df[1].unique()))  # 电影数量： 3533
    df.to_csv('../数据集/数据清洗MovieLens数据集/全部评分.csv', index=None, header=None)

    train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
    # test_size:将数据分割成训练集的比例、random_state:设置随机种子，保证每次运行生成相同的随机数
    train_set.to_csv('../数据集/数据清洗MovieLens数据集/train.csv', index=None, header=None)
    test_set.to_csv('../数据集/数据清洗MovieLens数据集/test.csv', index=None, header=None)


def droplist(appear_count=5):
    # 出现次数小于5次，会被标记，以列表形式返回
    df = pd.read_csv('../数据集/数据清洗MovieLens数据集/出现次数.csv', header=None, encoding='utf-8',
                     delimiter=",", engine='python')

    dplist = []

    for index, row in df.iterrows():
        if row[1] <= appear_count:
            dplist.append(row[0])

    print(len(dplist), dplist)

    return dplist


def washset(data, drop_list):
    delete_idex = []
    count = 0
    for index, row in data.iterrows():
        # ind和row分别代表了每一行的index和内容。
        if row[0] in drop_list:
            delete_idex.append(index)
            count += 1

    print("删除的记录个数为:", count)
    data = data.drop(delete_idex)

    return data


if __name__ == '__main__':
    handle_data(droprate=4, appear_counts=5)
    # 评分次数小于4分记录会被删除，删除后的记录少于5次交互，也会被删除
