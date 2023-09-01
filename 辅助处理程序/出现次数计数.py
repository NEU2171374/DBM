import pandas as pd
import numpy as np


df = pd.read_csv('../数据集/数据清洗推特数据集/test.csv', index_col=None, header=None)

# '../数据集/数据清洗MovieLens数据集/test.csv'

# df1 = df[df[2] == 3] # 筛选出3分的电影
df1 = df[0].value_counts()
# 数据透视，电影评分3分的电影-人数

df1.to_csv('../数据集/数据清洗推特数据集/推特test出现次数.csv', index=True, header=None)


