import numpy as np


test02 = np.load('../数据集/嵌入向量/deepWalk算法/deepWalk算法128维.npy', encoding="latin1")
doc = open('../数据集/嵌入向量/deepWalk算法/deepWalk算法128维.txt', 'a')  # 打开一个存储文件，并依次写入
print(test02, file=doc)  # 将打印内容写入文件中