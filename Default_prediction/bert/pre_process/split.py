import pandas as pd
from sklearn.model_selection import train_test_split

# 读取数据
data = pd.read_csv('dataset.csv')

# 划分数据集为训练集和测试集
train_data, test_data = train_test_split(data, test_size=0.2, random_state=1234)

# 保存数据到CSV文件
train_data.to_csv('train.csv', index=False)
test_data.to_csv('test.csv', index=False)
