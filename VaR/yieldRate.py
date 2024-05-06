import pandas as pd

# 读取Excel文件
df = pd.read_excel('data.xlsx', engine='openpyxl', index_col='date')
seq_column = df['seq']
df.drop(columns=['seq'], inplace=True)

# 计算每个日期对应的收益率
returns = (df.shift(-1) - df) / df

# date_column = returns.index
# returns['date'] = date_column
# returns = returns[['date'] + list(returns.columns[:-1])]
returns['seq'] = seq_column
# 输出计算结果，检查是否正确
print(returns.head())

# 将结果保存为CSV文件
returns.to_csv('data2.csv', index=True)  # 禁止保存索引列
