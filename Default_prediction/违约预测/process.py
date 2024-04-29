import pandas as pd

# 读取原始数据
df = pd.read_csv("predict2.csv")
# 根据 ShortName 判断是否含有 "ST"，并赋予新的列 "ST" 标记
df['ST'] = df['ShortName'].apply(lambda x: 0 if 'ST' in x else 1)
# 将 DeclareDate 转换为日期格式
df['DeclareDate'] = pd.to_datetime(df['DeclareDate'])
# 提取年份列
df['Year'] = df['DeclareDate'].dt.year
# 根据 Symbol 和 Year 计算每年 Label 为 1 和 Label 为 0 的数量
yearly_counts = df.groupby(['Symbol', 'Year', 'Label']).size().unstack(fill_value=0)
# 根据计数确定每年的 Label
yearly_label = (yearly_counts[1] > yearly_counts[0]).astype(int)
# 根据 Symbol 和 Year 计算每年 Label 为 1 和 Label 为 0 的数量
yearly_counts2 = df.groupby(['Symbol', 'Year', 'ST']).size().unstack(fill_value=0)
# 根据计数确定每年的 Label
yearly_ST = (yearly_counts2[1] > yearly_counts2[0]).astype(int)

# 将 yearly_label 和 yearly_ST 转换为 DataFrame
yearly_label = yearly_label.to_frame().reset_index()
yearly_ST = yearly_ST.to_frame().reset_index()
print(yearly_label)
print(yearly_ST)
yearly_ST.rename(columns={0: 'ST'}, inplace=True)
yearly_ST['Label'] = yearly_label[0]
print(yearly_ST)
yearly_ST.to_csv("data.csv", index=False)
yearly_ST.to_csv("data.xlsx", index=False)