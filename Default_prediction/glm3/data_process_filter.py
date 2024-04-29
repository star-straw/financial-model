# import pandas as pd
#
# # 读取data_CA.xlsx文件的sheet2和sheet3
# sheet2_df = pd.read_excel('data_CA.xlsx', sheet_name='sheet2')
# sheet3_df = pd.read_excel('data_CA.xlsx', sheet_name='sheet3')
#
# # 选取sheet2和sheet3中的Symbol列
# symbol_series_sheet2 = sheet2_df['Symbol']
# symbol_series_sheet3 = sheet3_df['Symbol']
#
# # 从sheet3中获取所有的Symbol
# all_symbols_sheet3 = symbol_series_sheet3.unique()
#
# # 从sheet2中选取75条不包含在sheet3中的数据
# unique_symbols_sheet2 = symbol_series_sheet2[~symbol_series_sheet2.isin(all_symbols_sheet3)].drop_duplicates().sample(76, replace=False)
#
# # 从sheet3中选取所有的Symbol
# unique_symbols_sheet3 = symbol_series_sheet3.drop_duplicates()
#
# # 合并sheet2和sheet3中的Symbol列
# combined_symbols = pd.concat([unique_symbols_sheet2, unique_symbols_sheet3])
#
# # 将合并后的数据写入CSV文件
# combined_symbols.to_excel('symbols.xlsx', index=False)

import pandas as pd

# # 读取data.xlsx文件和Unique_Symbols sheet
# data_df = pd.read_excel('data/data.xlsx')
# symbols_df = pd.read_excel('data/Symbols.xlsx')
#
# # 获取Symbol中的股票代码列表
# symbols_list = symbols_df['300'].astype(str).tolist()
# print(symbols_list)
#
# # 筛选data.xlsx中符合条件的行
# filtered_df = data_df[data_df['Symbol'].astype(str).isin(symbols_list)]
#
# # 将筛选后的数据保存到新的xlsx文件中
# filtered_df.to_excel('300.xlsx', index=False)

import pandas as pd

# 加载Excel文件
df = pd.read_excel('data/300.xlsx')

# 删除包含特定字符串的行
df = df[~df['Title'].str.contains('简报')]

# 保存到新的Excel文件中
df.to_excel('C_300.xlsx', index=False)
