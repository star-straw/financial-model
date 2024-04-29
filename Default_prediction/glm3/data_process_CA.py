import pandas as pd

# # 读取数据文件
# df = pd.read_excel('data_c.xlsx',sheet_name='Sheet2')
#
# # 使用正则表达式匹配符合要求的股票代码
# selected_rows = df[df['Symbol'].astype(str).str.match(r'^6\d{5}$')]

# # 将筛选后的数据写入新文件
# selected_rows.to_excel('data_CA.xlsx', index=False)
#
# 读取数据文件
df = pd.read_excel('300.xlsx')

# 提取不重复的Symbol列
unique_symbols = df['Symbol'].drop_duplicates()

# 将不重复的Symbol保存到原始Excel文件的新sheet
with pd.ExcelWriter('data/data_CA.xlsx', engine='openpyxl', mode='a') as writer:
    unique_symbols.to_excel(writer, sheet_name='sheet3', index=False, header=['300'])
