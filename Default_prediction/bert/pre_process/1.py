import pandas as pd

# # 读取 predict.csv 文件
# data = pd.read_csv("../predict.csv")
#
# # 增加名为 Label 的列，所有值设置为 0
# data["Label"] = 0
#
# # 保存修改后的文件
# data.to_csv("predict.csv", index=False)

# import pandas as pd
#
# # 读取 CSV 文件
# df = pd.read_csv("predict1.csv")
#
# # 提取 Label 列中的数字并转换成字符串
# df['Label'] = df['Label'].apply(lambda x: str(x).split("[")[1].split("]")[0])
#
# # 将处理后的 DataFrame 保存到新的 CSV 文件中
# df.to_csv('predict2.csv', index=False)

# import pandas as pd
#
# # 读取 CSV 文件
# df = pd.read_csv("dataset.csv")
#
# # 根据条件处理 label 列
# df['Label'] = df['Label'].apply(lambda x: 1 if x > 0.5 else 0)
#
# # 保存处理后的 DataFrame 到新的 CSV 文件中
# df.to_csv('processed_dataset.csv', index=False)

import pandas as pd

# # 读取处理后的 dataset.csv 和 predict2.csv
# processed_df = pd.read_csv("processed_dataset.csv")
# predict_df = pd.read_csv("predict2.csv")
#
# # 使用 concat 函数将两个 DataFrame 按行连接起来
# merged_df = pd.concat([processed_df, predict_df])
#
# # 保存合并后的 DataFrame 到新的 CSV 文件中
# merged_df.to_csv('merged_data.csv', index=False)

import pandas as pd
df = pd.read_csv("predict2.csv")
df.to_excel("data.xlsx", index=False)