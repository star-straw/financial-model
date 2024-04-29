import pandas as pd
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch
import os
tokenizer = AutoTokenizer.from_pretrained("F:\models\THUDM_chatglm3-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("F:\models\THUDM_chatglm3-6b", trust_remote_code=True).cuda()
# 加载预训练模型
# prefix_state_dict = torch.load(os.path.join("F:\models\ptuning", "pytorch_model.bin"))
# new_prefix_state_dict = {}
# for k, v in prefix_state_dict.items():
#     if k.startswith("transformer.prefix_encoder."):
#         new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
# model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
model = model.eval()

# 读取 Data.csv 文件
# df = pd.read_csv('Data.csv')
history = []

# # 随机选择2000条记录
# df_sample = df.sample(n=6)
# for index, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
#     query = row['review']
#     response = str(row['label'])  # 将 label 转换为字符串格式
#     history.append({"role": 'user', "content": query})
#     history.append({"role": 'assistant', "content": response})
limit = '请给下面的新闻标题的态度打一个区间0~1之间的的小数分，态度越积极，分数越接近1。反之，分数越接近0；记住你只需要给出分数即可/n 新闻标题：'

# query = limit+'2014年上市公司半年报预告:155公司近六成预喜25家净利翻番'
# response, history = model.chat(tokenizer, query, history=history)
# print(response)
# print(history)

# 读取文件
df_update = pd.read_csv('missing_rows.csv')
print("训练数据加载完毕")

# 设置起始行数
start_index = 0

# 更新 history
for index, row in tqdm(df_update.iloc[start_index:].iterrows(), total=len(df_update) - start_index):
    query = limit + row['Title']
    response, _ = model.chat(tokenizer, query, history=history)
    print(response)
    # 将 response 添加为 DataFrame 的新列
    df_update.at[index, 'Label'] = response
    # 将更新后的 DataFrame 以追加模式保存回 CSV 文件
    df_update.to_csv('gen1.csv', mode='a', header=False, index=False)

