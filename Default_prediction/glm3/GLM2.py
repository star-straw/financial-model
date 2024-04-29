import pandas as pd
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import torch
import concurrent.futures

# 初始化模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained("F:\models\THUDM_chatglm3-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("F:\models\THUDM_chatglm3-6b", trust_remote_code=True).cuda()
model = model.eval()


history = []


# 定义请求函数
def process_row(row):
    limit = '请给下面的新闻标题的态度打一个区间0~1之间的的小数分，态度越积极，分数越接近1。反之，分数越接近0；记住你只需要给出分数即可/n 新闻标题：'
    query = limit + row['Title']
    response, _ = model.chat(tokenizer, query, history=history)
    print(response)
    return response


# 读取文件
df_update = pd.read_excel('filtered_data.xlsx')
print("训练数据加载完毕")

# 创建线程池
max_threads = 8
with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
    # 提交任务
    futures = {executor.submit(process_row, row): row for index, row in df_update.iterrows()}

    # 获取结果并添加到 DataFrame 的新列
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        row = futures[future]
        response = future.result()
        df_update.at[row.name, 'Response'] = response

# 将更新后的 DataFrame 保存到 update_data.xlsx 文件
df_update.to_excel('update_data1.xlsx', index=False)
