from torch.utils.data import DataLoader
from model import *
from dataset import *
from dataProcessor import *
from transformers import BertTokenizer
from tqdm import tqdm



# 其他代码保持不变...

datadir = ""
bert_dir = "bert_cn"
# 加载训练好的模型
checkpoint_path = "best_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ClassifierModel(bert_dir).to(device)
model.load_state_dict(torch.load(checkpoint_path))
model.eval()  # 设置模型为评估模式

my_processor = MyPro()
label_list = my_processor.get_labels()
predict_data = my_processor.get_predict_examples(datadir)
tokenizer = BertTokenizer.from_pretrained(bert_dir)
predict_features = convert_examples_to_features(predict_data, label_list, 128, tokenizer)
predict_dataset = MyDataset(predict_features, 'predict')
# 加载测试数据
predict_data_loader = DataLoader(dataset=predict_dataset, batch_size=1, shuffle=False, pin_memory=True)

df =pd.read_csv("predict.csv")
predictions = []
test_total_accuracy = 0
with torch.no_grad():
    for batch_data in tqdm(predict_data_loader, desc="Predicting"):
        batch_data = {key: value.to(device) for key, value in batch_data.items()}  # Move batch to GPU
        output = model(**batch_data)
        # 应用 softmax 函数将 logits 转换为概率分布
        probabilities = torch.softmax(output, dim=1)
        # 获取概率最高的类别作为预测的标签
        predicted_labels = torch.argmax(probabilities, dim=1)
        predictions.append(predicted_labels)
        predictions.append(predicted_labels.cpu())
# 将预测结果写入到 DataFrame 中
df['Label'] = predictions
# 将 DataFrame 中的数据写入到 predict.csv 文件中
df.to_csv('predict1.csv', index=False)

