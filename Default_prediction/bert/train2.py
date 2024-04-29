from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import *
from dataset import *
from dataProcessor import *
import matplotlib.pyplot as plt
import time
import torch
from transformers import BertTokenizer
# 设置日志级别
from transformers import logging
logging.set_verbosity_warning()

# 加载训练数据
datadir = ""
bert_dir = "bert_cn"
my_processor = MyPro()
label_list = my_processor.get_labels()

train_data = my_processor.get_train_examples(datadir)
test_data = my_processor.get_test_examples(datadir)

tokenizer = BertTokenizer.from_pretrained(bert_dir)

train_features = convert_examples_to_features(train_data, label_list, 128, tokenizer)
test_features = convert_examples_to_features(test_data, label_list, 128, tokenizer)
train_dataset = MyDataset(train_features, 'train')
test_dataset = MyDataset(test_features, 'test')
train_data_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, pin_memory=True)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True, pin_memory=True)

train_data_len = len(train_dataset)
test_data_len = len(test_dataset)
print(f"训练集长度：{train_data_len}")
print(f"测试集长度：{test_data_len}")

# 创建网络模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
my_model = ClassifierModel(bert_dir).to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
learning_rate = 0.0001
optimizer = torch.optim.Adam(my_model.parameters(), lr=learning_rate, betas=(0.95, 0.995))

# 定义训练参数
epoch = 500  # 训练轮数
best_accuracy = 0.0  # 最佳测试集准确度
best_model_path = "best_model.pth"  # 最佳模型参数保存路径

# 定义日志记录器
writer = SummaryWriter("logs")

# 训练过程
for i in range(epoch):
    print(f"-------第{i+1}轮训练开始-------")

    # 训练阶段
    my_model.train()
    train_total_accuracy = 0
    for step, batch_data in enumerate(train_data_loader):
        batch_data = {key: value.to(device) for key, value in batch_data.items()}  # Move batch to GPU
        output = my_model(**batch_data)
        loss = loss_fn(output, batch_data['label_id'])
        train_accuracy = (output.argmax(1) == batch_data['label_id']).sum()
        train_total_accuracy += train_accuracy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar("train_loss", loss.item(), step)

    # 计算训练集准确度
    train_total_accuracy = train_total_accuracy.item() / train_data_len
    print(f"训练集上的准确率：{train_total_accuracy}")
    writer.add_scalar("train_accuracy", train_total_accuracy, i)

    # 测试阶段
    my_model.eval()
    test_total_accuracy = 0
    total_test_loss = 0
    with torch.no_grad():
        for batch_data in test_data_loader:
            batch_data = {key: value.to(device) for key, value in batch_data.items()}  # Move batch to GPU
            output = my_model(**batch_data)
            loss = loss_fn(output, batch_data['label_id'])
            total_test_loss += loss.item()
            test_accuracy = (output.argmax(1) == batch_data['label_id']).sum()
            test_total_accuracy += test_accuracy
        test_total_accuracy = test_total_accuracy.item() / test_data_len
        print(f"测试集上的准确率：{test_total_accuracy}")
        print(f"测试集上的loss：{total_test_loss}")
        writer.add_scalar("test_loss", total_test_loss, i)
        writer.add_scalar("test_accuracy", test_total_accuracy, i)

        # 如果当前测试集准确度超过80%，保存模型参数
        if test_total_accuracy > 0.7 and test_total_accuracy > best_accuracy:
            best_accuracy = test_total_accuracy
            torch.save(my_model.state_dict(), best_model_path)
            print("已保存最佳模型参数")

# 关闭日志记录器
writer.close()
