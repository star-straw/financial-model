from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import *
from dataset import *
from dataProcessor import *
import matplotlib.pyplot as plt
import time
from transformers import BertTokenizer
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
learning_rate = 0.001
#optimizer = torch.optim.SGD(my_model.parameters(), lr=learning_rate)
#  Adam 参数betas=(0.9, 0.99)
optimizer = torch.optim.Adam(my_model.parameters(), lr=learning_rate, betas=(0.9, 0.99))
# 定义优化器（使用RMSprop优化器）
# optimizer = optim.RMSprop(my_model.parameters(), lr=learning_rate)
# 总共的训练步数
total_train_step = 0
# 总共的测试步数
total_test_step = 0
step = 0
epoch = 50

writer = SummaryWriter("logs")
# writer.add_graph(myModel, input_to_model=myTrainDataLoader[1], verbose=False)
# writer.add_graph(myModel)
train_loss_his = []
train_totalaccuracy_his = []
test_totalloss_his = []
test_totalaccuracy_his = []
start_time = time.time()
my_model.train()
for i in range(epoch):
    print(f"-------第{i}轮训练开始-------")
    train_total_accuracy = 0
    for step, batch_data in enumerate(train_data_loader):
        batch_data = {key: value.to(device) for key, value in batch_data.items()}  # Move batch to GPU
        # writer.add_images("tarin_data", imgs, total_train_step)
        print(batch_data['input_ids'].shape)
        output = my_model(**batch_data)
        loss = loss_fn(output, batch_data['label_id'])
        train_accuracy = (output.argmax(1) == batch_data['label_id']).sum()
        train_total_accuracy = train_total_accuracy + train_accuracy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step = total_train_step + 1
        train_loss_his.append(loss)
        writer.add_scalar("train_loss", loss.item(), total_train_step)
    train_total_accuracy = train_total_accuracy / train_data_len
    print(f"训练集上的准确率：{train_total_accuracy}")
    train_totalaccuracy_his.append(train_total_accuracy)
    # 测试开始
    total_test_loss = 0
    my_model.eval()
    test_total_accuracy = 0
    with torch.no_grad():
        for batch_data in test_data_loader:
            batch_data = {key: value.to(device) for key, value in batch_data.items()}  # Move batch to GPU
            output = my_model(**batch_data)
            loss = loss_fn(output, batch_data['label_id'])
            total_test_loss = total_test_loss + loss
            test_accuracy = (output.argmax(1) == batch_data['label_id']).sum()
            test_total_accuracy = test_total_accuracy + test_accuracy
        test_total_accuracy = test_total_accuracy / test_data_len
        print(f"测试集上的准确率：{test_total_accuracy}")
        print(f"测试集上的loss：{total_test_loss}")
        test_totalloss_his.append(total_test_loss)
        test_totalaccuracy_his.append(test_total_accuracy)
        writer.add_scalar("test_loss", total_test_loss.item(), i)
# for parameters in myModel.parameters():
#    print(parameters)
end_time = time.time()
total_train_time = end_time-start_time
print(f'训练时间: {total_train_time}秒')
writer.close()
plt.plot([loss.cpu().item() for loss in train_loss_his], label='Train Loss')
plt.legend(loc='best')
plt.xlabel('Steps')
plt.show()
plt.plot([loss.cpu().item() for loss in test_totalloss_his], label='Test Loss')
plt.legend(loc='best')
plt.xlabel('Steps')
plt.show()

# 将训练准确率从GPU移到CPU上
train_totalaccuracy_his_cpu = [accuracy.cpu().item() for accuracy in train_totalaccuracy_his]
# 将测试准确率从GPU移到CPU上
test_totalaccuracy_his_cpu = [accuracy.cpu().item() for accuracy in test_totalaccuracy_his]
# 绘制训练准确率和测试准确率曲线
plt.plot(train_totalaccuracy_his_cpu, label='Train accuracy')
plt.plot(test_totalaccuracy_his_cpu, label='Test accuracy')
plt.legend(loc='best')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

