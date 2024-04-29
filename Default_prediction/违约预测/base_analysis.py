import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv("data.csv")

# 计算每年的准确率
accurate_predictions = df[df['ST'] == df['Label']]
accuracy_per_year = accurate_predictions.groupby('Year').size() / df.groupby('Year').size()

# 绘制准确率图像
plt.plot(accuracy_per_year.index, accuracy_per_year.values, marker='o')
plt.title('Accuracy of Label Predictions per Year')
plt.xlabel('Year')
plt.ylabel('Accuracy')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()
