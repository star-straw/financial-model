import matplotlib.pyplot as plt
import pandas as pd

from VaR import calculator, portfolio_value_changes, PortfolioVaRCalculator
from optimize import optimize_portfolio_allocation

def draw_multiple(portfolio_value_changes_list, labels):
    plt.figure(figsize=(10, 6))
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']  # 指定中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
    data2 = pd.read_csv("data2.csv")
    for i, portfolio_value_changes in enumerate(portfolio_value_changes_list):
        index_array = portfolio_value_changes.index.to_numpy()
        values_array = portfolio_value_changes.to_numpy()
        date_sequence = []
        for idx in index_array:
            date = data2.loc[idx-1, 'date']
            date_sequence.append(date)
        plt.plot(date_sequence, values_array, label=labels[i])

    plt.title('Portfolio Value Changes Comparison', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Portfolio Value', fontsize=14)
    plt.grid(True)
    plt.legend(loc='upper left')

    first_date = date_sequence[0]
    last_date = date_sequence[-1]
    plt.xticks([first_date] + date_sequence[::10] + [last_date])
    plt.gcf().autofmt_xdate()
    plt.show()

optimized_portfolio_value_changes_list = []  # 这里只有一个优化后的投资组合价值变动数据，你可以根据需要添加更多
optimized_labels = ['传统法','时间加权法','EWMA法']
for i in range(1,4):
    optimized_allocation, optimized_VaR = optimize_portfolio_allocation(calculator, 1000000, i) #1表示传统法计算VaR 2表示时间加权法 3表示EWMA法
    calculator_2 = PortfolioVaRCalculator('data2.csv', '2023-07-31', optimized_allocation, 0.95)
    portfolio_value_changes_2 = calculator_2.calculate_portfolio_value_changes('2023-08-01', '2024-04-16')
    optimized_portfolio_value_changes_list.append(portfolio_value_changes_2)

draw_multiple([portfolio_value_changes] + optimized_portfolio_value_changes_list, ['原始投资组合'] + optimized_labels)
