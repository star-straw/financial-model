import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2
from VaR import calculator,portfolio_value_changes,confidence_level


def kupiec_test(confidence_level, significance_level, method=1, calculator=calculator, portfolio_value_changes=portfolio_value_changes):
    if method == 3:
        VaR = calculator.calculate_portfolio_VaR_EWMA()
    elif method == 2:
        VaR = calculator.calculate_portfolio_VaR_time_weighted()
    else:
        VaR = calculator.calculate_portfolio_VaR()
    # 获取实际的组合盈亏额
    actual_losses = portfolio_value_changes.diff().dropna()
    # 计算观察期内超出 VaR 的次数
    exceed_count = sum(actual_losses < VaR)
    # 计算 VaR 违约率
    VaR_failure_rate = exceed_count / len(actual_losses)
    # 计算 Kupiec 检验统计量
    test_statistic = -2 * ((len(actual_losses)-exceed_count) * np.log(confidence_level / (1-VaR_failure_rate)) + exceed_count * np.log( (1-confidence_level) / VaR_failure_rate))
    # 计算 Kupiec 检验的 p-value
    p_value = chi2.cdf(test_statistic, 1)
    # 判断是否通过 Kupiec 检验
    if p_value < significance_level:
        return False, test_statistic, p_value
    else:
        return True, test_statistic, p_value

def main(confidence_level,significance_level,method=1,calculator=calculator, portfolio_value_changes=portfolio_value_changes):
    if method == 3:
        print("EWMA 法 Kupiec 检验结果:")
    elif method ==2:
        print("时间加权历史模拟法 Kupiec 检验结果:")
    else:
        print("传统历史模拟法 Kupiec 检验结果:")
    pass_kupiec, test_statistic, p_value = kupiec_test(confidence_level, significance_level, method,calculator, portfolio_value_changes)
    print(f"通过检验: {pass_kupiec}, 检验统计量: {test_statistic}, p-value: {p_value}")

def draw(portfolio_value_changes=portfolio_value_changes):
    # 读取 data2.csv 文件
    data2 = pd.read_csv('data2.csv')

    # 将 index 和 values 转换为 numpy 数组
    index_array = portfolio_value_changes.index.to_numpy()
    values_array = portfolio_value_changes.to_numpy()

    # 创建一个空列表用于存储匹配的日期序列
    date_sequence = []

    # 根据索引匹配 data2 中的日期值
    for idx in index_array:
        date = data2.loc[idx-1, 'date']
        date_sequence.append(date)

    # 绘制折线图
    plt.figure(figsize=(10, 6))
    plt.plot(date_sequence, values_array, color='blue', linewidth=2, label='Portfolio Value')
    plt.title('Portfolio Value Changes', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Portfolio Value', fontsize=14)
    plt.grid(True)
    plt.legend(loc='upper left')

    first_date = date_sequence[0]
    last_date = date_sequence[-1]
    plt.xticks([first_date] + date_sequence[::10] + [last_date])
    # 自动调整日期标签
    plt.gcf().autofmt_xdate()
    plt.show()


significance_level = 0.05
if __name__ == "__main__":
    main(confidence_level,significance_level,1)#1表示传统法计算VaR 2表示时间加权法 3表示EWMA法,默认使用方法一
    main(confidence_level,significance_level,2)
    main(confidence_level,significance_level,3)
    draw()
