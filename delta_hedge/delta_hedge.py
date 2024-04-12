from option import Option
import numpy as np
from openpyxl import Workbook


def delta_hedge_simulation(S, delta_hedge, T_len):
    stock_expense = []
    accum_cash = []
    Interest = []
    stock_units = []
    cost = 0
    prev_delta = 0

    for i, price in enumerate(S):
        delta = delta_hedge[i]
        delta_change = delta - prev_delta  # 计算delta差
        stock_units_change = delta_change * 100000  # 计算需要交易的现货份数(卖出为负值)

        cost += stock_units_change * price  # 交易现货增加的成本(卖出为负值)
        interest = cost * r / 52 * T_len  # 计算本期利息费用
        cost += interest  # 累计成本

        prev_delta = delta  # 更新delta

        # 存入数据，最后返回
        stock_units.append(stock_units_change)
        stock_expense.append(price * stock_units_change / 1000)
        accum_cash.append(cost / 1000)
        Interest.append(interest / 1000)

    return stock_units, stock_expense, accum_cash, Interest


if __name__ == '__main__':
    T_len = 0.5  # 对冲周期
    K = 50
    r = 0.05
    sigma = 0.20
    S = np.array([
        49, 49.75, 52, 50, 48.38, 48.25, 48.75, 49.63, 48.25, 48.25,
        51.12, 51.5, 49.88, 49.88, 48.75, 47.5, 48.00, 46.25, 48.13,
        46.63, 48.12  # 将到期日的标的资产价格加入S数组中
    ])
    if T_len == 0.5:
        S = np.array([
            49, 49.375, 49.75, 50.875, 52, 51, 50, 49.19, 48.38, 48.315,
            48.25, 48.5, 48.75, 49.19, 49.63, 48.94, 48.25, 48.25, 48.25,
            49.685, 51.12, 51.31, 51.5, 50.69, 49.88, 49.88, 49.88, 49.315,
            48.75, 48.125, 47.5, 47.75, 48, 47.125, 46.25, 47.19, 48.13,
            47.38, 46.63, 47.375, 48.12
        ])

    T = np.arange(20, 0, -T_len) / 52
    option_type = 1
    # S中包含了到期价格,故不传入最后一位股票价格
    call_option = Option(K, r, sigma, S[:-1], T, option_type)
    delta_hedge = call_option.delta

    # 到期delta根据期权价值的虚实决定 因为T=0 无法代入delta计算公式
    if (S[-1] > K and option_type == 1) or (S[-1] < K and option_type == -1):
        delta = 1
    else:
        delta = 0

    # 将到期的delta 加入delta_hedge序列 匹配S的长度
    delta_hedge = np.append(delta_hedge, delta)
    stock_units, stock_expense, accum_cash, Interest = delta_hedge_simulation(S, delta_hedge, T_len)
    print("股票价格:", S)
    print("delta:", delta_hedge)
    print("购买股票数量:", stock_units)
    print("股票费用:", stock_expense)
    print("累计现金流:", accum_cash)
    print("利息费用:", Interest)

    # 如果需要写入 Excel 文件，取消以下代码的注释
    # wb = Workbook()
    # ws = wb.active
    #
    # ws.append(["周", "股票价格", "delta", "购买股票数量", "股票费用", "累计现金流", "利息费用"])
    # for i in range(len(stock_expense)):
    #     ws.append([i, S[i], delta_hedge[i], stock_units[i], stock_expense[i], accum_cash[i], Interest[i]])
    # # 生成 Excel 文件名，包含 T_len 变量
    # filename = f"delta_hedge_{T_len}weeks.xlsx"
    # wb.save(filename)
