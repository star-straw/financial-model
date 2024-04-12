import numpy as np
from collections import Counter
import time
seed = int(time.time())
np.random.seed(123)

# 定义雪球产品参数
current_price = 100  # 标的资产当前价格
volatility = 0.1186  # 标的资产波动率
risk_free_rate = 0.023  # 无风险利率
maturity = 2  # 合约期限，以年为单位
trading_days = 242 #交易日数
knock_in =0.8
knock_out = 1.05
knock_in_price = knock_in * current_price  # 敲入价格
knock_out_price = knock_out * current_price  # 敲出价格
dividend_yield = 0  # 股息收益率
knock_out_coupon = 0.20 #敲出票息
bonus_coupon = 0.20 #红利票息
nominal_principal = 1000000 #名义本金
# 蒙特卡洛模拟
num_paths = 20000 #模拟路径数目
z_values = np.random.normal(0, 1, (num_paths, maturity * trading_days)) #随机扰动项 ε

# 蒙特卡洛模拟函数
def monte_carlo_simulation_with_progress(num_paths, current_price, volatility, risk_free_rate, maturity):
    dt = 1 / trading_days  # 时间步长
    num_steps = int (trading_days * maturity)
    price_paths = np.zeros((num_paths, num_steps + 1))  # 包含对照组的价格路径数组

    price_paths[:, 0] = current_price  # 初始化所有路径的初始价格


    # 计算所有路径的价格路径
    for j in range(1, num_steps + 1):
        drift = (risk_free_rate - dividend_yield - 0.5 * volatility ** 2) * dt
        diffusion = volatility * np.sqrt(dt) * z_values[:, j - 1]
        price_paths[:, j] = price_paths[:, j - 1] * np.exp(drift + diffusion)

    # 计算每条路径的收益
    payoffs = np.zeros(num_paths)  # 包含对照组的收益数组
    occurrences = Counter()  # 记录不同收益率出现的次数

    # 检查敲出条件
    knock_out_indices = np.argmax(price_paths[:, 1::20] > knock_out_price, axis=1)
    # print("Knock out indices:",knock_out_indices)
    knock_out_indices[knock_out_indices == 0] = num_steps + 1  # 将未敲出的设为最后一步
    # print("Knock out",knock_out_indices)
    knock_out_mask = knock_out_indices < num_steps + 1
    payoffs[knock_out_mask] = knock_out_coupon * knock_out_indices[knock_out_mask] * dt / maturity * np.exp(-risk_free_rate * dt * knock_out_indices[knock_out_mask])
    occurrences['knock_out'] = np.sum(knock_out_mask)

    # 检查敲入条件和到期时的情况
    final_prices = price_paths[:, -1]
    knock_in_mask = np.any(price_paths[:, 1:] < knock_in_price, axis=1)
    payoffs[knock_in_mask] = np.minimum(0, (final_prices[knock_in_mask] - current_price) / current_price) * np.exp(-risk_free_rate * maturity)
    occurrences['knock_in'] = np.sum(knock_in_mask)

    # 对于剩余的路径，为 bonus
    bonus_mask = ~(knock_out_mask | knock_in_mask)
    payoffs[bonus_mask] = bonus_coupon * np.exp(-risk_free_rate * maturity)
    occurrences['bonus'] = np.sum(bonus_mask)

    # # 输出不同收益类型的出现次数
    # print("Occurrences of each type of payoff:")
    # for payoff_type, count in occurrences.items():
    #     print(f"{payoff_type}: {count}")

    # 计算平均收益并贴现
    average_payoff = np.mean(payoffs) * nominal_principal #均值贴现
    return average_payoff



if __name__ == "__main__":
    # 执行模拟
    simulation = monte_carlo_simulation_with_progress(num_paths,current_price,volatility,risk_free_rate,maturity)
    # 输出雪球产品定价
    print(f"The estimated price of the snowball product is: {simulation:.2f}")








