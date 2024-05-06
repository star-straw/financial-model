import pandas as pd
from scipy.stats import norm
import numpy as np
class PortfolioVaRCalculator:
    def __init__(self, data_file, start_date, portfolio_value, confidence_level):
        self.data_file = data_file
        self.df = pd.read_csv(data_file, index_col='seq')
        self.start_date = start_date
        self.portfolio_value = portfolio_value
        self.confidence_level = confidence_level

    def calculate_portfolio_VaR(self):
        # 选择截止到指定日期的历史数据
        historical_data = self.df.loc[self.df['date'] < self.start_date]
        # 选取所有收益列
        asset_returns = historical_data.drop(columns=['date'])
        # 计算每个资产的日收益率乘以对应的投资金额，得到每个资产的日盈亏额
        profit_losses = asset_returns * self.portfolio_value
        # 计算组合每日盈亏额，即每个资产的盈亏额之和
        portfolio_profit_loss = profit_losses.sum(axis=1)
        # 对组合每日盈亏额进行排序
        sorted_profit_loss = portfolio_profit_loss.sort_values()
        # 计算 VaR
        VaR = sorted_profit_loss.quantile(1 - self.confidence_level)
        return VaR

    def calculate_portfolio_VaR_time_weighted(self, lambda_=0.99):
        historical_data = self.df.loc[self.df['date'] < self.start_date]
        asset_returns = historical_data.drop(columns=['date'])
        profit_losses = asset_returns * self.portfolio_value
        portfolio_profit_loss = profit_losses.sum(axis=1)
        t = len(profit_losses)
        time_weights = [(1 - lambda_) * lambda_ ** (t - i - 1) / (1 - lambda_ ** t) for i in range(t)]
        # 创建包含盈亏额和对应时间权重的 DataFrame
        data = {'profit_loss': portfolio_profit_loss, 'time_weight': time_weights}
        df = pd.DataFrame(data)
        # 按照盈亏额升序排列
        sorted_df = df.sort_values(by='profit_loss')
        # 计算时间权重的累积值
        sorted_df['cumulative_time_weight'] = sorted_df['time_weight'].cumsum()
        # 找到超过置信水平的位置
        exceed_index = sorted_df[sorted_df['cumulative_time_weight'] >= 1 - self.confidence_level].index[0]
        # 找到对应的盈亏额
        VaR = sorted_df.loc[exceed_index, 'profit_loss']
        return VaR

    def calculate_portfolio_VaR_EWMA(self, lambda_=0.94):
        historical_data = self.df.loc[self.df['date'] < self.start_date]
        asset_returns = historical_data.drop(columns=['date'])
        # 计算指数加权移动平均值
        ewma_sigma = pd.DataFrame(index=asset_returns.index, columns=asset_returns.columns)

        for col in asset_returns.columns:
            ema_prev = asset_returns[col].iloc[0]**2  # 初始化第一个平滑值为第一个数据点
            ewma_sigma.at[asset_returns.index[0], col] = np.sqrt(ema_prev)
            for i in range(1, len(asset_returns)):
                ema_prev = lambda_ * ema_prev + (1 - lambda_) * asset_returns[col].iloc[i-1]**2
                ewma_sigma.at[asset_returns.index[i], col] = np.sqrt(ema_prev)
        profit_losses = asset_returns.shift(-1) * ewma_sigma.shift(-1) / np.where(ewma_sigma != 0, ewma_sigma, 1) * self.portfolio_value
        # 计算组合每日盈亏额，即每个资产的盈亏额之和
        portfolio_profit_loss = profit_losses.sum(axis=1)
        # 对组合每日盈亏额进行排序
        sorted_profit_loss = portfolio_profit_loss.sort_values()
        # 计算 VaR
        VaR = sorted_profit_loss.quantile(1 - self.confidence_level)
        return VaR

    def calculate_portfolio_value_changes(self,start_date,end_date):
        # 选择从开始日期到结束日期的数据
        selected_data = self.df.loc[(self.df['date'] >= start_date) & (self.df['date'] <= end_date)]
        # 计算资产的收益率
        asset_returns = selected_data.drop(columns=['date'])
        # 计算每个资产的收益额
        asset_profit_losses = asset_returns * self.portfolio_value
        # 计算组合的每日盈亏额
        portfolio_profit_loss = asset_profit_losses.sum(axis=1)
        # 计算组合价值的变化
        portfolio_value_changes = portfolio_profit_loss.cumsum() + np.sum(self.portfolio_value)
        return portfolio_value_changes

confidence_level = 0.95
# 创建 PortfolioVaRCalculator 对象
calculator = PortfolioVaRCalculator('data2.csv', '2023-07-31', [300000, 400000, 300000], confidence_level)
portfolio_value_changes = calculator.calculate_portfolio_value_changes('2023-08-01', '2024-04-16')

if __name__ == "__main__":
    # 调用方法计算组合的 VaR（传统历史模拟法）
    portfolio_VaR = -calculator.calculate_portfolio_VaR()
    print(f"2023-08-01 的组合 VaR（传统历史模拟法，95% 置信水平）为：{portfolio_VaR:.4f}")

    # 调用方法计算组合的 VaR（时间加权历史模拟法）
    portfolio_VaR_time_weighted = -calculator.calculate_portfolio_VaR_time_weighted(lambda_=0.99)
    print(f"2023-08-01 的组合 VaR（时间加权历史模拟法，95% 置信水平，lambda=0.99）为：{portfolio_VaR_time_weighted:.4f}")

    # 调用方法计算组合的 VaR（EWMA 法）
    portfolio_VaR_EWMA = -calculator.calculate_portfolio_VaR_EWMA(lambda_=0.95)
    print(f"2023-08-01 的组合 VaR（EWMA 法，95% 置信水平，lambda=0.95）为：{portfolio_VaR_EWMA:.4f}")

