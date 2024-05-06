from scipy.optimize import minimize
import numpy as np
from VaR import PortfolioVaRCalculator,confidence_level,portfolio_value_changes,calculator
from VaR_validation import main, draw, significance_level
def optimize_portfolio_allocation(calculator, total_value, method):
    def objective_function(allocation):
        # 确保分配的总和等于总价值
        normalized_allocation = allocation / np.sum(allocation) * total_value
        calculator.portfolio_value = normalized_allocation
        if method == 1:
            return -calculator.calculate_portfolio_VaR()
        elif method == 2:
            return -calculator.calculate_portfolio_VaR_time_weighted()
        elif method == 3:
            return -calculator.calculate_portfolio_VaR_EWMA()
        else:
            raise ValueError("无效的方法选择，必须是传统法、时间加权法或者波动率加权法中的一种")

    # 初始分配猜测
    x0 = np.array([0.3, 0.4, 0.3])

    # 约束条件: 分配的总和等于总价值
    constraints = ({'type': 'eq', 'fun': lambda allocation: np.sum(allocation) - total_value})

    # 分配的边界 (0 <= allocation <= total_value)
    bounds = [(0, total_value)] * 3

    # 进行优化
    result = minimize(objective_function, x0, method='SLSQP', bounds=bounds, constraints=constraints)

    # 获取优化后的分配
    optimized_allocation = result.x / np.sum(result.x) * total_value

    calculator.portfolio_value = optimized_allocation
    # 计算优化后的 VaR
    if method == 1:
        optimized_VaR = calculator.calculate_portfolio_VaR()
    elif method == 2:
        optimized_VaR = calculator.calculate_portfolio_VaR_time_weighted()
    elif method == 3:
        optimized_VaR = calculator.calculate_portfolio_VaR_EWMA()
    else:
        optimized_VaR = None
        raise ValueError("无效的方法选择，必须是传统法、时间加权法或者波动率加权法中的一种")

    return optimized_allocation, optimized_VaR

method=3
if __name__ == "__main__":
    for method in range(1, 4):
        # 优化资产分配
        optimized_allocation, optimized_VaR = optimize_portfolio_allocation(calculator, 1000000, method) #1表示传统法计算VaR 2表示时间加权法 3表示EWMA法
        print("优化后的资产分配:")
        for i, allocation in enumerate(optimized_allocation):
            print(f"资产 {i+1}: ${allocation:.2f}")
        print(f"优化后的 VaR: {optimized_VaR:.4f}")

        calculator_2 = PortfolioVaRCalculator('data2.csv', '2023-07-31', optimized_allocation, 0.95)
        portfolio_value_changes_2 = calculator_2.calculate_portfolio_value_changes('2023-08-01', '2024-04-16')
        main(confidence_level,significance_level,method,calculator_2,portfolio_value_changes)
    # draw(portfolio_value_changes_2)
