from price import monte_carlo_simulation_with_progress, volatility, num_paths, risk_free_rate, current_price

def delta_greek(new_price, maturity):
    price_1 = monte_carlo_simulation_with_progress(num_paths, new_price, volatility, risk_free_rate, maturity)
    epsilon = 0.001 * new_price  # 定义一个微小的变化
    price_2 = monte_carlo_simulation_with_progress(num_paths, new_price + epsilon, volatility, risk_free_rate, maturity)
    deltas = (price_2 - price_1) / epsilon
    return deltas

if __name__ == "__main__":
    simulation = delta_greek(current_price ,2)
    print(simulation)