import numpy as np
from scipy.stats import norm


class Option:
    def __init__(self, K, r, sigma, S, T, option_type):
        self.d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        self.d2 = self.d1 - sigma * np.sqrt(T)
        self.delta = option_type * norm.cdf(option_type * self.d1)
        self.gamma = norm.pdf(self.d1) / (S * sigma * np.sqrt(T))
        self.theta = -(S * norm.pdf(self.d1) * sigma) / (
                2 * np.sqrt(T)
        ) - option_type * r * K * np.exp(-r * T) * norm.cdf(option_type * self.d2)
        self.vega = S * np.sqrt(T) * norm.pdf(self.d1)
        self.rho = (
                option_type * K * T * np.exp(-r * T) * norm.cdf(option_type * self.d2)
        )
        self.price = option_type * (
                S * norm.cdf(option_type * self.d1)
                - K * np.exp(-r * T) * norm.cdf(option_type * self.d2)
        )


if __name__ == '__main__':
    K = 50
    r = 0.05
    sigma = 0.20
    S = np.array([
        49,
        49.375,
        49.75,
        50.875,
        52,
        51,
        50,
        49.19,
        48.38,
        48.315,
        48.25,
        48.5,
        48.75,
        49.19,
        49.63,
        48.94,
        48.25,
        48.25,
        48.25,
        49.685,
        51.12,
        51.31,
        51.5,
        50.69,
        49.88,
        49.88,
        49.88,
        49.315,
        48.75,
        48.125,
        47.5,
        47.75,
        48,
        47.125,
        46.25,
        47.19,
        48.13,
        47.38,
        46.63,
        47.375,
    ])
    T = np.arange(20, 0, -0.5) / 52
    option_type = 1  # 1 代表欧式看涨期权，-1 代表欧式看跌期权

    option = Option(K, r, sigma, S, T, option_type)
    print("Price:", option.price)
    print("Delta:", option.delta)
    print("Gamma:", option.gamma)
    print("Theta:", option.theta)
    print("Vega:", option.vega)
    print("Rho:", option.rho)
