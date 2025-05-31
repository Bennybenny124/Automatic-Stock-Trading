import gymnasium as gym
from gymnasium import spaces
import numpy as np

class StockTradingEnv(gym.Env):
    def __init__(self, stock_data):
        super().__init__()
        self.stock_data = {k: v for k, v in stock_data.items() if not v.empty}
        self.tickers = list(self.stock_data.keys())
        assert self.tickers, "No valid stock data"

        sample_df = next(iter(self.stock_data.values()))
        self.n_features = len(sample_df.columns)

        self.action_space = spaces.Box(low=-1, high=1, shape=(len(self.tickers),), dtype=np.float32)
        self.obs_shape = self.n_features * len(self.tickers) + 2 + len(self.tickers) + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_shape,), dtype=np.float32)

        self.initial_balance = 1000
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.shares_held = {t: 0 for t in self.tickers}
        self.total_shares_sold = {t: 0 for t in self.tickers}
        self.total_sales_value = {t: 0 for t in self.tickers}
        self.current_step = 0
        self.max_steps = min(len(df) for df in self.stock_data.values()) - 1
        return self._next_observation(), {}

    def _next_observation(self):
        obs = np.zeros(self.obs_shape)
        idx = 0
        for ticker in self.tickers:
            df = self.stock_data[ticker]
            obs[idx:idx+self.n_features] = df.iloc[min(self.current_step, len(df)-1)].values
            idx += self.n_features
        obs[-4-len(self.tickers)] = self.balance
        obs[-3-len(self.tickers):-3] = [self.shares_held[t] for t in self.tickers]
        obs[-3] = self.net_worth
        obs[-2] = self.max_net_worth
        obs[-1] = self.current_step
        return obs

    def step(self, actions):
        self.current_step += 1
        done = self.current_step >= self.max_steps
        current_prices = {
            t: self.stock_data[t].iloc[self.current_step]["Close"]
            for t in self.tickers
        }

        for i, t in enumerate(self.tickers):
            price = current_prices[t]
            act = actions[i]
            if act > 0:
                shares = int(self.balance * act / price)
                self.balance -= shares * price
                self.shares_held[t] += shares
            elif act < 0:
                shares = int(self.shares_held[t] * -act)
                self.balance += shares * price
                self.shares_held[t] -= shares
                self.total_shares_sold[t] += shares
                self.total_sales_value[t] += shares * price

        self.net_worth = self.balance + sum(
            self.shares_held[t] * current_prices[t] for t in self.tickers
        )
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        reward = self.net_worth - self.initial_balance
        return self._next_observation(), reward, done, False, {}

    def render(self):
        print(f"Step {self.current_step}: Net Worth = {self.net_worth:.2f}, Balance = {self.balance:.2f}")
