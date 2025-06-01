# This version is armed with an advanced reward function, and works well in bear markets.
# All reward are calculated by the actual cost/income(in dollars), then scaled down by the initial balance.
# Smart holding is rewarded in this version, which has reduced frequent transaction.

import gymnasium as gym
from gymnasium import spaces
import numpy as np

class StockTradingEnv(gym.Env):
    def __init__(self, df):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.n_features = self.df.shape[1]
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        self.window_size = 20
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_features * self.window_size + 5,),
            dtype=np.float32,
        )
        
        self.action = 0
        self.initial_balance = 10000
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.shares_held = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.current_step = self.window_size - 1
        self.max_steps = len(self.df) - 1
        return self._next_observation(), {}

    def _next_observation(self):
        obs = np.zeros(self.n_features * self.window_size + 5, dtype=np.float32)
        for i in range(self.window_size):
            step_idx = self.current_step - (self.window_size - 1) + i
            obs[i * self.n_features : (i + 1) * self.n_features] = self.df.iloc[step_idx].values
        
        obs[-5] = self.action
        obs[-4] = self.balance
        obs[-3] = self.shares_held
        obs[-2] = self.net_worth
        obs[-1] = self.max_net_worth
        return obs

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= self.max_steps

        current_price = self.df.iloc[self.current_step]["Close"]
        act = float(action[0])
        reward = 0

        # Peaking the future (for training purpose)
        future_window = 20
        if self.current_step + future_window < len(self.df):
            future_prices = self.df.iloc[self.current_step + 1 : self.current_step + 1 + future_window]["Close"]
            avg_future_price = future_prices.mean()
        else:
            avg_future_price = current_price

        volume = 0
        if act > 0:
            volume = int(self.balance * act / current_price)
            cost = volume * current_price
            self.balance -= cost
            self.shares_held += volume
            reward -= cost
            reward += (avg_future_price - current_price) * volume # Opportunity cost

        elif act < 0:
            volume = int(self.shares_held * -act)
            revenue = volume * current_price
            self.balance += revenue
            self.shares_held -= volume
            reward += revenue
            reward += (current_price - avg_future_price) * volume # OC

        self.action = volume if act > 0 else -volume # The true transaction amount for observation

        # Reward smart handling
        if volume < 0.05:
            reward += (avg_future_price - current_price) * self.shares_held
        else:
            reward -= 1 if volume * current_price * 0.001 < 1 else volume * current_price * 0.001 # A 0.1% handling charge

        interest_rate = 0.0001 # A 2.5% anual interest rate
        reward += self.balance * interest_rate # An Interest rate encourages proper selling
        reward /= self.initial_balance # scale down the reward for the ease of the model

        # Update the assets
        self.net_worth = self.balance + self.shares_held * current_price
        self.max_net_worth = max(self.max_net_worth, self.net_worth)

        return self._next_observation(), reward, done, False, {}

    def render(self):
        print(f"Step {self.current_step}: Net Worth = {self.net_worth:.2f}, Balance = {self.balance:.2f}")
