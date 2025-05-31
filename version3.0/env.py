import gymnasium as gym
from gymnasium import spaces
import numpy as np

class StockTradingEnv(gym.Env):
    def __init__(self, df):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.n_features = self.df.shape[1]

        self.window_size = 10#new
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # self.observation_space = spaces.Box(
        #     low=-np.inf, high=np.inf, shape=(self.n_features + 4,), dtype=np.float32
        # )old

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_features * self.window_size + 4,),
            dtype=np.float32,
        )#new
        
        self.initial_balance = 1000
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.shares_held = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        #self.current_step = 0 old
        self.current_step = self.window_size - 1 #new
        self.max_steps = len(self.df) - 1
        return self._next_observation(), {}

    def _next_observation(self):
        # obs = np.zeros(self.n_features + 4)
        # obs[:self.n_features] = self.df.iloc[self.current_step].values old
        obs = np.zeros(self.n_features * self.window_size + 4, dtype=np.float32)
        for i in range(self.window_size):
            step_idx = self.current_step - (self.window_size - 1) + i
            obs[i * self.n_features : (i + 1) * self.n_features] = self.df.iloc[step_idx].values #new
        
        obs[-4] = self.balance
        obs[-3] = self.shares_held
        obs[-2] = self.net_worth
        obs[-1] = self.max_net_worth
        return obs

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= self.max_steps

        current_price = self.df.iloc[self.current_step]["Close"]
        act = action[0]

        # if act > 0:
        #     shares = int(self.balance * act / current_price)
        #     self.balance -= shares * current_price
        #     self.shares_held += shares
        # elif act < 0:
        #     shares = int(self.shares_held * -act)
        #     self.balance += shares * current_price
        #     self.shares_held -= shares
        #     self.total_shares_sold += shares
        #     self.total_sales_value += shares * current_price

        # self.net_worth = self.balance + self.shares_held * current_price
        # self.max_net_worth = max(self.max_net_worth, self.net_worth)
        # reward = self.net_worth - self.initial_balance 
        # old

        reward = 0
        window = 5  # 可調整為你想看的未來天數
        future_index = self.current_step + window
        if future_index < len(self.df):
            future_ma5 = self.df.iloc[future_index]["MA5"]

        if act > 0:
            # 買進
            shares = int(self.balance * act / current_price)
            self.balance -= shares * current_price
            self.shares_held += shares

            if self.current_step + window < len(self.df):
                # future_prices = self.df.iloc[self.current_step+1 : self.current_step+1+window]["Close"]
                # avg_future_price = future_prices.mean()
                reward = future_ma5 - current_price

        elif act < 0:
            # 賣出
            shares = int(self.shares_held * -act)
            self.balance += shares * current_price
            self.shares_held -= shares
            self.total_shares_sold += shares
            self.total_sales_value += shares * current_price

            if self.current_step + window < len(self.df):
                # future_prices = self.df.iloc[self.current_step+1 : self.current_step+1+window]["Close"]
                # avg_future_price = future_prices.mean()
                reward = current_price - future_ma5

        self.net_worth = self.balance + self.shares_held * current_price
        self.max_net_worth = max(self.max_net_worth, self.net_worth)#new

        return self._next_observation(), reward, done, False, {}

    def render(self):
        print(f"Step {self.current_step}: Net Worth = {self.net_worth:.2f}, Balance = {self.balance:.2f}")
