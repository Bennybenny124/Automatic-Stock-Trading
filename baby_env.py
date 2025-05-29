# BabyEnv.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class BabyEnv(gym.Env):
    def __init__(self, price, obs_window=252, trade_window=252):
        super().__init__()
        self.price = price
        self.obs_window = obs_window
        self.trade_window = trade_window

        self.action_space = spaces.Discrete(21)  # -10 ~ +10 股
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_window * 4,), dtype=np.float32)

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.start_index = np.random.randint(self.obs_window, len(self.price) - self.trade_window)
        self.current_step = self.start_index

        self.cash = 10000.0
        self.shares_held = 0

        self.action_history = np.zeros(self.obs_window, dtype=np.int32)
        self.holdings_history = np.zeros(self.obs_window, dtype=np.float32)
        self.cash_history = np.full(self.obs_window, self.cash, dtype=np.float32)

        self.prev_value = self.cash
        self.no_action_count = 0

        return self._get_observation(), {}

    def _get_observation(self):
        prices_window = self.price[self.current_step - self.obs_window : self.current_step]
        obs = np.stack([prices_window, self.action_history, self.holdings_history, self.cash_history], axis=0)
        return obs.flatten().astype(np.float32)

    def step(self, action):
        current_price = self.price[self.current_step]
        volume = action - 10  # -10~+10 股

        # Clamp volume by cash and shares
        if volume > 0:
            volume = min(volume, int(self.cash // current_price))
        elif volume < 0:
            volume = max(volume, -self.shares_held)

        self.cash -= volume * current_price
        self.shares_held += volume

        new_value = self.cash + self.shares_held * current_price
        reward = new_value - self.prev_value  # immediate value change

        # 非因果 reward 計算
        future_prices = self.price[self.current_step + 1 : self.current_step + 21]  # 未來 1 個月
        past_prices = self.price[self.current_step - 20 : self.current_step] if self.current_step - 20 >= 0 else []

        if volume > 0:
            # 1. 鼓勵未來價格上漲
            if len(future_prices) > 0:
                max_future_price = np.max(future_prices)
                reward += (max_future_price - current_price) * volume * 0.02
            # 2. 鼓勵在跌勢後買入
            if len(past_prices) > 0:
                min_past_price = np.min(past_prices)
                reward += (current_price - min_past_price) * volume * 0.01
        elif volume < 0:
            # 1. 鼓勵未來價格下跌
            if len(future_prices) > 0:
                min_future_price = np.min(future_prices)
                reward += (current_price - min_future_price) * abs(volume) * 0.02
            # 2. 鼓勵在漲勢後賣出
            if len(past_prices) > 0:
                max_past_price = np.max(past_prices)
                reward += (max_past_price - current_price) * abs(volume) * 0.01

        # 無動作懲罰
        if volume == 0:
            self.no_action_count += 1
            reward -= min(0.01 * self.no_action_count, 0.1)
        else:
            self.no_action_count = 0

        # 交易成本懲罰
        reward -= abs(volume) * current_price * 0.001

        self.prev_value = new_value

        # 更新歷史紀錄
        self.action_history = np.roll(self.action_history, -1)
        self.action_history[-1] = volume

        self.holdings_history = np.roll(self.holdings_history, -1)
        self.holdings_history[-1] = self.shares_held

        self.cash_history = np.roll(self.cash_history, -1)
        self.cash_history[-1] = self.cash

        self.current_step += 1
        terminated = self.current_step >= self.start_index + self.trade_window
        truncated = False

        return self._get_observation(), reward, terminated, truncated, {}

    def render(self):
        p = self.price[self.current_step] if self.current_step < len(self.price) else self.price[-1]
        print(f"Step: {self.current_step}, Price: {p:.2f}, Shares: {self.shares_held}, Cash: {self.cash:.2f}")

