import numpy as np
import gymnasium as gym
from gymnasium import spaces

class StockTradingEnv(gym.Env):
    def __init__(self, price, window_size=252):
        super().__init__()
        self.price = price
        self.window_size = window_size

        # 離散動作：從 -100% 到 +100%（共 21 種）
        self.action_space = spaces.Discrete(21, start = -10)

        # Observation: 每日價格、每日持股數、每日現金 -> 共 window_size * 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window_size * 4,), dtype=np.float32)

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.cash = 10000.0
        self.shares_held = 0

        self.action_history = np.zeros(self.window_size, dtype=np.int32)
        self.holdings_history = np.zeros(self.window_size, dtype=np.float32)
        self.cash_history = np.full(self.window_size, self.cash, dtype=np.float32)
        self.max_step = len(self.price) - 1

        return self._get_observation(), {}

    def _get_observation(self):
        prices_window = self.price[self.current_step - self.window_size : self.current_step]
        obs = np.stack([prices_window, self.action_history, self.holdings_history, self.cash_history], axis=0)
        return obs.flatten().astype(np.float32)

    def step(self, action):
        current_price = self.price[self.current_step]
        total_value = self.cash + self.shares_held * current_price

        volume = int((total_value * action / 10.0) // current_price)

        if volume > 0:
            max_affordable = int(self.cash // current_price)
            volume = min(volume, max_affordable)
        elif volume < 0:
            volume = max(volume, -self.shares_held)
        else: volume = 0

        self.cash -= volume * current_price
        self.shares_held += volume

        # 更新歷史資料
        self.action_history = np.roll(self.action_history, -1)
        self.action_history[-1] = action

        self.holdings_history = np.roll(self.holdings_history, -1)
        self.holdings_history[-1] = self.shares_held

        self.cash_history = np.roll(self.cash_history, -1)
        self.cash_history[-1] = self.cash

        # 計算 reward：與兩週前價值相比
        prev_index = self.current_step - 10
        if volume == 0:
            reward = 0.0
            i = 0
            while self.action_history[-1-i] == 0 and i + 1 < self.window_size:
                reward -= 1 * i
                i += 1
        else:
            past_price = self.price[prev_index]
            reward = (current_price - past_price) / (past_price + 1e-8) * volume / total_value
            reward -= abs(volume) * current_price * 0.001

        if volume == 0:
            reward = 0.0
            idle_steps = 0
            for i in range(1, min(self.window_size, 20)):
                if self.action_history[-i] == 0:
                    idle_steps += 1
                else:
                    break
            reward -= idle_steps * 0.01  # 最多罰 0.2
        else:
            past_price = self.price[self.current_step - 10]
            gain = (current_price - past_price) / (past_price + 1e-8) * volume / total_value
            reward = 10 * gain - abs(volume) * current_price * 0.001

            
        self.current_step += 1
        terminated = self.current_step >= self.max_step
        truncated = False

        return self._get_observation(), reward, terminated, truncated, {}

    def render(self):
        p = self.price[self.current_step] if self.current_step < len(self.price) else self.price[-1]
        print(f"Step: {self.current_step}, Price: {p:.2f}, Shares: {self.shares_held}, Cash: {self.cash:.2f}")

