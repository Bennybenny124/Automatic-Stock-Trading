import os
import json
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from env import StockTradingEnv  # 確保這是你定義好的環境

# === 參數設定 ===
window_size = 252
net_arch = [64, 64]
total_years = 10      # 假設每支股票大約有10年資料
train_epochs = 10     # 每支股票訓練10輪等效的長度

# === 載入多支股票資料 ===
with open("stocks/stocks.json", "r") as f:
    stock_files = json.load(f)["general"]

price_list = []
for file in stock_files:
    df = pd.read_csv(f"stocks/{file}.csv")
    prices = df["Close"].values
    if len(prices) >= window_size + 10:
        price_list.append(prices)

# === 包裝成 VecEnv ===
def make_env(price):
    def _init():
        env = StockTradingEnv(price, window_size=window_size)
        return Monitor(env)
    return _init

vec_env = DummyVecEnv([make_env(p) for p in price_list])

# === 建立並訓練 PPO 模型 ===
model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    policy_kwargs=dict(net_arch=net_arch),
    device="cpu",
    ent_coef=0.1
)

steps_per_env = (total_years * 252 - window_size)
model.learn(total_timesteps=steps_per_env * train_epochs)

# === 儲存模型 ===
os.makedirs("models", exist_ok=True)
model.save("models/multi_stock_trader")
