# baby_tester.py
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from baby_env import BabyEnv

# === 載入資料 ===
with open("stocks/stocks.json", "r") as f:
    stock_file = json.load(f)["baby"][0]  # 只測第一支股票
df = pd.read_csv(f"stocks/" + stock_file + ".csv")
price = df["Close"].values

# === 參數設定 ===
obs_window = 252
trade_window = 252
env = BabyEnv(price, obs_window=obs_window, trade_window=trade_window)

# === 載入模型 ===
model = PPO.load("models/BabyTrader")

# === 測試 ===
obs, _ = env.reset(seed=42)
done = False

history = {
    "price": [],
    "cash": [],
    "shares": [],
    "value": [],
    "action": [],
    "step": [],
}

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    current_price = env.price[env.current_step - 1]
    value = env.cash + env.shares_held * current_price

    history["step"].append(env.current_step)
    history["price"].append(current_price)
    history["cash"].append(env.cash)
    history["shares"].append(env.shares_held)
    history["value"].append(value)
    history["action"].append(action)

# === 畫圖 ===
steps = history["step"]
price = history["price"]
cash = history["cash"]
value = history["value"]
actions = history["action"]

buy_steps = [s for s, a in zip(steps, actions) if a > 0]
sell_steps = [s for s, a in zip(steps, actions) if a < 0]
buy_prices = [p for p, a in zip(price, actions) if a > 0]
sell_prices = [p for p, a in zip(price, actions) if a < 0]

plt.figure(figsize=(14, 8))

# 圖1: 股價 + 買賣點
plt.subplot(4, 1, 1)
plt.plot(steps, price, label="Price")
plt.title("Price")
plt.legend()

# 圖2: 現金變化
plt.subplot(4, 1, 2)
plt.plot(steps, cash, label="Cash", color="orange")
plt.title("Cash Over Time")

plt.subplot(4, 1, 3)
plt.plot(steps, actions, label="actions")
plt.title("Actions taken")

# 圖3: 總資產變化
plt.subplot(4, 1, 4)
plt.plot(steps, value, label="Total Value", color="purple")
plt.title("Portfolio Value Over Time")

plt.tight_layout()
plt.show()
