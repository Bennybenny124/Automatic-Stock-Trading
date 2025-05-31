import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env import StockTradingEnv

# === 測試參數 ===
window_size = 252
model_path = "models/multi_stock_trader"

# === 載入測試股票資料（以第一支為例）===
with open("stocks/stocks.json", "r") as f:
    stock_files = json.load(f)["general"]

test_file = stock_files[0]
df = pd.read_csv(f"stocks/{test_file}.csv")
df = pd.read_csv("stocks/INTC.csv")
prices = df["Close"].values

# === 建立測試環境 ===
env = StockTradingEnv(prices, window_size=window_size)
model = PPO.load(model_path)

# === 模擬測試，記錄資產變化 ===
obs, _ = env.reset()
done = False

cash_list = [env.cash]
shares_list = [env.shares_held]
value_list = [env.cash + env.shares_held * prices[env.current_step]]
price_list = [env.price[env.current_step]]
step_list = [env.current_step]

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    total_value = env.cash + env.shares_held * prices[env.current_step]
    done = terminated or truncated

    cash_list.append(env.cash)
    shares_list.append(env.shares_held)
    value_list.append(total_value)
    price_list.append(prices[env.current_step])
    step_list.append(env.current_step)

# === 畫圖 ===
plt.figure(figsize=(12, 6))
plt.plot(step_list, value_list, label="Total Asset Value")
plt.plot(step_list, cash_list, label="Cash")
plt.plot(step_list, np.array(shares_list) * np.array(price_list), label="Stock Holdings Value")
plt.plot(step_list, price_list, label="Stock Price", linestyle="--", alpha=0.4)
plt.title("Agent Asset Components During Test")
plt.xlabel("Step")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
