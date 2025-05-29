import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env import StockTradingEnv  # 或改成你實際的檔名

# === 載入資料 ===
ticker = "AAPL"
df = pd.read_csv(f"stocks/{ticker}.csv")
df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
stock_data = {ticker: df}

# === 建立環境與模型 ===
env = StockTradingEnv(stock_data)
model = PPO.load("models/PPO")

obs, _ = env.reset()
done = False

history = {
    "step": [],
    "net_worth": [],
    "balance": [],
    "price": [],
    "shares": [],
}

# === 測試 ===
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, truncated, _ = env.step(action)

    price = df.iloc[min(env.current_step, len(df) - 1)]["Close"]
    shares = env.shares_held[ticker]

    history["step"].append(env.current_step)
    history["net_worth"].append(env.net_worth)
    history["balance"].append(env.balance)
    history["price"].append(price)
    history["shares"].append(shares)

# === 畫圖 ===
steps = history["step"]

plt.figure(figsize=(14, 10))

plt.subplot(3, 1, 1)
plt.plot(steps, history["price"], label="Price")
plt.title("Stock Price")

plt.subplot(3, 1, 2)
plt.plot(steps, history["net_worth"], label="Net Worth", color="purple")
plt.title("Net Worth Over Time")

plt.subplot(3, 1, 3)
plt.plot(steps, history["shares"], label="Shares Held", color="green")
plt.title("Shares Held Over Time")

plt.tight_layout()
plt.show()
