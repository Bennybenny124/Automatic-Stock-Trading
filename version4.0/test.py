import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env import StockTradingEnv

# === 載入資料 ===
ticker = "AAPL"

df = pd.read_csv(f"stocks/{ticker}.csv")

offset = 2 * 252  # 兩年前以前的資料全部跳過，只用後面八年
df = df.iloc[offset:].reset_index(drop=True)

df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA20', 'MA60', 'MA240',
        'DIF', 'MACD', 'RSI', 'ATR', 'ADX', 'Bollinger_Upper', 'Bollinger_Lower', '%K', '%D']].dropna()
stock_data = {ticker: df}

# === 建立環境與模型 ===
#env = StockTradingEnv(stock_data)
env = StockTradingEnv(df)
model = PPO.load("models/PPO_v4", device="cpu")

obs, _ = env.reset()
done = False

history = {
    "step": [],
    "net_worth": [],
    "balance": [],
    "price": [],
    "shares": [],
    "action": [],
}

# === 測試 ===
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, truncated, _ = env.step(action)

    price = df.iloc[min(env.current_step, len(df) - 1)]["Close"]
    shares = env.shares_held

    history["step"].append(env.current_step)
    history["net_worth"].append(env.net_worth)
    history["balance"].append(env.balance)
    history["price"].append(price)
    history["shares"].append(shares)
    history["action"].append(obs[-5])

# === 畫圖 ===
steps = history["step"]

plt.figure(figsize=(14, 10))

plt.subplot(3, 1, 1)
plt.plot(steps, history["price"], label="Price", color="blue")
plt.title("Stock Price")

plt.subplot(3, 1, 2)
plt.plot(steps, history["net_worth"], label="Net Worth", color="red")
plt.plot(steps, history["balance"], label="Balance", color="yellow")
plt.title("Net Worth Over Time")

plt.subplot(3, 1, 3)
plt.plot(steps, history["shares"], label="Shares Held", color="green")
plt.plot(steps, history["action"], label="Traded Volume", color="blue")
plt.title("Shares Held Over Time")

plt.tight_layout()
plt.show()
