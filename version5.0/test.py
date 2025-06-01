import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env import StockTradingEnv

# === 載入資料 ===
ticker = "AAPL"
df = pd.read_csv(f"stocks/{ticker}.csv")
offset = 250  # 跳過前250天的資料
df = df.iloc[offset:].reset_index(drop=True)

# === 使用的特徵 ===
df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA20', 'MA60', 'MA240',
        'DIF', 'MACD', 'RSI', 'ATR', 'ADX', 'Bollinger_Upper', 'Bollinger_Lower', '%K', '%D']].dropna()
stock_data = {ticker: df}

# === 建立環境與模型 ===
env = StockTradingEnv(df)
model = PPO.load("models/PPO_all_stocks_3days_fee")
obs, _ = env.reset()
done = False

n_trials = 10
history_avg = {
    "net_worth": [],
    "balance": [],
    "price": [],
    "shares": [],
}

# 因為每次測試步數相同，所以先初始化一個陣列來累加結果
net_worth_accum = None
balance_accum = None
price_accum = None
shares_accum = None

for trial in range(n_trials):
    obs, _ = env.reset()
    done = False
    
    net_worth_list = []
    balance_list = []
    price_list = []
    shares_list = []
    
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, _ = env.step(action)
        
        step_idx = min(env.current_step, len(df) - 1)
        price = df.iloc[step_idx]["Close"]
        shares = env.shares_held
        
        net_worth_list.append(env.net_worth)
        balance_list.append(env.balance)
        price_list.append(price)
        shares_list.append(shares)
    
    # 第一次初始化累加陣列大小
    if net_worth_accum is None:
        net_worth_accum = np.array(net_worth_list)
        balance_accum = np.array(balance_list)
        price_accum = np.array(price_list)
        shares_accum = np.array(shares_list)
    else:
        net_worth_accum += np.array(net_worth_list)
        balance_accum += np.array(balance_list)
        price_accum += np.array(price_list)
        shares_accum += np.array(shares_list)

# 計算平均
history_avg["net_worth"] = net_worth_accum / n_trials
history_avg["balance"] = balance_accum / n_trials
history_avg["price"] = price_accum / n_trials
history_avg["shares"] = shares_accum / n_trials

# === 畫圖 ===
steps = list(range(len(history_avg["net_worth"])))

plt.figure(figsize=(14, 10))

plt.subplot(3, 1, 1)
plt.plot(steps, history_avg["price"], label="Price")
plt.title("Stock Price")

plt.subplot(3, 1, 2)
plt.plot(steps, history_avg["net_worth"], label="Net Worth", color="purple")
plt.title("Net Worth Over Time")

plt.subplot(3, 1, 3)
plt.plot(steps, history_avg["shares"], label="Shares Held", color="green")
plt.title("Shares Held Over Time")

plt.tight_layout()
plt.show()
