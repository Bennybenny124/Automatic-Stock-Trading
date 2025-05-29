# BabyTrainer.py
import os
import json
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from baby_env import BabyEnv

# === 參數設定 ===
obs_window = 252
trade_window = 252
train_epochs = 100

# === 載入資料 ===
with open("stocks/stocks.json", "r") as f:
    stock_files = json.load(f)["baby"]

price_list = []
for file in stock_files:
    df = pd.read_csv(f"stocks/{file}.csv")
    prices = df["Close"].values
    if len(prices) >= obs_window + trade_window:
        price_list.append(prices)

# === 環境包裝器 ===
def make_env(price):
    def _init():
        return Monitor(BabyEnv(price, obs_window=obs_window, trade_window=trade_window))
    return _init

# === 單支股票逐一訓練（循環） ===
model = PPO.load("models/BabyTrader.zip")
for epoch in range(train_epochs):
    for price in price_list:
        env = DummyVecEnv([make_env(price)])
        if model is None:
            model = PPO("MlpPolicy", env, verbose=1, ent_coef=0.01, device="cpu")
        else:
            model.set_env(env)

        model.learn(total_timesteps=trade_window)

# === 儲存模型 ===
model.save("models/BabyTrader")
