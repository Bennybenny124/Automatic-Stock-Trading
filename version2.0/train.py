import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env import StockTradingEnv  # 或改成你放的檔名

def load_data():
    tickers = ['AAPL']  # 測一支股票即可
    stock_data = {}
    for ticker in tickers:
        df = pd.read_csv(f'stocks/{ticker}.csv')
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        stock_data[ticker] = df
    return stock_data

def main():
    data = load_data()
    env = DummyVecEnv([lambda: StockTradingEnv(data)])
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("models/PPO")

if __name__ == "__main__":
    main()
