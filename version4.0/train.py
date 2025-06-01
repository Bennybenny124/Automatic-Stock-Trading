import pandas as pd
import json
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv
from env import StockTradingEnv

def load_data():
    with open('stocks/stocks.json', 'r') as f:
        tickers = json.load(f)["general"]

    stock_data = {}
    for ticker in tickers:
        df = pd.read_csv(f'stocks/{ticker}.csv')
        df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA20', 'MA60', 'MA240',
                 'DIF', 'MACD', 'RSI', 'ATR', 'ADX', 'Bollinger_Upper', 'Bollinger_Lower', '%K', '%D']]
        df = df.dropna().reset_index(drop=True)
        if len(df) > 250:
            stock_data[ticker] = df.iloc[250:].reset_index(drop=True)
    return stock_data

def main():
    data = load_data()
    model = PPO.load("models/PPO_all_stocks_10days_with_advanced_reward", device="cpu")

    policy_kwargs = dict(
        net_arch=[64, 64]
    )

    for i, (ticker, df) in enumerate(data.items()):
        print(f"訓練第{i+1}支股票：{ticker}")
        env = DummyVecEnv([lambda: StockTradingEnv(df)])
        if model is None:
            model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs, device="cpu")
        model.set_env(env)
        model.learn(total_timesteps=5000)

    model.save("models/PPO_all_stocks_10days_with_advanced_reward")

if __name__ == "__main__":
    main()
