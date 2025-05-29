import pandas as pd
import numpy as np

def clamp(value, upperbound = 1, lowerbound = -1):
    if value >= upperbound: return upperbound
    elif value <= lowerbound: return lowerbound
    else: return value

def get_features(dataframe, filename):
    df = dataframe.copy()

    indicators = pd.DataFrame()

    # --- Moving Averages ---
    indicators['MA5'] = df['Close'].rolling(window=5).mean()
    indicators['MA20'] = df['Close'].rolling(window=20).mean()
    indicators['MA60'] = df['Close'].rolling(window=60).mean()
    indicators['MA240'] = df['Close'].rolling(window=240).mean()

    # --- MACD ---
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    indicators['DIF'] = ema12 - ema26
    indicators['MACD'] = indicators['DIF'].ewm(span=9, adjust=False).mean()

    # --- RSI ---
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    indicators['RSI'] = 100 - (100 / (1 + rs))

    # --- ADX ---
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr = pd.concat([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1))
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    indicators['ATR'] = atr

    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    tr14 = tr.rolling(14).sum()
    plus_di = 100 * (plus_dm.rolling(14).sum() / tr14)
    minus_di = 100 * (-minus_dm.rolling(14).sum() / tr14)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    indicators['ADX'] = dx.rolling(14).mean()

    # --- Bollinger Bands ---
    ma20 = indicators['MA20']
    std20 = df['Close'].rolling(window=20).std()
    indicators['Bollinger_Upper'] = ma20 + 2 * std20
    indicators['Bollinger_Lower'] = ma20 - 2 * std20

    # --- Stochastic Oscillator ---
    low14 = df['Low'].rolling(window=14).min()
    high14 = df['High'].rolling(window=14).max()
    indicators['%K'] = 100 * (df['Close'] - low14) / (high14 - low14)
    indicators['%D'] = indicators['%K'].rolling(window=3).mean()

    # 合併進 df
    df = pd.concat([df, indicators], axis=1)
    df.to_csv(filename, index=False)

def get_trend(df, n):
    row = df.iloc[n]  # 取第n天的指標數據
    price = row['Close']
    trend = 0

    ma = clamp((row['MA5'] - row['MA20']) / price * 10) # ok
    macd = clamp((row['DIF'] - row['MACD']) / price * 80) # ok
    bollinger = clamp(2 * (row['MA5'] - row['Bollinger_Lower']) / (row['Bollinger_Upper'] - row['Bollinger_Lower']) - 1) # ok
    rsi = clamp(row['RSI'] / 50 - 1)
    kd = (1 if row['%D'] < 80 else 0.5) if row['%K'] > row['%D'] else (-1 if row['%D'] > 20 else -0.5)

    clamp((row['%K'] - row['%D']) / 50) # ok
    
    # 波動指標 
    atr = clamp(row['ATR'] / price * 10, 1, 0)
    adx = 1 if (row['ADX'] >= 25) else row['ADX'] / 25
    
    trend += 0.1 * ma + 0.1 * macd + 0.4 * bollinger + 0.4 * rsi
    trend *= adx

    return (trend, ma, macd, bollinger, rsi)

