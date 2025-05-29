import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from features import get_trend

df = pd.read_csv("AAPL.csv")
prices = df['Close'].values
n = len(prices)
conditions = np.zeros(n)
mas = np.zeros(n)
macds = np.zeros(n)
bollingers = np.zeros(n)
rsis = np.zeros(n)
kds = np.zeros(n)
baseline = np.zeros(n)

for i in range(n):
    conditions[i], mas[i], macds[i], bollingers[i], rsis[i] = get_trend(df, i)
    baseline[i] = 150

conditions = conditions*10+150
mas = mas*10+150
macds = macds*10+150
bollingers = bollingers*10+150
rsis = rsis*10+150
kds = kds*10+150

plt.figure()
plt.plot(prices)
plt.plot(conditions, label = 'eval')
plt.plot(mas, label = 'MA')
plt.plot(macds, label = 'MACD')
plt.plot(bollingers, label = 'Bollinger')
plt.plot(rsis, label = 'RSI')
plt.plot(kds, label = 'K/D')
plt.plot(baseline)
plt.legend()
plt.show()