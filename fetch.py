# Fetches market information from Yahoo Finance
import yfinance as yf
import json
from features import get_features

group = "general"

with open("stocks/stocks.json", "r") as f:
    stocks_from = json.load(f)

for stock in stocks_from[group]:
    ticker = yf.Ticker(stock)
    data = ticker.history(period="10y", interval="1d")
    get_features(data, "stocks/" + stock + ".csv")