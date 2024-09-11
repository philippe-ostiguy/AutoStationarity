import yfinance as yf
import matplotlib.pyplot as plt
spy = yf.Ticker("SPY")
data = spy.history(period="10y")

plt.figure(figsize=(10, 5))
plt.plot(data.index, data['Close'], label='SPY Close Price')
plt.title('SPY Closing Prices (Last 10 Years)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()

monthly_returns = data['Close'].pct_change().dropna()

plt.figure(figsize=(10, 5))
plt.plot(monthly_returns.index, monthly_returns, label='SPY Monthly Return')
plt.title('SPY Monthly Returns (Last 10 Years)')
plt.xlabel('Date')
plt.ylabel('Monthly Return')
plt.legend()
plt.show()


from statsmodels.tsa.stattools import adfuller
def get_p_value(series):
    result = adfuller(series, autolag='AIC')
    return result[1]

price_p_value = get_p_value(data['Close'])

returns_p_value = get_p_value(monthly_returns)

print(f"\np-value for stock price: {price_p_value}")
print(f"p-value for monthly returns: {returns_p_value}")


