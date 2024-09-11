from neuralforecast.models import PatchTST
from neuralforecast import NeuralForecast
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

spy = yf.Ticker("SPY")
spy_data = spy.history(start="2010-01-01", end="2023-12-31")
spy_data = spy_data.resample('M').last().reset_index()
spy_data.columns = ['ds', 'y'] + list(spy_data.columns[2:])
spy_data = spy_data[['ds', 'y']]
spy_data['unique_id'] = 'SPY'
spy_data['returns'] = spy_data['y'].pct_change()
spy_data = spy_data.dropna()

train_size = int(len(spy_data) * 0.8)
train_data = spy_data[:train_size]
test_data = spy_data[train_size:]

model = PatchTST(
    h=1,
    input_size=24,
    scaler_type='standard',
    max_steps=100
)

nf = NeuralForecast(
    models=[model],
    freq='M'
)

nf.fit(df=train_data)

test_size = len(test_data)
y_hat_test_price = pd.DataFrame()
current_train_data = train_data.copy()

future_predict = pd.DataFrame({'ds': [test_data['ds'].iloc[0]], 'unique_id': ['SPY']})

y_hat_price = nf.predict(current_train_data, futr_df=future_predict)
y_hat_test_price = pd.concat([y_hat_test_price, y_hat_price.iloc[[-1]]])

for i in range(test_size - 1):
    combined_data = pd.concat([current_train_data, test_data.iloc[[i]]])
    future_predict['ds'] = test_data['ds'].iloc[i + 1]
    y_hat_price = nf.predict(combined_data, futr_df=future_predict)
    y_hat_test_price = pd.concat([y_hat_test_price, y_hat_price.iloc[[-1]]])
    current_train_data = combined_data

predictions_prices = y_hat_test_price['PatchTST'].values
true_values = test_data['y'].values

mse = np.mean((predictions_prices - true_values)**2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(predictions_prices - true_values))

print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")

plt.figure(figsize=(12, 6))
plt.plot(train_data['ds'], train_data['y'], label='Training Data', color='blue')
plt.plot(test_data['ds'], true_values, label='True Values', color='green')
plt.plot(test_data['ds'], predictions_prices, label='Predictions', color='red')
plt.legend()
plt.title('SPY Stepwise Forecast using PatchTST')
plt.xlabel('Date')
plt.ylabel('SPY Price')
plt.show()


model = PatchTST(
    h=1,
    input_size=24,
    scaler_type='standard',
    max_steps=100
)

nf = NeuralForecast(
    models=[model],
    freq='M'
)

nf.fit(df=train_data[['ds', 'returns', 'unique_id']].rename(columns={'returns': 'y'}))

y_hat_test_ret = pd.DataFrame()
current_train_data = train_data[['ds', 'returns', 'unique_id']].rename(columns={'returns': 'y'}).copy()

future_predict = pd.DataFrame({'ds': [test_data['ds'].iloc[0]], 'unique_id': ['SPY']})

y_hat_ret = nf.predict(current_train_data, futr_df=future_predict)
y_hat_test_ret = pd.concat([y_hat_test_ret, y_hat_ret.iloc[[-1]]])

for i in range(test_size - 1):
    combined_data = pd.concat([current_train_data, test_data[['ds', 'returns', 'unique_id']].rename(columns={'returns': 'y'}).iloc[[i]]])
    future_predict['ds'] = test_data['ds'].iloc[i + 1]
    y_hat_ret = nf.predict(combined_data, futr_df=future_predict)
    y_hat_test_ret = pd.concat([y_hat_test_ret, y_hat_ret.iloc[[-1]]])
    current_train_data = combined_data

predicted_returns = y_hat_test_ret['PatchTST'].values
true_returns = test_data['returns'].values

predicted_prices_ret = []
for i, ret in enumerate(predicted_returns):
    if i == 0:
        last_true_price = train_data['y'].iloc[-1]
    else:
        last_true_price = test_data['y'].iloc[i-1]
    predicted_prices_ret.append(last_true_price * (1 + ret))

mse = np.mean((np.array(predicted_prices_ret) - true_values)**2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(np.array(predicted_prices_ret) - true_values))

print(f"MSE (transformed): {mse:.2f}")
print(f"RMSE (transformed): {rmse:.2f}")
print(f"MAE (transformed): {mae:.2f}")

plt.figure(figsize=(12, 6))
plt.plot(train_data['ds'], train_data['y'], label='Training Data', color='blue')
plt.plot(test_data['ds'], true_values, label='True Prices', color='green')
plt.plot(test_data['ds'], predicted_prices_ret, label='Predicted Prices', color='red')
plt.legend()
plt.title('SPY Stepwise Forecast using PatchTST (Prices)')
plt.xlabel('Date')
plt.ylabel('SPY Price')
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(test_data['ds'], predictions_prices - true_values, label='Price Prediction Errors (Direct)', color='red')
plt.plot(test_data['ds'], np.array(predicted_prices_ret) - true_values, label='Price Prediction Errors (Return First)', color='green')
plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
plt.legend()
plt.title('SPY Prediction Errors')
plt.xlabel('Date')
plt.ylabel('Price / Error')
plt.show()


spy_data['first_diff'] = spy_data['y'].diff()
spy_data['pct_change'] = spy_data['y'].pct_change()
spy_data['log'] = np.log(spy_data['y'])
spy_data =spy_data.drop(spy_data.index[0])
spy_data = spy_data.reset_index(drop=True)

def replace_inf_nan(series):
    if np.isnan(series.iloc[0]) or np.isinf(series.iloc[0]):
        series.iloc[0] = 0
    mask = np.isinf(series) | np.isnan(series)
    series = series.copy()
    series[mask] = np.nan
    series = series.ffill()
    return series


columns_to_test =  ['first_diff', 'pct_change','log', 'y']
for col in columns_to_test:
    spy_data[col] = replace_inf_nan(spy_data[col])

has_nan = np.isnan(spy_data[columns_to_test]).any().any()
has_inf = np.isinf(spy_data[columns_to_test]).any().any()

print(f"\nContains NaN: {has_nan}")
print(f"Contains inf: {has_inf}\n")



plt.figure(figsize=(10, 6))
plt.plot(spy_data.index, spy_data['y'])
plt.title('SPY Data - Original Values')
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(spy_data.index, spy_data['first_diff'])
plt.title('SPY Data - First Difference')
plt.xlabel('Index')
plt.ylabel('First Difference')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(spy_data.index, spy_data['pct_change'])
plt.title('SPY Data - Percentage Change')
plt.xlabel('Index')
plt.ylabel('Percentage Change')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(spy_data.index, spy_data['log'])
plt.title('SPY Data - Log Values')
plt.xlabel('Index')
plt.ylabel('Log Value')
plt.grid(True)
plt.show()


from statsmodels.tsa.stattools import adfuller
from tabulate import tabulate

results = []

for column in columns_to_test:
    result = adfuller(spy_data[column].dropna())
    results.append([column, result[0], result[1]])

headers = ["Column", "ADF Statistic", "p-value"]
table = tabulate(results, headers=headers, floatfmt=".5f", tablefmt="grid")

print("Augmented Dickey-Fuller Test Results:")
print(table)
