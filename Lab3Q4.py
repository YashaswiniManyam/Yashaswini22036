import pandas as pd
import matplotlib.pyplot as plt
import statistics

df_stock_price = pd.read_excel('Lab Session1 Data.xlsx', sheet_name='IRCTC Stock Price')

price_mean = statistics.mean(df_stock_price['Price'])
price_variance = statistics.variance(df_stock_price['Price'])

print(f"Mean of Price: {price_mean}")
print(f"Variance of Price: {price_variance}")

wed_prices = df_stock_price[df_stock_price['Day'] == 'Wed']['Price']
print(wed_prices.values)
wed_mean = statistics.mean(wed_prices)

print(f"Sample Mean on Weds: {wed_mean}")

april_prices = df_stock_price[df_stock_price['Month'] == 'Apr']['Price']
april_mean = statistics.mean(april_prices)

print(f"Sample Mean for April: {april_mean}")

# Probability of making a loss over the stock
loss_probability = len(df_stock_price[df_stock_price['Chg%'] < 0]) / len(df_stock_price)
print(f"Probability of making a loss: {loss_probability}")