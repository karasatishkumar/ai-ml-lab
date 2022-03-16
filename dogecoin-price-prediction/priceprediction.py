import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import regression
sns.set()
plt.style.use('seaborn-whitegrid')

# 1. Read the historical data
data = pd.read_csv("DOGE-USD.csv")
print(data.head())

# 2. We are going to use the close price for our prediction. Plot a graph with date and close price

data.dropna()
plt.figure(figsize=(10, 4))
plt.title("DogeCoin Price USD")
plt.xlabel("Date")
plt.ylabel("Close")
plt.plot(data["Date"], data["Close"])
plt.show()

# 3. Train the model
from autots import AutoTS
model = AutoTS(forecast_length=10, frequency='infer', ensemble='simple', drop_data_older_than_periods=200)
model = model.fit(data, date_col='Date', value_col='Close', id_col=None)

# 4. Get the prediction
prediction = model.predict()
forecast = prediction.forecast
print("DogeCoin Price Prediction")
print(forecast)