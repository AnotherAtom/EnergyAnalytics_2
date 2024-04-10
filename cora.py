import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.activations import relu

# We load the spot prices
file_Path = os.path.join(os.getcwd(),'Elspotprices2.csv')
df_prices = pd.read_csv(file_Path)

# We convert the HourUTC column to datetime and set the column only at the DK2 price area
df_prices["HourUTC"] = pd.to_datetime(df_prices["HourUTC"])
df_prices = df_prices.loc[(df_prices['PriceArea']=="DK2")][["HourUTC","SpotPriceDKK"]]
df_prices = df_prices.loc[df_prices["HourUTC"].dt.year.isin([2019,2020,2021,2022,2023])]
df_prices = df_prices.reset_index(drop=True)

# We load the production and consumption data
file_Path = os.path.join(os.getcwd(),'ProdConData.csv')
df_data = pd.read_csv(file_Path)

# "e convert the HourUTC column to datetime
df_data["HourUTC"] = pd.to_datetime(df_data["HourUTC"])
df_data = df_data.loc[(df_data['PriceArea']=="DK2")]#[["HourUTC","ExchangeSE_MWh","GrossConsumptionMWh","OffshoreWindGe100MW_MWh"]]
df_data = df_data.loc[df_data["HourUTC"].dt.year.isin([2019,2020,2021,2022,2023])]
df_data = df_data.reset_index(drop=True)

data_mearge = pd.merge(df_prices, df_data, on='HourUTC')



import seaborn as sns
data_corr = data_mearge.drop(['HourUTC','ExchangeNO_MWh','ExchangeNL_MWh','ExchangeGB_MWh'], axis=1)
correlation_matrix = data_corr.corr()
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

print(correlation_matrix['SpotPriceDKK'])