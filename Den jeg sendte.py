# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 09:26:03 2024

@author: Nicolai Norregaard
"""

#%% Load the Data
import numpy as np
import matplotlib.pyplot as plt
import pmdarima as pm
from pmdarima import pipeline, arima, model_selection
# from sklearn.metrics import root_mean_squared_error
import os
import pandas as pd
import datetime as dt
file_P = os.path.join(os.getcwd(),'data/Elspotprices2.csv')
df_data = pd.read_csv(file_P)
df_data["HourUTC"] = pd.to_datetime(df_data["HourUTC"])
df_data = df_data.loc[df_data["HourUTC"].dt.year.isin([2023])]
df_data = df_data.loc[(df_data['PriceArea']=="DK2")]
df_data = df_data.reset_index(drop=True)
file_C = os.path.join(os.getcwd(),'data/ProdConData.csv')
df_exogenous = pd.read_csv(file_C)
df_exogenous["HourUTC"] = pd.to_datetime(df_exogenous["HourUTC"])
df_exogenous = df_exogenous.loc[df_exogenous["HourUTC"].dt.year.isin([2023])]
df_exogenous = df_exogenous[df_exogenous['PriceArea'] == 'DK2']
df_exogenous = df_exogenous.reset_index(drop=True)
merged_df = pd.merge(df_data, df_exogenous)
merged_df = merged_df.loc[(merged_df['PriceArea']=="DK2")]
#%% Filter the training set
t_s = pd.Timestamp(dt.datetime(2023, 1, 1, 0, 0, 0))
t_e = pd.Timestamp(dt.datetime(2023, 12, 31, 23, 0, 0))
t_e_training = pd.Timestamp(dt.datetime(2023, 10, 1, 0, 0, 0))
n_test= len(merged_df.loc[(merged_df['HourUTC']>=t_e_training) & (merged_df['HourUTC']<=t_e)])
n_training=len(merged_df.loc[(merged_df['HourUTC']<=t_e_training)])
data = merged_df.loc[(merged_df['HourUTC']>=t_s) & (merged_df['HourUTC']<=t_e)]
data = data.reset_index(drop=True)

train, test = model_selection.train_test_split(data["SpotPriceDKK"], train_size=n_training)

n_train = len(train)
n_test = len(test)
n_data = len(data)

plt.figure(figsize=(6, 4), dpi=100)
plt.plot(np.arange(1,n_train+1), train)
plt.plot(np.arange(n_train+1,n_data+1), test)
plt.legend(["Training set", "Testing set"])
plt.show()
plt.grid(alpha=0.25)
plt.ylabel("Average power (kW)")
plt.tight_layout()
#%% Fit the Pipe
import os
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import pmdarima as pm
from pmdarima import pipeline, arima, model_selection
from sklearn.metrics import root_mean_squared_error
import pandas as pd

#%%
X_train, X_test = model_selection.train_test_split(merged_df[["LocalPowerMWh","OnshoreWindGe50kW_MWh","ExchangeGreatBelt_MWh"]], train_size=n_training)

# Define X_train_ar as explained in the Hands on pdf
n = n_train
X_train_ar = np.column_stack([np.arange(1, n+1), X_train])

pipe = pipeline.Pipeline([
    ("fourier", pm.preprocessing.FourierFeaturizer(m=24, k = 12)),
    ("arima", arima.AutoARIMA(stepwise=False, trace=1, error_action="ignore",
                              seasonal=False, maxiter=1, 
                              suppress_warnings=True))])

pipe.fit(train, X = X_train_ar)
#%% Make the Prediction

rolling_forecast = []
Persistence24   = []

dataset = merged_df["SpotPriceDKK"]

N = int(len(test)/24)
rolling_forecast = [None] * (N * 24)
Persistence24 = [None] * (N * 24)
for i in range(N):

    # Create X_f for the forecasting period
    X_f = np.column_stack([np.arange(1, 24+1), 
                           X_test[i*24:(i+1)*24]])

    forecast = pipe.predict(n_periods=24, X = X_f)
    print(forecast)
    rolling_forecast.extend(forecast)

    pipe.update(test[i*24:(i+1)*24], X = X_f)
    Persistence24.extend(dataset[len(train)+i*24-24:len(train)+(i+1)*24-24])
    

rolling_forecast = [0 if x < 0 else x for x in rolling_forecast]
#%%

'''
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Truncate the test set if it's longer than the forecasts
test_trimmed = test[:len(rolling_forecast)]

# Calculate RMSE for the forecast
rmse_forecast = mean_squared_error(test_trimmed, rolling_forecast, squared=False)
mae_forecast = mean_absolute_error(test_trimmed, rolling_forecast)

# Calculate RMSE for the persistence model
rmse_persistence = mean_squared_error(test_trimmed, Persistence24, squared=False)
mae_persistence = mean_absolute_error(test_trimmed, Persistence24)

rmse_forecast, mae_forecast, rmse_persistence, mae_persistence

'''
#%%
plt.figure(figsize=(10, 6), dpi=100)  # Increase figure size
plt.plot(np.arange(1, len(train) + 1), train, color="black", linewidth=1)
plt.plot(np.arange(len(train) + 1, len(train) + len(rolling_forecast) + 1), rolling_forecast, color="blue", linestyle='--', marker='o', markersize=3)
plt.plot(np.arange(len(train) + 1, len(train) + len(Persistence24) + 1), Persistence24, color="green", linestyle=':', marker='x', markersize=3)
plt.plot(np.arange(len(train) + 1, len(train) + len(test) + 1), test, color="red", linewidth=2)
plt.legend(["Training set", "Forecasted values", "Persistence", "Actual values"], loc="upper left")
plt.grid(alpha=0.25)
plt.tight_layout()  # This should be before plt.show()


# Set a limit on the y-axis if needed
# plt.ylim([lower_limit, upper_limit])

plt.show()
