# Imports
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import numpy as np
import cvxpy as cp

"""
# Funktioner til dataimporten
def PricesDK(df_prices):
    
    # Set the Sell price equal to the spot price
    df_prices["Sell"] = df_prices["SpotPriceDKK"]
    
    # Define the fixed Tax and TSO columns
    df_prices["Tax"] = 0.8
    df_prices["TSO"] = .1
    
    
    ### Add the DSO tariffs ###


    #dso shold be devided by 100 and multiplied by 1000 to get the price in DKK/MWh

    # winter low
    df_prices.loc[(df_prices["HourDK"].dt.month.isin([10,11,12,1,2,3])) 
     & (df_prices["HourDK"].dt.hour.isin([0,1,2,3,4,5,6])), "DSO"] = 15*10


    # winter high
    df_prices.loc[(df_prices["HourDK"].dt.month.isin([10,11,12,1,2,3])) 
     & (df_prices["HourDK"].dt.hour.isin([7,8,9,10,11,12,13,14,15,16,17,22,23])), "DSO"] = 45*10
    
    # winter peak
    df_prices.loc[(df_prices["HourDK"].dt.month.isin([10,11,12,1,2,3])) 
     & (df_prices["HourDK"].dt.hour.isin([18,19,20,21])), "DSO"] = 135*10
    

    #summer low
    df_prices.loc[(df_prices["HourDK"].dt.month.isin([4,5,6,7,8,9])) 
     & (df_prices["HourDK"].dt.hour.isin([0,1,2,3,4,5,6])), "DSO"] = 15*10
    
    # summer high
    df_prices.loc[(df_prices["HourDK"].dt.month.isin([4,5,6,7,8,9])) 
     & (df_prices["HourDK"].dt.hour.isin([7,8,9,10,11,12,13,14,15,16,17,22,23])), "DSO"] = 23*10

    # summer peak
    df_prices.loc[(df_prices["HourDK"].dt.month.isin([4,5,6,7,8,9])) 
     & (df_prices["HourDK"].dt.hour.isin([18,19,20,21])), "DSO"] = 60*10

    # Calculate VAT
    df_prices["VAT"] = 0.25*(df_prices["SpotPriceDKK"]+df_prices["TSO"]+df_prices["DSO"]+df_prices["Tax"])
    
    # Calculate Buy price
    df_prices["Buy"] = df_prices["VAT"]+df_prices["SpotPriceDKK"]+df_prices["TSO"]+df_prices["DSO"]+df_prices["Tax"]
    
    return df_prices

def LoadData():
    
    import os
    import pandas as pd
    
    ### Load electricity prices ###
    price_path = os.path.join(os.getcwd(),'Elspotprices2.csv')
    df_prices = pd.read_csv(price_path)
    
    # Convert to datetime
    df_prices["HourDK"] = pd.to_datetime(df_prices["HourDK"])
    
    
    # Convert prices to DKK/mwh - tjæk om det er rigtigt
    df_prices['SpotPriceDKK'] = df_prices['SpotPriceDKK']/1000
    
    # Filter only DK2 prices
    df_prices = df_prices.loc[df_prices['PriceArea']=="DK2"]
    
    # Keep only the local time and price columns
    df_prices = df_prices[['HourDK','SpotPriceDKK',"HourUTC"]]
    
    # Keep only 2022 and 2023
    #df_prices = df_prices.loc[df_prices["HourDK"].dt.year.isin([2018,2019,2020,2021,2022,2023])]
    
    # Reset the index
    df_prices = df_prices.reset_index(drop=True)
    
    ###  Load prosumer data ###
    file_P = os.path.join(os.getcwd(),'ProdConData.csv')
    df_pro = pd.read_csv(file_P)
    df_pro["HourDK"] = pd.to_datetime(df_pro["HourDK"])
    df_pro = df_pro.reset_index(drop=True)
    df_pro.rename(columns={'Consumption': 'Load'}, inplace=True)
    df_pro.rename(columns={'HourDK': 'HourDK'}, inplace=True)

    return df_prices, df_pro

df_prices, df_pro = LoadData()
df_prices = PricesDK(df_prices)

# Define the start and end time to filter your price data for 2019
t_s_19 = pd.Timestamp(dt.datetime(2019,1,1,0,0,0))
t_e_19 = pd.Timestamp(dt.datetime(2019,12,31,23,0,0))

# Filter df_prices to get the desired values and use .values to obtain the actual values of the prices
p19 = df_prices.loc[(df_prices["HourDK"]>=t_s_19) & (df_prices["HourDK"]<=t_e_19),"SpotPriceDKK"].values
"""

# Predicted values for the test month

p_vector = np.load('/Users/madslangkjaerjakobsen/Desktop/ene/LSTM_EX_pre.npy')
plt.plot(p_vector) 

# Parameters for optimization of the battery
params = {
    'Pmax': 1, # 1 MW
    'Cmax': 2, # 2 MWh
    'Cmin': 0.1,  # 10% og Cmax
    'n_c': 0.95,  # 95% charging efficiency
    'n_d': 0.95,  # 95% discharging efficiency
    'C_0': 0.5,   # 50% Cmax
    'C_n': 0.5,   # 50% Cmax
}

def Optimizer(params, p):   
    
    n = len(p)
    
    # Define the decision variables
    p_c = cp.Variable(n)
    p_d = cp.Variable(n)
    X   = cp.Variable(n)
    
    # Define the cost function
    profit = cp.sum(p_d@p - p_c@p)
    
    # Add constraints
    constraints = [p_c >= 0, p_d >= 0, p_c <= params['Pmax'], p_d <= params['Pmax']]
    constraints += [X >= params['Cmin'], X <= params['Cmax']]
    constraints += [X[0]==params['C_0'] + p_c[0]*params['n_c'] - p_d[0]/params['n_d']]
    constraints += [X[1:] == X[:-1] + p_c[1:]*params['n_c'] - p_d[1:]/params['n_d']]
    constraints += [X[n-1]>=params['C_n']]
    
    # Solve the problem
    problem = cp.Problem(cp.Maximize(profit), constraints)
    problem.solve(solver=cp.ECOS)

    return profit.value, p_c.value, p_d.value, X.value

# create a dataframe to store the results
Bat = pd.DataFrame(columns = ['p_c', 'p_d', 'SOC'])

# Optimize the battery operation for each day of the year
profit, Bat['p_c'], Bat['p_d'], Bat['SOC'] = Optimizer(params, p_vector)

# Print the aggregated profits for all 5 years
print(f"The predicted aggregated profit is {round(profit,1)} DKK")

plt.plot(p_vector)
plt.xlabel('Hour')
plt.ylabel('Price [DKK/MWh]')
plt.title('Predicted Spotprices DK2')

# The strategy of optimized for the test months used on other data
actual_prices = np.load('/Users/madslangkjaerjakobsen/Desktop/ene/LSTM_EX_data.npy')

# Calculate the benefit
benefit = (Bat['p_c'] - Bat['p_d']) * actual_prices

# Sum up the benefit over all time steps
total_benefit = benefit.sum()
print(f"The actual aggregated profit is {round(total_benefit,1)} DKK")
print(f"This is an overestimate of {round(abs((total_benefit-profit)/total_benefit*100),1)}%")
