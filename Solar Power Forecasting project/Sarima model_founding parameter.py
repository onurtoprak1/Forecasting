import pandas as pd
from pmdarima import auto_arima
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX


df = pd.read_excel('Forecast Dataset.xlsx', nrows=2000)

#print(df.to_string())
#df['DATE'] = pd.to_datetime(df['DATE'])
#df.set_index('DATE', inplace=True)
df.set_index('TIME', inplace=True)
#print(df.head())

df.index.freq = 'H'

#df['POWER GENERATION (MW)'].plot(figsize=(20,5))
#plt.title('POWER GENERATION (MW)')
#plt.show()

#result = seasonal_decompose(df['POWER GENERATION (MW)'], model='add')
#result.plot();

print(auto_arima(df['POWER GENERATION (MW)'],seasonal=True,m=24).summary())

#start_from = 900
#df_train = df.iloc[:start_from]
#df_testing = df.iloc[start_from:]
#pprint(df_train.tail())
#pprint(len(df_train))

#my_model = SARIMAX(df['POWER GENERATION (MW)'],order=(0,0,1),seasonal_order=(1,0,1,24))
#results = my_model.fit()
#print(results.summary())

#starting = len(df)
#predictions = results.predict(start=starting, end=starting+(24*k), dynamic=False, type='level')
#print(predictions)

#for i in range(len(predictions)):
#    print(f"expected={df_testing['POWER GENERATION (MW)'][i]}")

#title='mm'
#ylabel='power'
#xlabel=''

#ax = df['POWER GENERATION (MW)'].plot(legend=True,title=title)
#predictions.plot(legend=True)

#plt.show()




# print(df['TIME'])

#plt.plot(df['POWER GENERATION (MW)'])
#plt.ylabel('POWER GENERATION (MW)')
#plt.title('Understanding data time variance')
#plt.show()

# # This is an implementation of phase 1-EE474 PROJECT
# 
# # def e2375897_Toprak_GF(path1 , path2):
# import pandas as pd
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# import matplotlib.pyplot as plt
# 
# def forecast_pv_generation(path1): # it was pv_data
#     """
#     Forecast power generation of a given PV array with a horizon of one day and one-hour resolution.
# 
#     Parameters:
#     - pv_data (pandas.DataFrame): DataFrame with a datetime index and a 'power' column representing historical power generation.
# 
#     Returns:
#     - forecasted_power (pandas.Series): Series containing the forecasted power generation.
#     """
# 
#     # Assuming 'power' is the column containing historical power generation data
#     # pv_series = pv_data['power']
#     # excel_file_path = 'Forecast Dataset.xlsx'
# 
#     df = pd.read_excel(path1)
#     c_name = 'POWER GENERATION (MW)'
#     power_column = df[c_name]
# 
# 
#     # Fit a SARIMA model
#     order = (1, 1, 1)  # (p, d, q)
#     seasonal_order = (1, 1, 1, 24)  # (P, D, Q, s)
# 
#     model = SARIMAX(power_column, order=order, seasonal_order=seasonal_order)
#     fitted_model = model.fit(disp=False)
# 
#     # Forecast one day ahead with hourly resolution
#     forecast_steps = 24
#     forecasted_power = fitted_model.get_forecast(steps=forecast_steps).predicted_mean
# 
#     return forecasted_power
# 
# # Example usage:
# # Assuming pv_data is a DataFrame with a datetime index and a 'power' column
# # Make sure to replace this with your actual data
# pv_data = pd.DataFrame(
#     {'power': [10, 12, 15, 18, 20, 22, 25, 30, 28, 24, 20, 18, 15, 12, 10, 8, 6, 5, 4, 3, 2, 1, 0, 0]})
# pv_data.index = pd.date_range(start='2023-01-01', periods=len(pv_data), freq='H')
# 
# forecasted_power = forecast_pv_generation('Forecast Dataset.xlsx')
# 
# # Plot the historical data and forecast
# plt.plot(pv_data.index, pv_data['power'], label='Historical Power')
# plt.plot(pd.date_range(start=pv_data.index[-1], periods=len(forecasted_power), freq='H'), forecasted_power,
#         label='Forecasted Power', linestyle='dashed')
# plt.legend()
# plt.xlabel('Time')
# plt.ylabel('Power Generation')
# plt.title('PV Power Generation Forecast')
# plt.show()
