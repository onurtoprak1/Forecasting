# # This is an implementation of phase 1-EE474 PROJECT

import pandas as pd
from pmdarima import auto_arima
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np

# INPUTS
# pathf: Please write the path of Forecast Dataset.xlsx
# k: Choose the which day after data (1 is next day, 2 is 2nd day...)
# clouds: Enter the Cloudness (1x24) percentages over over % (like 78)

#OUTPUT
# new_pred: Predicted power generation of a day (1x24)

def e2375897_Toprak_GF(pathf , k, clouds):

    cloudness = np.array(clouds)
    realcloud = cloudness * 0.01
    usageofcloud = 1 - realcloud
    
    df = pd.read_excel(pathf)
    
    df.set_index('TIME', inplace=True)
    df.index.freq = 'H'
    
    #print(auto_arima(df['POWER GENERATION (MW)'],seasonal=True,m=24).summary()) # to obtain the order parameters of SARIMAX model
    
    my_model = SARIMAX(df['POWER GENERATION (MW)'],order=(0,0,1),seasonal_order=(1,0,1,24))
    results = my_model.fit()
    
    starting = len(df)
    predictions = results.predict(start=starting+((k-1)*24), end=starting+(24*k)-1)
    predictions[predictions < 0] = 0
    new_pred = (np.array(usageofcloud))*(np.array(predictions))

    return new_pred


a= e2375897_Toprak_GF('Forecast Dataset.xlsx' , 1, [0,0,0,0,0,0,0,0,0,0,0,0,0,70,70,80,90,0,0,0,1,0,0,0])
print(a)

hour = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]

plt.title('mm')
plt.ylabel('power')
plt.xlabel('hour')
plt.legend()
plt.plot(hour,a)

plt.show()
