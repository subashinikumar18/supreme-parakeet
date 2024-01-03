def sarima_method(_arg1,_arg2,_arg3):
    import pandas as pd
    from pandas import DataFrame
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import statsmodels.api as sm
    import dateutil
    import datetime
    import warnings
    warnings.filterwarnings('ignore')
    
    data = DataFrame({'dt': _arg1,'Sales': _arg2})
    data = data.sort_values(by = 'dt')
    data['dt'] = pd.to_datetime(data['dt'])
    
    
# train entire dataset
    data.index = pd.to_datetime(data['dt']) 
    data = data.resample('M').mean()
    data = data[:-(_arg3)]
    
    # future dataset
    step = dateutil.relativedelta.relativedelta(months=1)
    start = data.index[len(data)-1] + step
    index = pd.date_range(start, periods=_arg3, freq='M')
    columns = ['Sales']
    df = pd.DataFrame(index=index, columns=columns)
    df = df.fillna(0)
   
    # model fit
   
    fit1 = sm.tsa.statespace.SARIMAX(data['Sales'], order=(1, 1, 1),seasonal_order=(1,1,1,12)).fit()
    df['Sales']=fit1.forecast(_arg3)
    df = df.fillna(0)
    x = pd.concat([data, df])
    return x['Sales'].tolist()


import tabpy_client
connection = tabpy_client.client('http://localhost:9004/')

connection.deploy('SARIMA Prediction',sarima_method,'forecast values', override=True)
