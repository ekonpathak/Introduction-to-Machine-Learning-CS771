import numpy as np
import pickle as pkl
import pandas as pd
# Define your prediction method here
# df is a dataframe containing timestamps, weather data and potentials
def my_predict( df ):
    df['Time'] = pd.to_datetime(df['Time'])
    df['day']=df['Time'].dt.day
    df['month']=df['Time'].dt.month
    df['hour']=df['Time'].dt.hour
    df['minute']=df['Time'].dt.minute
    X=df[['temp', 'humidity', 'no2op1', 'no2op2', 'o3op1',
       'o3op2','day', 'month', 'hour', 'minute']].to_numpy()
    with open( "NO2_model.sav", "rb" ) as file:
        modelNO2 = pkl.load( file )
    with open( "OZONE_model.sav", "rb" ) as file:
        modelOZONE = pkl.load( file )
    pred1 = modelNO2.predict(X)
    pred2 = modelOZONE.predict(X)

    return ( pred2, pred1 )