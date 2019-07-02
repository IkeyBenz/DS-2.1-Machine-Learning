#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from ta.momentum import money_flow_index
from ta.momentum import rsi  # Relative Strength Index
from ta.momentum import stoch  # Stochastic Oscilator
from ta.momentum import uo  # Ultimate Oscilator
from ta.momentum import wr  #William Percent Range
from ta.trend import macd  # Moving Average Convergence/Divergence 



# *************************** DATA PREPROCESSING ******************************

# Load Data
df = pd.read_csv('^GSPC.csv')


# Data Augmentation
df['Relative_Strength_Index'] = rsi(df['Close'])
df['Money_Flow_Index'] = money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'])
df['Stoch_Oscilator'] = stoch(df['High'], df['Low'], df['Close'])
df['Ultimate_Oscilator'] = uo(df['High'], df['Low'], df['Close'])
df['William_Percent'] = wr(df['High'], df['Low'], df['Close'])
df['MACD'] = macd(df['Close'])


# Some indicators require many days in advance before they produce any
# values. So the begining rows of our df may have NaNs. Lets drop them:
df = df.dropna()

# Scaling Data
from sklearn.preprocessing import MinMaxScalern
sc = MinMaxScaler(feature_range=(0, 1))
scaled_df = sc.fit_transform(df.iloc[:, 1:].values)


# Creating a data structure with 60 timesteps and 1 output
X_train = np.array([scaled_df[i:i+60, :] for i in range(len(scaled_df)-60)])
y_train = np.array([scaled_df[i+60, 0] for i in range(len(scaled_df)-60)])


# *************************** BUILDING RNN ************************************
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential([
        
    LSTM(  # Add First LSTM Layer
        units=50,
        return_sequences=True,  # Allow access to hidden state for next layers
        input_shape=(X_train.shape[1], X_train.shape[2])
    ),
    Dropout(0.2),  # Prevent Overfitting
    
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    
    LSTM(units=50),  # Don't return sequences becasue this is last LSTM layer
    Dropout(0.2),
    
    Dense(units=1)  # Output layer
    
])
    
regressor.compile(optimizer='adam', loss='mean_squared_error')

regressor.fit(X_train, y_train, epochs=100, batch_size=32)