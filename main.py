#Import libraries & Dependencies
from _functions import *
import numpy as np
import pandas as pd
import hvplot.pandas
from tensorflow import random
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
from numpy.random import seed
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
seed(1)
random.set_seed(2)
print('Import libraries, set random seed for reproducibility')
print('')
print('CSV Import: Fear and greed sentiment data for Bitcoin, variable name: sentiment_df')
sentiment_df = pd.read_csv('Data/btc_sentiment.csv', index_col="date", infer_datetime_format=True, parse_dates=True)
sentiment_df = sentiment_df.drop(columns="fng_classification")
print('CSV Import: Historical closing prices for Bitcoin, variable name: prices_df')
prices_df = pd.read_csv('Data/btc_historic.csv', index_col="Date", infer_datetime_format=True, parse_dates=True)['Close']
prices_df = prices_df.sort_index()
print('')
print('Fear and greed sentiment data & Historical closing prices (First two records):')
lstm_df = sentiment_df.join(prices_df, how="inner")
print(lstm_df.head(2))

run_lstm(lstm_df, window_size, close_feature, close_target)

run_lstm(lstm_df, window_size, fng_feature, fng_target)

print('Historical closing prices are a better predictor for the LSTM when compared to the F&G Index.')