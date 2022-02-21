import numpy as np
import pandas as pd
import hvplot.pandas
from tensorflow import random
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
from numpy.random import seed
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


close_feature = 1 #Using close to predict Close, index#1
close_target = 1 #Close, index#1
fng_feature = 0 #Using FNG to predict Close, index#0
fng_target = 1 #Close, index#1
window_size = 10 #Experiment with window sizes anywhere from 1 to 10 and see how the model performance changes

def window_data(df, window, feature_col_number, target_col_number):
    X = []
    y = []
    for i in range(len(df) - window - 1):
        features = df.iloc[i:(i + window), feature_col_number]
        target = df.iloc[(i + window), target_col_number]
        X.append(features)
        y.append(target)
    return np.array(X), np.array(y).reshape(-1, 1)
print('')
print(f'Window size: Model will predict close price based on previous {window_size} days')


def run_lstm(lstm_df, window_size, feature, target):
    X, y = window_data(lstm_df, window_size, feature, target)
    split = int(0.7 * len(X))
    X_train = X[: split]
    X_test = X[split:]
    y_train = y[: split]
    y_test = y[split:]
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    scaler.fit(y_train)
    y_train = scaler.transform(y_train)
    y_test = scaler.transform(y_test)
    # Reshape features for the model
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    print (f"X_train sample size:\n{X_train.shape} \n")
    print('')
    print (f"X_test sample size:\n{X_test.shape} \n")
    print(f'Important reminders for building the LSTM model:')
    print(f'')
    print(f'1. DROPOUTS help prevent overfitting, smaller batch size is recommended')
    print(f'2. INPUT_SHAPE is the number of time steps and the number of indicators')
    print(f'3. BATCHING has a different input shape of Samples/TimeSteps/Features')
    print(f'4. Additional LSTM layers: set Return Sequences to TRUE')
    print('5. Do not shuffle the data')
    print(f'6. Use at least 10 epochs')
    model = Sequential()
    number_units = 30
    dropout_fraction = 0.2
    model.add(LSTM(
        units=number_units,
        return_sequences=True,
        input_shape=(X_train.shape[1], 1))
        )
    model.add(Dropout(dropout_fraction))
    model.add(LSTM(units=number_units, return_sequences=True))
    model.add(Dropout(dropout_fraction))
    model.add(LSTM(units=number_units))
    model.add(Dropout(dropout_fraction))
    print(f'OUTPUT LAYER = model.add(Dense(1))')
    model.add(Dense(1))
    print('')
    print(f'Compiling model..')
    model.compile(optimizer="adam", loss="mean_squared_error")
    print('Summarizing the model..')
    model.summary()
    print('Training the model..') 
    model.fit(X_train, y_train, epochs=10, shuffle=False, batch_size=90, verbose=1)
    print('')
    model.evaluate(X_test, y_test, verbose=0)
    print('Predict Prices on Test data')
    predicted = model.predict(X_test)
    print('Recover the original prices instead of the scaled version')
    print('IMPORTANT: Use `inverse_transform` function to the predicted and y_test values to recover the actual closing prices.') 
    predicted_prices = scaler.inverse_transform(predicted)
    real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
    stocks = pd.DataFrame({
        "Real": real_prices.ravel(),
        "Predicted": predicted_prices.ravel()
    }, index = lstm_df.index[-len(real_prices): ]) 
    return px.line(stocks, title='BTCUSD Closing Price Prediction')