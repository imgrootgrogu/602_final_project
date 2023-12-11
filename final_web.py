import streamlit as st
import requests
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import plotly.express as px
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import *


from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_percentage_error


from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

keys = ['YR7C1B1NPZ97A348', 'JCQFVNBK3U8QMZ0T', 'P1Y9WYCX6JNW4N1M', 'HP94RXLQOYEWVCMZ']
MY_KEY = "YR7C1B1NPZ97A348"
st.title('Nintendo And Rockstar Stock Forecast')
stocks = ('NTDOY', 'TTWO')
selected_stock = st.selectbox('Select stock for prediction', stocks)
st.write("Dear Graders: our API has a 25 inquiry limit per day, our app will crash in tears after the limit runs out. Please be gentle.")

def fetch_stock_data(symbol, api_key, year_needed):

    DAILY_ENDPOINT = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={api_key}&datatype=csv"
    r = requests.get(DAILY_ENDPOINT).content
    df = pd.read_csv(io.StringIO(r.decode('utf-8')))
    df = pd.DataFrame(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[df["timestamp"].dt.year>=year_needed]
    if symbol == "NTDOY":

      share_split_date = pd.to_datetime('10/04/2022')

      before_split = df[df['timestamp'] < share_split_date]
      after_split = df[df['timestamp'] >= share_split_date]

      # Divide the prices before the share split date by 5
      before_split[['open', 'high', 'low', 'close']] = before_split[['open', 'high', 'low', 'close']] / 5

      # Concatenate the adjusted data
      adjusted_df = pd.concat([after_split, before_split])

      return adjusted_df
    else:

      return df

def call_data(dataset):
    if dataset == "NTDOY":
        ntdoy_df = fetch_stock_data("NTDOY", MY_KEY, 2007)
        return ntdoy_df
    else:
        ttwo_df = fetch_stock_data("TTWO", MY_KEY, 2007)
        return ttwo_df
choose_dataset = call_data(selected_stock)

def plot_graph(dataset):
    fig = px.line(dataset, x=dataset.index, y=['open', 'high', 'low', 'close'],
                  labels={'value': 'Price', 'variable': 'Type'},
                  title=f'{selected_stock} Stock Prices',
                  line_shape='linear')

    return fig


data_graph = plot_graph(choose_dataset)
st.write("Recent five days", choose_dataset.head())
st.plotly_chart(data_graph)


def data_scaling(data):
  data.set_index('timestamp', inplace=True)
  target = data['close']

  training_features = ['open', 'high', 'low', 'volume']
  scaler = StandardScaler()
  feature_transform = scaler.fit_transform(data[training_features])
  feature_transform = pd.DataFrame(columns=training_features,
                      data=feature_transform,
                      index=data.index)
  target_transform = scaler.fit_transform(target.values.reshape(-1, 1))

  return feature_transform, target_transform, scaler
feature_transform, target_transform, scaler = data_scaling(choose_dataset)
def data_split(feature_transform, target_transform):
  timesplit= TimeSeriesSplit(n_splits=5)
  for train_index, test_index in timesplit.split(feature_transform):
    X_train, X_test = feature_transform[:len(train_index)], feature_transform[len(train_index): (len(train_index)+len(test_index))]
    y_train, y_test = target_transform[:len(train_index)], target_transform[len(train_index): (len(train_index)+len(test_index))]
  trainX =np.array(X_train)

  testX =np.array(X_test)
  X_train = trainX.reshape(X_train.shape[0], 1, X_train.shape[1])
  X_test = testX.reshape(X_test.shape[0], 1, X_test.shape[1])
  return trainX, X_train, X_test, y_train, y_test
trainX, X_train, X_test, y_train, y_test =  data_split(feature_transform, target_transform)
lstm = Sequential()
lstm.add(LSTM(50, input_shape=(1, trainX.shape[1]), activation='relu', return_sequences=False))
lstm.add(Dense(1))
lstm.compile(loss='mean_squared_error', optimizer='adam')
st.header("Please wait in patient for training process. :D ")
epochs = 10
my_bar = st.progress(0)
for epoch in range(epochs):
        history = lstm.fit(X_train, y_train, epochs=1, batch_size=6, verbose=0, shuffle=False)
        my_bar.progress((epoch+1)/epochs, text=f'epoch {epoch}')
my_bar.empty()
y_pred = lstm.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mape = mean_absolute_percentage_error(y_test, y_pred)


st.subheader(f"RMSE:{rmse}")
st.subheader(f"MAPE:{mape}")

def plot_lstm(y_test, y_pred, title='True vs Predicted Values'):
    y_test_flat = y_test.flatten()
    y_pred_flat = y_pred.flatten()
    df = pd.DataFrame({'True value': y_test_flat, 'LSTM prediction': y_pred_flat})

    fig = px.line(df, x=range(len(y_test_flat)), y=['True value', 'LSTM prediction'], labels={'y': 'Scaled USD', 'x': 'Time Scale'})
    fig.update_layout(title=title)
    return fig

lstm_graph = plot_lstm(y_test, y_pred)
st.plotly_chart(lstm_graph)

training_days = st.slider(label="Choose how many days of data for training", min_value=10, max_value=100)
forecast_days = st.slider(label="Choose how many days of data for forecasting", min_value=3, max_value=10)


def forecast(n_past, n_days):
  prediction_next_days = []
  my_bar = st.progress(0)
  for i in range(n_days):
    last_sequence = X_train[i: n_past+i, :]
    next_day_pred = lstm.predict(last_sequence)
    predicted_price = scaler.inverse_transform(next_day_pred)[0, 0]
    prediction_next_days.append(predicted_price)
    my_bar.progress((i + 1) / n_days, text='Data Process....')
  return prediction_next_days, my_bar

predictions, my_bar = forecast(training_days, forecast_days)
my_bar.empty()
def forecast_plot(stock_price, n_days):
  df = pd.DataFrame({'Day': list(range(1, n_days+1)), 'Stock Price': stock_price})
  fig = px.line(df, x='Day', y='Stock Price ($)', title='Stock Prices for Next 10 Work Days')
  fig.update_layout(xaxis_title='Day', yaxis_title='Stock Price')
  return fig

pred_plot = forecast_plot(predictions, forecast_days)
st.plotly_chart(pred_plot)