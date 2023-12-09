import streamlit as st
import requests
import pandas as pd
import io
import matplotlib.pyplot as plt
import plotly.express as px
keys = ['YR7C1B1NPZ97A348', 'JCQFVNBK3U8QMZ0T', 'P1Y9WYCX6JNW4N1M', 'HP94RXLQOYEWVCMZ']
MY_KEY = "HP94RXLQOYEWVCMZ"
st.title('Nintendo And Rockstar Stock Forecast')
stocks = ('NTDOY', 'TTWO')
selected_stock = st.selectbox('Select stock for prediction', stocks)


def fetch_stock_data(symbol, api_key, year_needed):

    DAILY_ENDPOINT = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={api_key}&datatype=csv"
    r = requests.get(DAILY_ENDPOINT).content
    df = pd.read_csv(io.StringIO(r.decode('utf-8')))
    df = pd.DataFrame(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[df["timestamp"].dt.year >= year_needed]
    if symbol == "NTDOY":
      # Identify the date of the share split
      share_split_date = pd.to_datetime('10/04/2022')

      # Filter data before and after the share split date
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
    fig = px.line(dataset, x='timestamp', y=['open', 'high', 'low', 'close'],
                  labels={'value': 'Price', 'variable': 'Type'},
                  title=f'{selected_stock} Stock Prices',
                  line_shape='linear')

    return fig


data_graph = plot_graph(choose_dataset)
st.write("Recent five days", choose_dataset.head())
st.plotly_chart(data_graph)
