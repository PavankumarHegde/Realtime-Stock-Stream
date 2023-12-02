import streamlit as st
import yfinance as yf
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Function to download stock data using yfinance
def download_stock_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data

# Function to preprocess data for LSTM
def preprocess_data_for_lstm(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size), 0])
        y.append(data[i + window_size, 0])
    return np.array(X), np.array(y)

# Function to create and train LSTM model
def create_lstm_model(window_size):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(window_size, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# Function to make predictions using the models
def make_predictions(catboost_model, xgboost_model, lstm_model, data, window_size):
    # Assuming 'Close' is the target variable
    features = data.drop('Close', axis=1)
    
    # CatBoost prediction
    predictions_catboost = catboost_model.predict(features)

    # XGBoost prediction
    predictions_xgboost = xgboost_model.predict(features.values)

    # LSTM prediction
    lstm_data = data['Close'].values.reshape(-1, 1)
    lstm_data = lstm_data / np.max(lstm_data)
    X_lstm, _ = preprocess_data_for_lstm(lstm_data, window_size)
    X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))
    predictions_lstm = lstm_model.predict(X_lstm)

    return predictions_catboost, predictions_xgboost, predictions_lstm

# Streamlit app
def main():
    st.title("Stock Price Prediction App")

    # Sidebar
    st.sidebar.header("Settings")

    # Convert the tuple to a datetime object
    start_date_default = datetime(2021, 1, 1)
    end_date_default = datetime(2023, 1, 1)

    symbol = st.sidebar.text_input("Enter stock symbol:", value="AAPL")
    start_date = st.sidebar.date_input("Start date", value=start_date_default)
    end_date = st.sidebar.date_input("End date", value=end_date_default)
    window_size = st.sidebar.slider("LSTM Window Size", min_value=1, max_value=30, value=10)

    # Download stock data
    data = download_stock_data(symbol, start_date, end_date)

    # CatBoost model
    catboost_model = CatBoostRegressor()
    features = data.drop('Close', axis=1)
    target = data['Close']
    catboost_model.fit(features, target)

    # XGBoost model
    xgboost_model = XGBRegressor()
    xgboost_model.fit(features.values, target)

    # LSTM model
    lstm_data = data['Close'].values.reshape(-1, 1)
    lstm_data = lstm_data / np.max(lstm_data)
    X_lstm, y_lstm = preprocess_data_for_lstm(lstm_data, window_size)
    lstm_model = create_lstm_model(window_size)
    lstm_model.fit(X_lstm, y_lstm, epochs=50, batch_size=32)

    # Make predictions
    predictions_catboost, predictions_xgboost, predictions_lstm = make_predictions(
        catboost_model, xgboost_model, lstm_model, data, window_size
    )

    # Combine predictions
    predictions_combined = (predictions_catboost + predictions_xgboost + predictions_lstm) / 3.0

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

    # Actual Prices
    actual_prices = data['Close'][window_size:]
    ax1.plot(data.index[window_size:], actual_prices, label='Actual Price', linewidth=2)

    # Ensure predictions have the same length as actual prices
    num_predictions_catboost = min(len(actual_prices), len(predictions_catboost))
    num_predictions_xgboost = min(len(actual_prices), len(predictions_xgboost))
    num_predictions_combined = min(len(actual_prices), len(predictions_combined))

    ax1.plot(data.index[window_size:window_size + num_predictions_catboost], predictions_catboost[:num_predictions_catboost], label='CatBoost Predictions', linestyle='--')
    ax1.plot(data.index[window_size:window_size + num_predictions_xgboost], predictions_xgboost[:num_predictions_xgboost], label='XGBoost Predictions', linestyle='--')
    #ax1.plot(data.index[window_size:window_size + num_predictions_combined], predictions_combined[:num_predictions_combined], label='Combined Predictions', linestyle='--')

    # Legends and labels for the first plot
    ax1.legend()
    ax1.set_title(f'Stock Price Prediction for {symbol}')
    ax1.set_ylabel('Stock Price')

    # LSTM Predictions
    ax2.plot(data.index[window_size:], actual_prices, label='Actual Price', linewidth=2)
    num_predictions_lstm = min(len(actual_prices), len(predictions_lstm))
    ax2.plot(data.index[window_size:window_size + num_predictions_lstm], predictions_lstm[:num_predictions_lstm], label='LSTM Predictions', linestyle='--')

    # Legends and labels for the second plot
    ax2.legend()
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Stock Price')

    plt.tight_layout()
    st.pyplot()

    # Predict next price
    next_price_catboost = catboost_model.predict(features.iloc[[-1]])[0]
    next_price_xgboost = xgboost_model.predict(features.iloc[[-1]])[0]
    next_price_lstm = lstm_model.predict(lstm_data[-window_size:].reshape(1, window_size, 1))[0][0]
    next_price_combined = (next_price_catboost + next_price_xgboost + next_price_lstm) / 3.0

    st.subheader("Next Price Predictions:")
    st.write(f"CatBoost: {next_price_catboost}")
    st.write(f"XGBoost: {next_price_xgboost}")
    st.write(f"LSTM: {next_price_lstm}")
    st.write(f"Combined: {next_price_combined}")

if __name__ == '__main__':
    main()

