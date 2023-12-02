import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Function to load data
def load_data(stock_name, start_date, end_date):
    try:
        data = yf.download(stock_name, start=start_date, end=end_date)
        if data.empty:
            raise ValueError("No data available for the given date range.")
        return data[['Close']]
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# Function to preprocess data
def preprocess_data(data):
    if isinstance(data, pd.DataFrame):
        if data.empty or 'Close' not in data.columns:
            raise ValueError("Invalid data format. The DataFrame must have a 'Close' column.")
        data = data['Close'].values.reshape(-1, 1)
    elif not isinstance(data, np.ndarray):
        raise ValueError("Invalid data format. Input must be a DataFrame or NumPy array.")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

# Function to create LSTM model
def create_lstm_model(time_step):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to create dataset
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# Function to train LSTM model
def train_lstm_model(train_data, time_step):
    x_train, y_train = create_dataset(train_data, time_step)
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

    model = create_lstm_model(time_step)
    model.fit(x_train, y_train, epochs=10, batch_size=32)  # You might need to adjust epochs and batch_size

    return model

# ... (previous code)

# Function to predict stock price
def predict_stock_price(stock_data, scaler, model, time_step):
    test_data = stock_data[-time_step:]
    
    if len(test_data.shape) == 1:  # Check if test_data is 1D
        test_data = test_data.reshape((1, -1, 1))
    else:
        test_data = test_data.reshape((test_data.shape[0], test_data.shape[1], 1))
    
    test_data = scaler.transform(test_data)
    
    prediction = model.predict(test_data)
    prediction = scaler.inverse_transform(prediction)
    
    return prediction[0][0]

# ... (rest of the code)

# Main Streamlit app
def main():
    st.title('Stock Price Prediction with LSTM')

    stock_name = st.text_input('Enter Stock Symbol (e.g., AAPL):', 'AAPL')
    start_date = st.text_input('Enter Start Date (YYYY-MM-DD):', '2020-01-01')
    end_date = st.text_input('Enter End Date (YYYY-MM-DD):', '2021-01-01')

    if st.button('Predict'):
        st.write(f"Predicting stock prices for {stock_name} from {start_date} to {end_date}")

        # Load Data
        stock_data = load_data(stock_name, start_date, end_date)['Close'].values.reshape(-1, 1)

        # Preprocess Data
        try:
            scaled_data, scaler = preprocess_data(stock_data)

            # Train-test split
            train_size = int(len(scaled_data) * 0.8)
            train_data, test_data = scaled_data[0:train_size], scaled_data[train_size:]

            # Train LSTM Model
            lstm_model = train_lstm_model(train_data, time_step=100)  # You might need to adjust time_step

            # Save the model weights
            lstm_model.save_weights('lstm_model_weights.h5')

            # Predict
            x_test, y_test = create_dataset(test_data, time_step=100)

            # Check if x_test has enough data points
            if len(x_test) >= 1:
                x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

                predictions = []

                # Iterate over x_test to make predictions
                for i in range(len(x_test)):
                    prediction = predict_stock_price(x_test[i], scaler, lstm_model, time_step=100)
                    predictions.append(prediction)

                predictions = np.array(predictions)

                # Calculate accuracy and error rates
                accuracy = np.mean(np.abs(predictions - y_test) / y_test) * 100
                error_rate = np.mean(np.abs(predictions - y_test))

                st.write(f"Mean Absolute Percentage Error (MAPE): {accuracy:.2f}%")
                st.write(f"Mean Absolute Error (MAE): ${error_rate:.2f}")

                # Plot the predictions
                st.line_chart(pd.DataFrame({'Actual': scaler.inverse_transform(y_test), 'Predicted': predictions.flatten()}))

            else:
                st.warning("Insufficient data for prediction. Please choose a different date range.")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()

