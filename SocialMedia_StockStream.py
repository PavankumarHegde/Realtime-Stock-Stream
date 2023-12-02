from stocknews import StockNews
import stocknews
from ntscraper import Nitter
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense



from matplotlib.pyplot import axis
import streamlit as st  # streamlit library
import pandas as pd  # pandas library
import yfinance as yf  # yfinance library
import datetime  # datetime library
from datetime import date
from plotly import graph_objs as go  # plotly library
from plotly.subplots import make_subplots
from prophet import Prophet  # prophet library
# plotly library for prophet model plotting
from prophet.plot import plot_plotly
import time  # time library
from streamlit_option_menu import option_menu  # select_options library

import numpy as np
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import plotly.express as px

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt



import plotly.graph_objects as go
from plotly.subplots import make_subplots

import webbrowser

st.set_page_config(layout='wide', initial_sidebar_state='expanded')

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Function to download stock data using yfinance
def download_stock_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data

def load_data(stock_name, start_date, end_date):
    data = yf.download(stock_name, start=start_date, end=end_date)
    return data

def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    return scaled_data, scaler

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

def create_lstm_model(time_step):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
    
def train_lstm_model(train_data, time_step):
    x_train, y_train = create_dataset(train_data, time_step)
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

    model = create_lstm_model(time_step)
    model.fit(x_train, y_train, epochs=10, batch_size=32)  # You might need to adjust epochs and batch_size

    return model

def predict_stock_price(stock_data, scaler, model, time_step):
    test_data = stock_data[-time_step:].reshape(1,-1)
    test_data = scaler.transform(test_data)
    test_data = test_data.reshape(1, time_step, 1)
    
    prediction = model.predict(test_data)
    prediction = scaler.inverse_transform(prediction)
    return prediction[0][0]

def make_catboost_predictions(catboost_model, features):
    return catboost_model.predict(features)
    
    
def calculate_error_metrics(predictions, actual_prices):
    accuracy = np.mean(np.abs(predictions - actual_prices[1:]) / actual_prices[1:]) * 100
    error_rate = np.mean(np.abs(predictions - actual_prices[1:]))
    return accuracy, error_rate
    
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

def lstm_stock_prediction(train_data, test_data, validation_data, validation_percentage):
    # Normalize the data
    scaler = MinMaxScaler()

    train_data_scaled = scaler.fit_transform(train_data.values.reshape(-1, 1))
    test_data_scaled = scaler.transform(test_data.values.reshape(-1, 1))

    # Create sequences for LSTM
    window_size = 10
    X_train, y_train, X_test, y_test = [], [], [], []

    for i in range(len(train_data_scaled) - window_size):
        X_train.append(train_data_scaled[i:i + window_size])
        y_train.append(train_data_scaled[i + window_size])

    for i in range(len(test_data_scaled) - window_size):
        X_test.append(test_data_scaled[i:i + window_size])
        y_test.append(test_data_scaled[i + window_size])

    # Split the data for validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_percentage, random_state=42)

    X_train, y_train, X_test, y_test, X_val, y_val = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), np.array(X_val), np.array(y_val)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

    # Build and train the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=1, epochs=50)  # Increase the number of epochs

    lstm_predictions_scaled = model.predict(X_test)
    lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled)

    return lstm_predictions

# Sidebar Section Starts Here
today = date.today()  # today's date
st.write('''# StockStream ''')  # title
#st.sidebar.image("Images/StockStreamLogo1.png", width=250,
                 #use_column_width=False)  # logo
st.sidebar.write('''# StockStream ''')

with st.sidebar: 
        selected = option_menu("Utilities", ["Stocks Performance Comparison", "Real-Time Stock Price", "Stock Prediction", 'Heatmap'])

start = st.sidebar.date_input(
    'Start', datetime.date(2022, 1, 1))  # start date input
end = st.sidebar.date_input('End', datetime.date.today())  # end date input
# Sidebar Section Ends Here

# read csv file
stock_df = pd.read_csv("StockStreamTickersData.csv")

# Stock Performance Comparison Section Starts Here
if(selected == 'Stocks Performance Comparison'):  # if user selects 'Stocks Performance Comparison'
    st.subheader("Stocks Performance Comparison")
    tickers = stock_df["Company Name"]
    # dropdown for selecting assets
    dropdown = st.multiselect('Pick your assets', tickers)

    with st.spinner('Loading...'):  # spinner while loading
        time.sleep(2)
        # st.success('Loaded')

    dict_csv = pd.read_csv('StockStreamTickersData.csv', header=None, index_col=0).to_dict()[1]  # read csv file
    symb_list = []  # list for storing symbols
    for i in dropdown:  # for each asset selected
        val = dict_csv.get(i)  # get symbol from csv file
        symb_list.append(val)  # append symbol to list

    def relativeret(df):  # function for calculating relative return
        rel = df.pct_change()  # calculate relative return
        cumret = (1+rel).cumprod() - 1  # calculate cumulative return
        cumret = cumret.fillna(0)  # fill NaN values with 0
        return cumret  # return cumulative return

    if len(dropdown) > 0:  # if user selects atleast one asset
        df = relativeret(yf.download(symb_list, start, end))[
            'Adj Close']  # download data from yfinance
        # download data from yfinance
        raw_df = relativeret(yf.download(symb_list, start, end))
        raw_df.reset_index(inplace=True)  # reset index

        closingPrice = yf.download(symb_list, start, end)[
            'Adj Close']  # download data from yfinance
        volume = yf.download(symb_list, start, end)['Volume']
        
        st.subheader('Raw Data {}'.format(dropdown))
        st.write(raw_df)  # display raw data
        chart = ('Line Chart', 'Area Chart', 'Bar Chart')  # chart types
        # dropdown for selecting chart type
        dropdown1 = st.selectbox('Pick your chart', chart)
        with st.spinner('Loading...'):  # spinner while loading
            time.sleep(2)

        st.subheader('Relative Returns {}'.format(dropdown))
                
        if (dropdown1) == 'Line Chart':  # if user selects 'Line Chart'
            st.line_chart(df)  # display line chart
            # display closing price of selected assets
            st.write("### Closing Price of {}".format(dropdown))
            st.line_chart(closingPrice)  # display line chart

            # display volume of selected assets
            st.write("### Volume of {}".format(dropdown))
            st.line_chart(volume)  # display line chart

        elif (dropdown1) == 'Area Chart':  # if user selects 'Area Chart'
            st.area_chart(df)  # display area chart
            # display closing price of selected assets
            st.write("### Closing Price of {}".format(dropdown))
            st.area_chart(closingPrice)  # display area chart

            # display volume of selected assets
            st.write("### Volume of {}".format(dropdown))
            st.area_chart(volume)  # display area chart

        elif (dropdown1) == 'Bar Chart':  # if user selects 'Bar Chart'
            st.bar_chart(df)  # display bar chart
            # display closing price of selected assets
            st.write("### Closing Price of {}".format(dropdown))
            st.bar_chart(closingPrice)  # display bar chart

            # display volume of selected assets
            st.write("### Volume of {}".format(dropdown))
            st.bar_chart(volume)  # display bar chart

        else:
            st.line_chart(df, width=1000, height=800,
                          use_container_width=False)  # display line chart
            # display closing price of selected assets
            st.write("### Closing Price of {}".format(dropdown))
            st.line_chart(closingPrice)  # display line chart

            # display volume of selected assets
            st.write("### Volume of {}".format(dropdown))
            st.line_chart(volume)  # display line chart

    else:  # if user doesn't select any asset
        st.write('Please select atleast one asset')  # display message
# Stock Performance Comparison Section Ends Here
    
# Real-Time Stock Price Section Starts Here
elif(selected == 'Real-Time Stock Price'):  # if user selects 'Real-Time Stock Price'
    st.subheader("Real-Time Stock Price")
    tickers = stock_df["Company Name"]  # get company names from csv file
    # dropdown for selecting company
    a = st.selectbox('Pick a Company', tickers)

    with st.spinner('Loading...'):  # spinner while loading
            time.sleep(2)

    dict_csv = pd.read_csv('StockStreamTickersData.csv', header=None, index_col=0).to_dict()[1]  # read csv file
    symb_list = []  # list for storing symbols

    val = dict_csv.get(a)  # get symbol from csv file
    symb_list.append(val)  # append symbol to list

    if "button_clicked" not in st.session_state:  # if button is not clicked
        st.session_state.button_clicked = False  # set button clicked to false

    def callback():  # function for updating data
        # if button is clicked
        st.session_state.button_clicked = True  # set button clicked to true
    if (
        st.button("Search", on_click=callback)  # button for searching data
        or st.session_state.button_clicked  # if button is clicked
    ):
        if(a == ""):  # if user doesn't select any company
            st.write("Click Search to Search for a Company")
            with st.spinner('Loading...'):  # spinner while loading
             time.sleep(2)
        else:  # if user selects a company
            # download data from yfinance
            data = yf.download(symb_list, start=start, end=end)
            data.reset_index(inplace=True)  # reset index
            st.subheader('Raw Data of {}'.format(a))  # display raw data
            st.write(data)  # display data

            def plot_raw_data():  # function for plotting raw data
                fig = go.Figure()  # create figure
                fig.add_trace(go.Scatter(  # add scatter plot
                    x=data['Date'], y=data['Open'], name="stock_open"))  # x-axis: date, y-axis: open
                fig.add_trace(go.Scatter(  # add scatter plot
                    x=data['Date'], y=data['Close'], name="stock_close"))  # x-axis: date, y-axis: close
                fig.layout.update(  # update layout
                    title_text='Line Chart of {}'.format(a) , xaxis_rangeslider_visible=True)  # title, x-axis: rangeslider
                st.plotly_chart(fig)  # display plotly chart

            def plot_candle_data():  # function for plotting candle data
                fig = go.Figure()  # create figure
                fig.add_trace(go.Candlestick(x=data['Date'],  # add candlestick plot
                                             # x-axis: date, open
                                             open=data['Open'],
                                             high=data['High'],  # y-axis: high
                                             low=data['Low'],  # y-axis: low
                                             close=data['Close'], name='market data'))  # y-axis: close
                fig.update_layout(  # update layout
                    title='Candlestick Chart of {}'.format(a),  # title
                    yaxis_title='Stock Price',  # y-axis: title
                    xaxis_title='Date')  # x-axis: title
                st.plotly_chart(fig)  # display plotly chart

            chart = ('Candle Stick', 'Line Chart')  # chart types
            # dropdown for selecting chart type
            dropdown1 = st.selectbox('Pick your chart', chart)
            with st.spinner('Loading...'):  # spinner while loading
             time.sleep(2)
            if (dropdown1) == 'Candle Stick':  # if user selects 'Candle Stick'
                plot_candle_data()  # plot candle data
            elif (dropdown1) == 'Line Chart':  # if user selects 'Line Chart'
                plot_raw_data()  # plot raw data
            else:  # if user doesn't select any chart
                plot_candle_data()  # plot candle data

# Real-Time Stock Price Section Ends Here

# Stock Price Prediction Section Starts Here
elif(selected == 'Stock Prediction'):  # if user selects 'Stock Prediction'
    st.subheader("Stock Prediction")

    tickers = stock_df["Company Name"]  # get company names from csv file
    
    # dropdown for selecting company
    a = st.selectbox('Pick a Company', tickers)
    with st.spinner('Loading...'):  # spinner while loading
             time.sleep(2)
    dict_csv = pd.read_csv('StockStreamTickersData.csv', header=None, index_col=0).to_dict()[1]  # read csv file
    
    symb_list = []  # list for storing symbols
    val = dict_csv.get(a)  # get symbol from csv file
    symb_list.append(val)  # append symbol to list
    if(a == ""):  # if user doesn't select any company
        st.write("Enter a Stock Name")  # display message
    else:  # if user selects a company
        # download data from yfinance
        testing_percentage = st.slider("Testing Data Percentage", 0.1, 0.9, 0.2, step=0.1)
        validation_percentage = st.slider("Validation Data Percentage", 0.1, 0.9, 0.2, step=0.1)
        train_data_percentage = 1.0 - (testing_percentage + validation_percentage)
        
        prediction_algorithms = st.selectbox('Choose Stock Prediction Algorithms', ['','LSTM', 'CATBoost', 'AllInOne'])
        if(prediction_algorithms==""):
            st.write("Choose")
        if(prediction_algorithms=="LSTM"):
        
            data = yf.download(symb_list, start=start, end=end)
            data.reset_index(inplace=True)
            st.subheader('Raw Data of {}'.format(a))
            st.write(data)
            
            
            df = yf.download(val, start=start, end=end)['Adj Close']
            st.dataframe(df)
            
            split_index_train = int(len(df) * train_data_percentage)
            split_index_test = int(len(df) * (train_data_percentage + testing_percentage))
            train_data, test_data = df[:split_index_train], df[split_index_train:split_index_test]
            validation_data = df[split_index_test:]
            lstm_predictions = lstm_stock_prediction(train_data, test_data, validation_data, validation_percentage)
            test_dates = df.index[-len(test_data):]
            min_length = len(lstm_predictions)
            lstm_df = pd.DataFrame({'Actual': test_data.values[:min_length], 'LSTM': lstm_predictions[:min_length].reshape(-1)}, index=test_dates[:min_length])
            st.write("LSTM Stock Prediction")
            st.line_chart(lstm_df, use_container_width=True)
            lstm_mse = mean_squared_error(test_data[-len(lstm_predictions):], lstm_predictions)
            st.write(f"LSTM Prediction MSE: {lstm_mse}")
            accuracy = 1.0 - lstm_mse / np.var(test_data[-len(lstm_predictions):])
            next_day_prediction = lstm_predictions[-1][0]
            trained_data_percentage = len(train_data) / len(df)
            tested_data_percentage = len(test_data) / len(df)
            validated_data_percentage = len(validation_data) / len(df)
            error_rate = 1.0 - accuracy
            scraper = Nitter()
            tweets = scraper.get_tweets('AAPL', mode='hashtag', number=200)
            
            
            twitter_data = pd.DataFrame([{'Tweet': tweet['text'], 'Date': tweet['date'], 'Likes': tweet['stats']['likes'], "Retweets": tweet['stats']['retweets'], "Comments": tweet['stats']['comments']} for tweet in tweets['tweets']])
            twitter_data['Date'] = pd.to_datetime(twitter_data['Date'], format='%b %d, %Y · %I:%M %p UTC', errors='coerce')

            # Format the 'Date' column as 'YYYY-MM-DD'
            twitter_data['Date'] = twitter_data['Date'].dt.strftime('%Y-%m-%d')

            # Step 2: News Data Retrieval
            sn = stocknews.StockNews('AAPL', save_news=False)  # Use 'stocknews.StockNews' instead of just 'StockNews'
            df_news = sn.read_rss()

            # Now, you can use the news data as before
            news_list = []

            for i in range(40):
                news_item = {
        			'published': df_news['published'][i],
        			'title': df_news['title'][i],
        			'summary': df_news['summary'][i]
    			}
                news_list.append(news_item)

            for news_item in news_list:
                news_item['Title_Sentiment'] = TextBlob(news_item['title']).sentiment.polarity
                news_item['Summary_Sentiment'] = TextBlob(news_item['summary']).sentiment.polarity

            news_data = pd.DataFrame(news_list)
            news_data.rename(columns={'published': 'Date'}, inplace=True)
            news_data['Date'] = pd.to_datetime(news_data['Date'])
            news_data['Date'] = news_data['Date'].dt.strftime('%Y-%m-%d')
            tmp_news= news_data

            for news_item in news_list:
                news_item['Title_Sentiment'] = TextBlob(news_item['title']).sentiment.polarity
                news_item['Summary_Sentiment'] = TextBlob(news_item['summary']).sentiment.polarity

            news_data = pd.DataFrame(news_list)
            news_data.rename(columns={'published': 'Date'}, inplace=True)
            news_data['Date'] = pd.to_datetime(news_data['Date'])
            news_data['Date'] = news_data['Date'].dt.strftime('%Y-%m-%d')
            tmp_news= news_data
            tmp_tweet=twitter_data


            # Step 3: Historical Stock Data Retrieval
            stock_symbol = 'AAPL'
            historical_data = yf.download(stock_symbol, start='2021-01-01', end='2023-11-29')

            # Step 3: Historical Stock Data Retrieval
            stock_symbol = 'AAPL'
            historical_data = yf.download(stock_symbol, start='2021-01-01', end='2023-11-29')

            twitter_data['Tweet_Sentiment'] = twitter_data['Tweet'].apply(lambda x: TextBlob(x).sentiment.polarity)
            tmp_tweet=twitter_data
            intersection_data = pd.merge(tmp_news, tmp_tweet, on='Date', how='inner')

            #sentiment_columns = ['Title_Sentiment', 'Summary_Sentiment', 'Tweet_Sentiment']

            # Calculate the mean along the rows (axis=1)
            #intersection_data['Mean_Sentiment'] = intersection_data[sentiment_columns].mean(axis=1)

            # Drop the individual sentiment columns if needed
            #intersection_data = intersection_data.drop(columns=sentiment_columns)

            stock_symbol = 'AAPL'
            historical_data = yf.download(stock_symbol, start='2021-01-01', end='2023-12-2')

            # Reset index to have 'Date' as a column
            historical_data.reset_index(inplace=True)

            intersection_data['Date'] = pd.to_datetime(intersection_data['Date'], errors='coerce')

            # Assuming 'Date' is the column containing the date in the historical_data DataFrame
            historical_data['Date'] = pd.to_datetime(historical_data['Date'], errors='coerce')

            # Now, you can proceed with the merge
            merged_data1 = pd.merge(intersection_data, historical_data, on='Date', how='inner')

            intersection_data['Date'] = pd.to_datetime(intersection_data['Date'], errors='coerce')

            # Assuming 'Date' is the column containing the date in the historical_data DataFrame
            historical_data['Date'] = pd.to_datetime(historical_data['Date'], errors='coerce')

            # Now, you can proceed with the merge
            merged_data1 = pd.merge(intersection_data, historical_data, on='Date', how='inner')

            merged_data1['Date'] = pd.to_datetime(merged_data1['Date'])
            sentiment_columns = ['Title_Sentiment', 'Summary_Sentiment', 'Tweet_Sentiment']




            # Calculate the mean along the rows (axis=1)
            intersection_data['Mean_Sentiment'] = intersection_data[sentiment_columns].mean(axis=1)

            #Drop the individual sentiment columns if needed
            intersection_data = intersection_data.drop(columns=sentiment_columns)


            merged_data1 = pd.merge(intersection_data, historical_data, on='Date', how='inner')
            result_df = merged_data1.groupby('Date', as_index=False)['Mean_Sentiment'].mean()
            result_df['Ticker']= 'AAPL'
            result_df['Category']= 'Tech'
            overall_mean_sentiment = result_df['Mean_Sentiment'].mean()
            
            # Define the stock symbol (AAPL for Apple Inc.)
            stock_symbol = "AAPL"

            # Fetch the stock data for the recent 1 day
            stock_d = yf.download(stock_symbol, period="1d")

            # Extract the open price from the stock data
            recent_day_open_price = stock_d['Open'][0]
            
            #st.write(f"Sentiment: {overall_mean_sentiment:.2f}")
            
            if(overall_mean_sentiment>0.05):
            	st.write(f"Sentiment: {overall_mean_sentiment:.2f}")
            	
            	if(next_day_prediction>recent_day_open_price):
            		st.write(f"Stock Trend is Up")
            	else:
            		st.write(f"Stock Trend is Neutral")
            elif(overall_mean_sentiment<-0.05):
            	st.write(f"Sentiment: {overall_mean_sentiment:.2f}")
            	
            	if(next_day_predictionrecent_day_open_price):
            		st.write(f"Stock Trend is Down")
            	else:
            		st.write(f"Stock Trend is Neutral")
            else:
            	st.write(f"Unable To Predict")

            
            st.write(f"Accuracy: {accuracy:.4f}")
            st.write(f"Next Day Price Prediction: {next_day_prediction:.2f}")
            st.write(f"Trained Data Percentage: {trained_data_percentage:.2%}")
            st.write(f"Tested Data Percentage: {tested_data_percentage:.2%}")
            st.write(f"Validated Data Percentage: {validated_data_percentage:.2%}")
            st.write(f"Error Rate: {error_rate:.2%}")
            
            
            st.write(merged_data1)

            
        if(prediction_algorithms=="CATBoost"):
            data = download_stock_data(val, start, end)
            # CatBoost model
            catboost_model = CatBoostRegressor()
            features = data.drop('Close', axis=1)
            target = data['Close']
            catboost_model.fit(features, target)
            
            predictions_catboost = make_catboost_predictions(catboost_model, features)
            
            window_size = st.slider("LSTM Window Size", min_value=1, max_value=30, value=10)
            accuracy, error_rate = calculate_error_metrics(predictions_catboost, target[window_size:])
            
            num_predictions_catboost = min(len(data), len(predictions_catboost))
            fig_catboost, ax_catboost = plt.subplots(1, sharex=True, figsize=(10, 8))
            actual_prices_catboost = target
            ax_catboost.plot(data.index, actual_prices_catboost, label='Actual Price', linewidth=2)
            ax_catboost.plot(data.index[window_size:window_size + num_predictions_catboost], predictions_catboost[:num_predictions_catboost], label='CatBoost Predictions', linestyle='--', color='green')
            ax_catboost.set_xlabel('Date')
            ax_catboost.set_ylabel('Stock Price')
            ax_catboost.legend()
            
            st.pyplot(fig_catboost)
            
            st.write(f"Mean Absolute Percentage Error (MAPE): {accuracy:.2f}%")
            st.write(f"Mean Absolute Error (MAE): ${error_rate:.2f}")
            next_day_prediction = catboost_model.predict(features.iloc[-1, :].values.reshape(1, -1))[0]
            st.write(f"Next day's predicted stock price: ${next_day_prediction:.2f}")
            
            
            
        if(prediction_algorithms=="AllInOne"):
            st.write("Enter Prediction Algorithm")
            st.set_option('deprecation.showPyplotGlobalUse', False)
            
            start_date_default = start
            end_date_default = end
            symbol = st.sidebar.text_input("Enter stock symbol:", value=val)
            start_date = st.sidebar.date_input("Start date", value=start)
            end_date = st.sidebar.date_input("End date", value=end)
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

            lstm_data = data['Close'].values.reshape(-1, 1)
            lstm_data_normalized = lstm_data / np.max(lstm_data)
            X_lstm, y_lstm = preprocess_data_for_lstm(lstm_data_normalized, window_size)
            lstm_model = create_lstm_model(window_size)
            lstm_model.fit(X_lstm, y_lstm, epochs=50, batch_size=32, validation_split=0.1)  # Add validation split for monitoring training progress
            
            
            
            
            stock_data = load_data(val, start_date, end_date)['Close'].values.reshape(-1, 1)
            last_date_stock_price = stock_data[-1]
            #st.write(f"Most Recent Stock Price: ${most_recent_stock_price:.2f}")
            formatted_stock_price = float(last_date_stock_price)
            

            # Make predictions
            predictions_catboost, predictions_xgboost, predictions_lstm = make_predictions(
        catboost_model, xgboost_model, lstm_model, data, window_size
    )

            df = yf.download(val, start=start, end=end)['Adj Close']
            st.dataframe(df)
            
            
            
            
            split_index_train = int(len(df) * train_data_percentage)
            split_index_test = int(len(df) * (train_data_percentage + testing_percentage))
            train_data, test_data = df[:split_index_train], df[split_index_train:split_index_test]
            
            validation_data = df[split_index_test:]
            lstm_predictions = lstm_stock_prediction(train_data, test_data, validation_data, validation_percentage)
            test_dates = df.index[-len(test_data):]
            min_length = len(lstm_predictions)
            lstm_df = pd.DataFrame({'Actual': test_data.values[:min_length], 'LSTM': lstm_predictions[:min_length].reshape(-1)}, index=test_dates[:min_length])
            
            
            st.write(lstm_df)
            st.write("LSTM Stock Prediction")
            st.line_chart(lstm_df, use_container_width=True)
            lstm_mse = mean_squared_error(test_data[-len(lstm_predictions):], lstm_predictions)
            st.write(f"LSTM Prediction MSE: {lstm_mse}")
            accuracy = 1.0 - lstm_mse / np.var(test_data[-len(lstm_predictions):])
            next_day_prediction = lstm_predictions[-1][0]
            trained_data_percentage = len(train_data) / len(df)
            tested_data_percentage = len(test_data) / len(df)
            validated_data_percentage = len(validation_data) / len(df)
            error_rate = 1.0 - accuracy
            st.write(f"Accuracy: {accuracy:.4f}")
            st.write(f"Next Day Price Prediction: {next_day_prediction:.2f}")
            st.write(f"Trained Data Percentage: {trained_data_percentage:.2%}")
            st.write(f"Tested Data Percentage: {tested_data_percentage:.2%}")
            st.write(f"Validated Data Percentage: {validated_data_percentage:.2%}")
            st.write(f"Error Rate: {error_rate:.2%}")
            predictions_combined = (predictions_catboost + predictions_xgboost + next_day_prediction) / 3.0

            # Plotting
            fig, ax = plt.subplots(1, sharex=True, figsize=(10, 8))

            # Actual Prices
            actual_prices = data['Close'][window_size:]
            ax.plot(data.index[window_size:], actual_prices, label='Actual Price', linewidth=2)

            # Ensure predictions have the same length as actual prices
            
            num_predictions_catboost = min(len(actual_prices), len(predictions_catboost))
            num_predictions_xgboost = min(len(actual_prices), len(predictions_xgboost))
            num_predictions_lstm = min(len(actual_prices), len(lstm_df))
            num_predictions_combined = min(len(actual_prices), len(predictions_combined))


            ax.plot(data.index[window_size:window_size + num_predictions_catboost], predictions_catboost[:num_predictions_catboost], label='CatBoost Predictions', linestyle='--', color='green')
            ax.plot(data.index[window_size:window_size + num_predictions_xgboost], predictions_xgboost[:num_predictions_xgboost], label='XGBoost Predictions', linestyle='--', color='red')
            #ax.plot(data.index[window_size:window_size + num_predictions_lstm], lstm_df[:num_predictions_lstm], label='LSTM Predictions', linestyle='--', color='purple')
            

            # Legends and labels for the first plot
            ax.legend()
            ax.set_title(f'Stock Price Prediction for {symbol}')
            ax.set_ylabel('Stock Price')

            plt.tight_layout()
            st.pyplot()

            # Predict next price
            next_price_catboost = catboost_model.predict(features.iloc[[-1]])[0]
            next_price_xgboost = xgboost_model.predict(features.iloc[[-1]])[0]
            # Normalize the features for predicting the next prices using the same scaling as during training
            next_price_lstm_data = (lstm_data[-window_size:] - np.min(lstm_data)) / (np.max(lstm_data) - np.min(lstm_data))

            # Predict next price using the LSTM model
            next_price_lstm = lstm_model.predict(next_price_lstm_data.reshape(1, window_size, 1))[0][0]

            next_price_combined = (next_price_catboost + next_price_xgboost + next_day_prediction) / 3.0
            #most_recent_stock_price = stock_data['Close'].iloc[-1]
            

            st.subheader("Next Price Predictions:")
            st.write(f"Date: 30-11-2023")
            st.write(f'The last date\'s stock price for {val} is: ${formatted_stock_price:.2f}')

            st.write(f"CatBoost: {next_price_catboost}")
            st.write(f"XGBoost: {next_price_xgboost}")
            st.write(f"LSTM: {next_day_prediction}")
            st.write(f"Combined: {next_price_combined}")
            
            scraper = Nitter()
            tweets = scraper.get_tweets('AAPL', mode='hashtag', number=200)
            
            
            twitter_data = pd.DataFrame([{'Tweet': tweet['text'], 'Date': tweet['date'], 'Likes': tweet['stats']['likes'], "Retweets": tweet['stats']['retweets'], "Comments": tweet['stats']['comments']} for tweet in tweets['tweets']])
            twitter_data['Date'] = pd.to_datetime(twitter_data['Date'], format='%b %d, %Y · %I:%M %p UTC', errors='coerce')

            # Format the 'Date' column as 'YYYY-MM-DD'
            twitter_data['Date'] = twitter_data['Date'].dt.strftime('%Y-%m-%d')

            # Step 2: News Data Retrieval
            sn = stocknews.StockNews('AAPL', save_news=False)  # Use 'stocknews.StockNews' instead of just 'StockNews'
            df_news = sn.read_rss()

            # Now, you can use the news data as before
            news_list = []

            for i in range(40):
                news_item = {
        			'published': df_news['published'][i],
        			'title': df_news['title'][i],
        			'summary': df_news['summary'][i]
    			}
                news_list.append(news_item)

            for news_item in news_list:
                news_item['Title_Sentiment'] = TextBlob(news_item['title']).sentiment.polarity
                news_item['Summary_Sentiment'] = TextBlob(news_item['summary']).sentiment.polarity

            news_data = pd.DataFrame(news_list)
            news_data.rename(columns={'published': 'Date'}, inplace=True)
            news_data['Date'] = pd.to_datetime(news_data['Date'])
            news_data['Date'] = news_data['Date'].dt.strftime('%Y-%m-%d')
            tmp_news= news_data

            for news_item in news_list:
                news_item['Title_Sentiment'] = TextBlob(news_item['title']).sentiment.polarity
                news_item['Summary_Sentiment'] = TextBlob(news_item['summary']).sentiment.polarity

            news_data = pd.DataFrame(news_list)
            news_data.rename(columns={'published': 'Date'}, inplace=True)
            news_data['Date'] = pd.to_datetime(news_data['Date'])
            news_data['Date'] = news_data['Date'].dt.strftime('%Y-%m-%d')
            tmp_news= news_data
            tmp_tweet=twitter_data


            # Step 3: Historical Stock Data Retrieval
            stock_symbol = 'AAPL'
            historical_data = yf.download(stock_symbol, start='2021-01-01', end='2023-11-29')

            # Step 3: Historical Stock Data Retrieval
            stock_symbol = 'AAPL'
            historical_data = yf.download(stock_symbol, start='2021-01-01', end='2023-11-29')

            twitter_data['Tweet_Sentiment'] = twitter_data['Tweet'].apply(lambda x: TextBlob(x).sentiment.polarity)
            tmp_tweet=twitter_data
            intersection_data = pd.merge(tmp_news, tmp_tweet, on='Date', how='inner')

            #sentiment_columns = ['Title_Sentiment', 'Summary_Sentiment', 'Tweet_Sentiment']

            # Calculate the mean along the rows (axis=1)
            #intersection_data['Mean_Sentiment'] = intersection_data[sentiment_columns].mean(axis=1)

            # Drop the individual sentiment columns if needed
            #intersection_data = intersection_data.drop(columns=sentiment_columns)

            stock_symbol = 'AAPL'
            historical_data = yf.download(stock_symbol, start='2021-01-01', end='2023-11-29')

            # Reset index to have 'Date' as a column
            historical_data.reset_index(inplace=True)

            intersection_data['Date'] = pd.to_datetime(intersection_data['Date'], errors='coerce')

            # Assuming 'Date' is the column containing the date in the historical_data DataFrame
            historical_data['Date'] = pd.to_datetime(historical_data['Date'], errors='coerce')

            # Now, you can proceed with the merge
            merged_data1 = pd.merge(intersection_data, historical_data, on='Date', how='inner')

            intersection_data['Date'] = pd.to_datetime(intersection_data['Date'], errors='coerce')

            # Assuming 'Date' is the column containing the date in the historical_data DataFrame
            historical_data['Date'] = pd.to_datetime(historical_data['Date'], errors='coerce')

            # Now, you can proceed with the merge
            merged_data1 = pd.merge(intersection_data, historical_data, on='Date', how='inner')

            merged_data1['Date'] = pd.to_datetime(merged_data1['Date'])
            sentiment_columns = ['Title_Sentiment', 'Summary_Sentiment', 'Tweet_Sentiment']




            # Calculate the mean along the rows (axis=1)
            intersection_data['Mean_Sentiment'] = intersection_data[sentiment_columns].mean(axis=1)

            #Drop the individual sentiment columns if needed
            intersection_data = intersection_data.drop(columns=sentiment_columns)


            merged_data1 = pd.merge(intersection_data, historical_data, on='Date', how='inner')
            result_df = merged_data1.groupby('Date', as_index=False)['Mean_Sentiment'].mean()
            result_df['Ticker']= 'AAPL'
            result_df['Category']= 'Tech'
            overall_mean_sentiment = result_df['Mean_Sentiment'].mean()
            
            # Define the stock symbol (AAPL for Apple Inc.)
            stock_symbol = "AAPL"

            # Fetch the stock data for the recent 1 day
            stock_d = yf.download(stock_symbol, period="1d")

            # Extract the open price from the stock data
            recent_day_open_price = stock_d['Open'][0]
            
            #st.write(f"Sentiment: {overall_mean_sentiment:.2f}")
            
            if(overall_mean_sentiment>0.05):
            	st.write(f"Sentiment: {overall_mean_sentiment:.2f}")
            	
            	if(next_day_prediction>recent_day_open_price):
            		st.write(f"Stock Trend is Up")
            	else:
            		st.write(f"Stock Trend is Neutral")
            elif(overall_mean_sentiment<-0.05):
            	st.write(f"Sentiment: {overall_mean_sentiment:.2f}")
            	
            	if(next_day_predictionrecent_day_open_price):
            		st.write(f"Stock Trend is Down")
            	else:
            		st.write(f"Stock Trend is Neutral")
            else:
            	st.write(f"Unable To Predict")


            
            # Fundamental Data:
            pricing_dataset, fundamental_dataset, news_dataset, tweet_dataset = st.tabs(['Pricing Data', 'Fundamental Data', 'Top 10 News', 'Twitter Data'])
            if pricing_dataset:
            	st.header('Price Movements')
            	data2 = data
            	data2['% Change'] = data['Adj Close'] / data['Adj Close'].shift(1) - 1
            	data2.dropna(inplace = True)
            	st.write(data2)
            	annual_return = data2['% Change'].mean()*252*100
            	st.write('Annual Returns is ', annual_return, '%')
            	stdev = np.std(data2['% Change']*np.sqrt(252))
            	st.write('Standard Deviation is ', stdev*100, '%')
            	st.write('Risk Adj. Return is ', annual_return/(stdev*100))
        
            
        	
	
            from alpha_vantage.fundamentaldata import FundamentalData
            with fundamental_dataset:
            	ticker=a
            	st.header('Fundamental Data')
            	
            	balance_sheet, income_statement, cash_flow_statement = st.tabs(['Balance Sheet', 'Income Statement', 'Cash Flow Statement'])
            	key = 'U5YPTYM5MNLHJR3E'#XY050M3WU6CETL2D'
        	#U5YPTYM5MNLHJR3E
        	#XY050M3WU6CETL2C
        	#XY050M3WU6CETL2D
        	
            	fd = FundamentalData(key, output_format = 'pandas')
            	with balance_sheet:
            		st.subheader('Balance Sheet')
            		balance_sheet = fd.get_balance_sheet_annual(ticker)[0]
            		bs = balance_sheet.T[2:]
            		bs.columns = list(balance_sheet.T.iloc[0])
            		st.write(bs)
            	with income_statement:
            		st.subheader('Income Statement')
            		income_statement = fd.get_income_statement_annual(ticker)[0]
            		is1 = income_statement.T[2:]
            		is1.columns = list(income_statement.T.iloc[0])
            		st.write(is1)
            	with cash_flow_statement:
            		st.subheader('Cash Flow Statement')
            		cash_flow = fd.get_cash_flow_annual(ticker)[0]
            		cf = cash_flow.T[2:]
            		cf.columns = list(cash_flow.T.iloc[0])
            		st.write(cf)
        		#Alpha Vantage
        	
        
            	import stocknews
            	with news_dataset:
            		st.header(f'News Data About {ticker}')
            		sn = StockNews(ticker, save_news=False)
            		df_news = sn.read_rss()
            		for i in range(10):
            			st.subheader(f'News {i+1}')
            			st.write(df_news['published'][i])
            			st.write(df_news['title'][i])
            			st.write(df_news['summary'][i])
            			title_sentiment = df_news['sentiment_title'][i]
            			st.write(f'Title Sentiment {title_sentiment}')
            			news_sentiment = df_news['sentiment_summary'][i]
            			st.write(f'News Sentiment {news_sentiment}')
            with tweet_dataset:
            	ticker=a
            	st.header(f'Twitter Data About {ticker}')
            	tweet_data_file = 'NewTechTweetData.csv'
            	tweet_data1 = pd.read_csv(tweet_data_file, encoding='utf-8', sep='delimeter')

            	st.write(tweet_data1)
            	st.dataframe(tweet_data1)
             	#analyze_aapl_sentiment(tweet_data_file)

        
        data = yf.download(symb_list, start=start, end=end)
        data.reset_index(inplace=True)  # reset index
        st.subheader('Raw Data of {}'.format(a))  # display raw data
        st.write(data)  # display data

        def plot_raw_data():  # function for plotting raw data
            fig = go.Figure()  # create figure
            fig.add_trace(go.Scatter(  # add scatter plot
                x=data['Date'], y=data['Open'], name="stock_open"))  # x-axis: date, y-axis: open
            fig.add_trace(go.Scatter(  # add scatter plot
                x=data['Date'], y=data['Close'], name="stock_close"))  # x-axis: date, y-axis: close
            fig.layout.update(  # update layout
                title_text='Time Series Data of {}'.format(a), xaxis_rangeslider_visible=True)  # title, x-axis: rangeslider
            st.plotly_chart(fig)  # display plotly chart

        plot_raw_data()  # plot raw data
        # slider for selecting number of years
        n_years = st.slider('Years of prediction:', 1, 4)
        period = n_years * 365  # calculate number of days

        # Predict forecast with Prophet
        # create dataframe for training data
        df_train = data[['Date', 'Close']]
        df_train = df_train.rename(
            columns={"Date": "ds", "Close": "y"})  # rename columns

        m = Prophet()  # create object for prophet
        m.fit(df_train)  # fit data to prophet
        future = m.make_future_dataframe(
            periods=period)  # create future dataframe
        forecast = m.predict(future)  # predict future dataframe

        # Show and plot forecast
        st.subheader('Forecast Data of {}'.format(a))  # display forecast data
        st.write(forecast)  # display forecast data

        st.subheader(f'Forecast plot for {n_years} years')  # display message
        fig1 = plot_plotly(m, forecast)  # plot forecast
        st.plotly_chart(fig1)  # display plotly chart

        st.subheader("Forecast components of {}".format(a))  # display message
        fig2 = m.plot_components(forecast)  # plot forecast components
        st.write(fig2)  # display plotly chart

