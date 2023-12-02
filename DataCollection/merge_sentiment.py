import sqlite3
import pandas as pd
from datetime import datetime

# SQLite setup
twitter_db_path = 'twitter_data.db'
reddit_db_path = 'reddit_data.db'
news_db_path = 'news_data.db'

# Function to fetch data from SQLite database
def fetch_data(db_path, table_name):
    conn = sqlite3.connect(db_path)
    query = f'SELECT * FROM {table_name};'
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# Function to calculate average sentiment
def calculate_average_sentiment(data, sentiment_columns):
    data['average_sentiment'] = data[sentiment_columns].mean(axis=1)
    return data

# Function to filter data for the same date
def filter_data_for_same_date(data, date_column, target_date):
    data['formatted_date'] = pd.to_datetime(data[date_column]).dt.strftime('%Y-%m-%d')
    filtered_data = data[data['formatted_date'] == target_date]
    return filtered_data

# Fetch data from databases
twitter_data = fetch_data(twitter_db_path, 'twitter_data')
reddit_data = fetch_data(reddit_db_path, 'reddit_data')
news_data = fetch_data(news_db_path, 'news_data')

# Set the target date for analysis
target_date = datetime.now().strftime('%Y-%m-%d')

# Filter data for the same date
twitter_data_filtered = filter_data_for_same_date(twitter_data, 'tweet_date', target_date)
reddit_data_filtered = filter_data_for_same_date(reddit_data, 'reddit_created_utc', target_date)
news_data_filtered = filter_data_for_same_date(news_data, 'news_published_at', target_date)

# Check if 'ticker' column exists in both DataFrames before merging
if 'ticker' not in twitter_data_filtered.columns or 'ticker' not in reddit_data_filtered.columns:
    raise ValueError("Both DataFrames must have a 'ticker' column for merging.")

# Calculate average sentiment for each dataset
sentiment_columns_twitter = ['tweet_sentiment']
sentiment_columns_reddit = ['reddit_sentiment']
sentiment_columns_news = ['news_sentiment_title', 'news_sentiment_summary', 'news_overall_sentiment']

twitter_data_filtered = calculate_average_sentiment(twitter_data_filtered, sentiment_columns_twitter)
reddit_data_filtered = calculate_average_sentiment(reddit_data_filtered, sentiment_columns_reddit)
news_data_filtered = calculate_average_sentiment(news_data_filtered, sentiment_columns_news)

# Merge dataframes based on the common 'ticker' column
merged_data = pd.merge(twitter_data_filtered, reddit_data_filtered, on='ticker', how='outer')
merged_data = pd.merge(merged_data, news_data_filtered, on='ticker', how='outer')

# Save the merged data to a CSV file
output_file_path = 'sentiment_analysis_output.csv'
merged_data.to_csv(output_file_path, index=False)

print(f"Sentiment analysis results saved to {output_file_path}")

