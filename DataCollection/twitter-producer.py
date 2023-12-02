import json
from kafka import KafkaProducer
import pandas as pd
from ntscraper import Nitter
from textblob import TextBlob

# Kafka setup
bootstrap_servers = 'localhost:9092'
topic = 'twitter_data'
producer = KafkaProducer(bootstrap_servers=bootstrap_servers,
                         value_serializer=lambda v: json.dumps(v).encode('utf-8'))


def produce_twitter_data(ticker):
    # Twitter data retrieval
    scraper = Nitter()
    tweets = scraper.get_tweets(ticker, mode='hashtag', number=5)
    twitter_data = pd.DataFrame([{
        'Ticker': ticker,
        'Tweet': tweet['text'],
        'Date': tweet['date'],
        'Likes': tweet['stats']['likes'],
        'Retweets': tweet['stats']['retweets'],
        'Comments': tweet['stats']['comments']
    } for tweet in tweets['tweets']])

    # Sentiment analysis using TextBlob
    twitter_data['Tweet_Sentiment'] = twitter_data['Tweet'].apply(lambda x: TextBlob(x).sentiment.polarity)

    # Format the 'Date' column as 'YYYY-MM-DD'
    twitter_data['Date'] = pd.to_datetime(twitter_data['Date'], format='%b %d, %Y · %I:%M %p UTC', errors='coerce')
    twitter_data['Date'] = twitter_data['Date'].dt.strftime('%Y-%m-%d')

    for _, tweet in twitter_data.iterrows():
        twitter_item = {
            'Ticker': tweet['Ticker'],
            'Tweet': tweet['Tweet'],
            'Date': tweet['Date'],
            'Likes': tweet['Likes'],
            'Retweets': tweet['Retweets'],
            'Comments': tweet['Comments'],
            'Tweet_Sentiment': tweet['Tweet_Sentiment']
        }
        producer.send(topic, value=twitter_item)
        print(f"Produced Twitter Data: {twitter_item}")

if __name__ == "__main__":
    ticker = 'AAPL'  # Replace with the desired ticker
    produce_twitter_data(ticker)
