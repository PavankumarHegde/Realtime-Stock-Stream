from kafka import KafkaConsumer
import sqlite3
import json

# Kafka setup
bootstrap_servers = 'localhost:9092'
topic = 'twitter_data'
consumer = KafkaConsumer(topic, bootstrap_servers=bootstrap_servers,
                         value_deserializer=lambda v: json.loads(v.decode('utf-8')))

# SQLite setup
conn = sqlite3.connect('twitter_data.db')
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS twitter_data (
        ticker TEXT,
        tweet_text TEXT,
        tweet_date TEXT,
        likes INTEGER,
        retweets INTEGER,
        comments INTEGER,
        tweet_sentiment REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
''')
conn.commit()

def store_twitter_data(data):
    cursor.execute('''
        INSERT INTO twitter_data (ticker, tweet_text, tweet_date, likes, retweets, comments, tweet_sentiment)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (data['Ticker'], data['Tweet'], data['Date'], data['Likes'], data['Retweets'], data['Comments'], data['Tweet_Sentiment']))
    conn.commit()

def consume_twitter_data():
    for message in consumer:
        twitter_data = message.value
        print(f"Consumed Twitter Data: {twitter_data}")
        store_twitter_data(twitter_data)

if __name__ == "__main__":
    consume_twitter_data()

