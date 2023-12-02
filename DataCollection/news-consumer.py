from kafka import KafkaConsumer
import sqlite3
import json

# Kafka setup
bootstrap_servers = 'localhost:9092'
topic = 'news_data'
consumer = KafkaConsumer(topic, bootstrap_servers=bootstrap_servers,
                         value_deserializer=lambda v: json.loads(v.decode('utf-8')))

# SQLite setup
conn = sqlite3.connect('news_data.db')
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS news_data (
        ticker TEXT,
        news_title TEXT,
        news_summary TEXT,
        news_url TEXT,
        news_published_at TEXT,
        news_sentiment_title REAL,
        news_sentiment_summary REAL,
        news_overall_sentiment REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
''')
conn.commit()

def store_news_data(data):
    cursor.execute('''
        INSERT INTO news_data (
            ticker, news_title, news_summary, news_url,
            news_published_at, news_sentiment_title, news_sentiment_summary, news_overall_sentiment
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        data['ticker'], data['title'], data['summary'], data['url'],
        data['published_at'], data['sentiment_title'], data['sentiment_summary'], data['overall_sentiment']
    ))
    conn.commit()

def consume_news_data():
    for message in consumer:
        news_data = message.value
        print(f"Consumed News Article: {news_data}")
        store_news_data(news_data)

if __name__ == "__main__":
    consume_news_data()

