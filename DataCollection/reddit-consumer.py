from kafka import KafkaConsumer
import sqlite3
import json

# Kafka setup
bootstrap_servers = 'localhost:9092'
topic = 'reddit_data'
consumer = KafkaConsumer(topic, bootstrap_servers=bootstrap_servers,
                         value_deserializer=lambda v: json.loads(v.decode('utf-8')))

# SQLite setup
conn = sqlite3.connect('reddit_data.db')
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS reddit_data (
        ticker TEXT,
        reddit_title TEXT,
        reddit_url TEXT,
        reddit_score INTEGER,
        reddit_created_utc INTEGER,
        reddit_sentiment REAL,  -- Add sentiment column for Reddit title
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
''')
conn.commit()

def store_reddit_data(data):
    cursor.execute('''
        INSERT INTO reddit_data (ticker, reddit_title, reddit_url, reddit_score, reddit_created_utc, reddit_sentiment)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (data['ticker'], data['title'], data['url'], data['score'], data['created_utc'], data['sentiment']))
    conn.commit()

def consume_reddit_data():
    for message in consumer:
        reddit_data = message.value
        print(f"Consumed Reddit Post: {reddit_data}")
        store_reddit_data(reddit_data)

if __name__ == "__main__":
    consume_reddit_data()

