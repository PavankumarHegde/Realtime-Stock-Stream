from kafka import KafkaConsumer
import sqlite3
import json

# Kafka setup
bootstrap_servers = 'localhost:9092'
topic = 'stocknews_data'
consumer = KafkaConsumer(topic, bootstrap_servers=bootstrap_servers,
                         value_deserializer=lambda v: json.loads(v.decode('utf-8')))

# SQLite setup
conn = sqlite3.connect('stocknews_data.db')
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS stocknews_data (
        title TEXT,
        description TEXT,
        url TEXT,
        published_at DATETIME,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
''')
conn.commit()

def store_stocknews_data(data):
    cursor.execute('''
        INSERT INTO stocknews_data (title, description, url, published_at)
        VALUES (?, ?, ?, ?)
    ''', (data['title'], data['description'], data['url'], data['published_at']))
    conn.commit()

def consume_stocknews_data():
    for message in consumer:
        stocknews_data = message.value
        print(f"Consumed Stock News: {stocknews_data}")
        store_stocknews_data(stocknews_data)

if __name__ == "__main__":
    consume_stocknews_data()
