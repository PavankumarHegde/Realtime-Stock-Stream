from kafka import KafkaConsumer
import sqlite3
import json

# Kafka setup
bootstrap_servers = 'localhost:9092'
topic = 'stock_data'
consumer = KafkaConsumer(topic, bootstrap_servers=bootstrap_servers,
                         value_deserializer=lambda v: json.loads(v.decode('utf-8')))

# SQLite setup
conn = sqlite3.connect('stock_data.db')
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS stock_data (
        symbol TEXT,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume INTEGER,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
''')
conn.commit()

def store_stock_data(data, symbol):
    cursor.execute('''
        INSERT INTO stock_data (symbol, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (symbol, data['Open'], data['High'], data['Low'], data['Close'], data['Volume']))
    conn.commit()

def consume_stock_data():
    symbol = 'AAPL'  # Replace with the actual symbol you used in the producer script
    for message in consumer:
        stock_data = message.value
        print(f"Consumed: {stock_data}")
        store_stock_data(stock_data, symbol)


if __name__ == "__main__":
    consume_stock_data()
