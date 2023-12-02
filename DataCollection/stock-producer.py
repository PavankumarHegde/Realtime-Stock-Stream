from kafka import KafkaProducer
import yfinance as yf
import json
import time

# Kafka setup
bootstrap_servers = 'localhost:9092'
topic = 'stock_data'
producer = KafkaProducer(bootstrap_servers=bootstrap_servers,
                         value_serializer=lambda v: json.dumps(v).encode('utf-8'))

def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period='1d')
    return data[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[-1]

def produce_stock_data():
    ticker = 'AAPL'  # Example stock symbol, replace with your desired symbol
    while True:
        stock_data = fetch_stock_data(ticker)
        producer.send(topic, value=stock_data.to_dict())
        print(f"Produced: {stock_data.to_dict()}")
        time.sleep(10)  # Adjust the sleep duration as needed

if __name__ == "__main__":
    produce_stock_data()
