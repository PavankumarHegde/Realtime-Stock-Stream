import json
import requests
from kafka import KafkaProducer
from textblob import TextBlob
import time

news_api_key = '8d95451d7aa3489da66c8aaee534c062'

# Kafka setup
bootstrap_servers = 'localhost:9092'
topic = 'news_data'
producer = KafkaProducer(bootstrap_servers=bootstrap_servers,
                         value_serializer=lambda v: json.dumps(v).encode('utf-8'))

def fetch_news_articles(ticker):
    while True:
        api_url = f'https://newsapi.org/v2/everything?q={ticker}&apiKey={news_api_key}'
        response = requests.get(api_url)
        news_data = response.json()

        for article in news_data.get('articles', []):
            news_item = {
                'ticker': ticker,
                'title': article.get('title', ''),
                'summary': article.get('description', ''),
                'url': article.get('url', ''),
                'published_at': article.get('publishedAt', ''),
                'sentiment_title': TextBlob(article.get('title', '')).sentiment.polarity,
                'sentiment_summary': TextBlob(article.get('description', '')).sentiment.polarity
            }

            # Calculate overall sentiment as an average of title and summary sentiments
            news_item['overall_sentiment'] = (news_item['sentiment_title'] + news_item['sentiment_summary']) / 2

            producer.send(topic, value=news_item)
            print(f"Produced News Article: {news_item}")

        # Introduce a delay between iterations (e.g., every 10 minutes)
        time.sleep(600)

if __name__ == "__main__":
    fetch_news_articles('AAPL')

