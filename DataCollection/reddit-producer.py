import praw
import json
from kafka import KafkaProducer
import time
from textblob import TextBlob

# Reddit API setup
reddit = praw.Reddit(
    client_id='qYtFIdWq6kTcN9-s_SXgsg',
    client_secret='BqN97vMHXNqQLh_DQM-pJG4D63P0ng',
    user_agent='AcademinResearch/1.0 (Linux; Python/3.8)',
)

# Kafka setup
bootstrap_servers = 'localhost:9092'
topic = 'reddit_data'
producer = KafkaProducer(bootstrap_servers=bootstrap_servers,
                         value_serializer=lambda v: json.dumps(v).encode('utf-8'))

def fetch_reddit_posts(ticker):
    subreddit = reddit.subreddit('stocks')  # You can change this to a different subreddit
    keyword = f'{ticker}'  # Use the provided ticker

    while True:
        for submission in subreddit.search(keyword, time_filter='day', sort='new', limit=5):
            post_data = {
                'ticker': ticker,
                'title': submission.title,
                'url': submission.url,
                'score': submission.score,
                'created_utc': submission.created_utc,
                'sentiment': TextBlob(submission.title).sentiment.polarity  # Sentiment for Reddit title
            }
            producer.send(topic, value=post_data)
            print(f"Produced Reddit Post: {post_data}")

        # Introduce a delay between iterations (e.g., every 10 minutes)
        time.sleep(600)

if __name__ == "__main__":
    fetch_reddit_posts('AAPL')

