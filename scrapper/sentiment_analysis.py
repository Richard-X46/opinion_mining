import os
import openai
import pandas as pd
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Groq API client setup
client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

class SentimentAnalyzer:
    def __init__(self):
        self.model = "llama3-70b-8192"  # Or "mixtral-8x7b-32768"

    def analyze_sentiment_batch(self, comments, batch_size=10):
        """Analyze sentiment for multiple comments in a single API call"""
        combined_text = "\n---\n".join(comments)
        
        try:
            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": "Analyze the sentiment of each comment. Reply with only comma-separated values: NEGATIVE, NEUTRAL, or POSITIVE"},
                    {"role": "user", "content": combined_text}
                ],
                max_tokens=100,
                temperature=0.1
            )
            
            results = response.choices[0].message.content.strip().split(',')
            return [result.strip().upper() for result in results]
        except Exception as e:
            print(f"Batch sentiment analysis error: {e}")
            return ['NEUTRAL'] * len(comments)  # Fallback to neutral

if __name__ == "__main__":
    results = analyze_comments()

    if results:
        sentiments = [r['sentiment'] for r in results]
        total = len(sentiments)
        sentiment_counts = {
            'Positive': sentiments.count('Positive') / total * 100,
            'Neutral': sentiments.count('Neutral') / total * 100,
            'Negative': sentiments.count('Negative') / total * 100
        }

        print("\nOverall Sentiment Analysis:")
        for sentiment, percentage in sentiment_counts.items():
            print(f"{sentiment}: {percentage:.1f}%")