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

    def analyze_sentiment(self, text):
        """
        Analyze sentiment of a given comment using Groq's LLM.
        Returns one of: 'Positive', 'Neutral', 'Negative'
        """
        prompt = f"""
        Determine the sentiment of the following Reddit comment.
        Respond with one word only: Positive, Neutral, or Negative.

        Comment: \"{text.strip()}\"
        """

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a sentiment analysis assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1,
                temperature=0
            )
            result = response.choices[0].message.content.strip()
            return result if result in ["Positive", "Neutral", "Negative"] else "Neutral"
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            return "Neutral"

def fetch_comments_from_db():
    try:
        conn = psycopg2.connect(
            host=os.getenv("HOST"),
            port=os.getenv("PORT"),
            database=os.getenv("DATABASE"),
            user=os.getenv("USER"),
            password=os.getenv("PASSWORD")
        )
        query = "SELECT * FROM Comments"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        print(f"Database error: {e}")
        return pd.DataFrame()

def analyze_comments():
    analyzer = SentimentAnalyzer()
    df = fetch_comments_from_db()
    results = []

    if not df.empty:
        for index, row in df.iterrows():
            sentiment = analyzer.analyze_sentiment(row['comment'])
            results.append({
                'comment': row['comment'],
                'sentiment': sentiment
            })
            print(f"Comment: {row['comment'][:100]}...\nSentiment: {sentiment}\n")

    return results

def analyze_unified_sentiment(comments_df):
    analyzer = SentimentAnalyzer()
    detailed_results = []
    sentiment_label_map = {"Negative": 0, "Neutral": 1, "Positive": 2}

    for _, row in comments_df.iterrows():
        sentiment = analyzer.analyze_sentiment(row['comment'])
        sentiment_score = sentiment_label_map[sentiment]
        detailed_results.append({
            'text': row['comment'],
            'sentiment': sentiment,
            'sentiment_score': sentiment_score
        })

    total = len(detailed_results)
    sentiments = [r['sentiment_score'] for r in detailed_results]
    
    sentiment_stats = {
        'Positive': sentiments.count(2) / total * 100,
        'Neutral': sentiments.count(1) / total * 100,
        'Negative': sentiments.count(0) / total * 100
    }

    dominant_sentiment = max(sentiment_stats.items(), key=lambda x: x[1])[0]

    return {
        'detailed_results': detailed_results,
        'sentiment_stats': sentiment_stats,
        'dominant_sentiment': dominant_sentiment
    }

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