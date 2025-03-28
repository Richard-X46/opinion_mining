import torch
from transformers import BertTokenizer, BertForSequenceClassification
import psycopg2
import pandas as pd
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

class SentimentAnalyzer:
    def __init__(self):
        # Load pre-trained model and tokenizer for sentiment analysis
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=3  # 3 labels: negative (0), neutral (1), positive (2)
        )
        
    def analyze_sentiment(self, text):
        # Prepare the input
        
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=512
        )
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=1)
            predicted_label = torch.argmax(predictions, dim=1).item()
        
        # Map numerical labels to sentiment categories
        sentiment_map = {
            0: "Negative",
            1: "Neutral",
            2: "Positive"
        }
        
        return sentiment_map[predicted_label]

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
    # Initialize sentiment analyzer
    analyzer = SentimentAnalyzer()
    
    # Fetch comments from database
    df = fetch_comments_from_db()
    
    # Analyze sentiment for each comment
    results = []
    if not df.empty:
        for index, row in df.iterrows():
            sentiment = analyzer.analyze_sentiment(row['comment'])
            results.append({
                'comment': row['comment'],
                'sentiment': sentiment
            })
            print(f"Comment: {row['comment'][:100]}... \nSentiment: {sentiment}\n")
    
    return results

def analyze_unified_sentiment(comments_df):
    """
    Perform unified sentiment analysis that coordinates all sentiment-related outputs.
    Returns a dictionary with coordinated sentiment information.
    """
    analyzer = SentimentAnalyzer()
    
    # Analyze each comment and store detailed results
    detailed_results = []
    for _, row in comments_df.iterrows():
        comment_text = row['comment']
        
        # Get BERT sentiment
        bert_sentiment = analyzer.analyze_sentiment(comment_text)
        sentiment_label_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
        sentiment_score = sentiment_label_map[bert_sentiment]
        
        # Store the detailed analysis
        detailed_results.append({
            'text': comment_text,
            'bert_sentiment': bert_sentiment,
            'sentiment_score': sentiment_score
        })
    
    # Calculate overall sentiment distribution
    total = len(detailed_results)
    sentiments = [r['sentiment_score'] for r in detailed_results]
    
    sentiment_stats = {
        'Positive': sentiments.count(2) / total * 100,
        'Neutral': sentiments.count(1) / total * 100,
        'Negative': sentiments.count(0) / total * 100
    }
    
    # Determine dominant sentiment
    dominant_sentiment = max(sentiment_stats.items(), key=lambda x: x[1])[0]
    
    return {
        'detailed_results': detailed_results,
        'sentiment_stats': sentiment_stats,
        'dominant_sentiment': dominant_sentiment
    }

if __name__ == "__main__":
    results = analyze_comments()
    
    if results:
        # Calculate overall sentiment statistics
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