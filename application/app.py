from flask import Flask, render_template, request, flash
from scrapper.comment_fetcher import reddit, store_comments_for_url
from scrapper.keyword_extract import extract_keywords
from scrapper.sentiment_analysis import SentimentAnalyzer
from scrapper.ls_psql import query_db, connect_to_db
from dotenv import load_dotenv
import os
import praw.exceptions
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'default-secret-key')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        url = request.form['url']
        
        try:
            # First store comments in database
            store_comments_for_url(url)
            
            # Connect to database
            conn = connect_to_db(
                os.getenv("HOST"),
                os.getenv("PORT"),
                os.getenv("DATABASE"),
                os.getenv("USER"),
                os.getenv("PASSWORD")
            )
            
            # Fetch comments from database
            query = f"SELECT comment, comment_level FROM Comments WHERE specific_url = '{url}' LIMIT 10"
            comments_df = query_db(conn, query)
            
            # Process comments from database
            comments = []
            all_text = ""
            for _, row in comments_df.iterrows():
                comments.append({
                    'text': row['comment'],
                    'level': row['comment_level']
                })
                all_text += row['comment'] + " "
            
            # Extract keywords
            keywords = extract_keywords(all_text, top_n=10)
            
            # Analyze sentiment
            analyzer = SentimentAnalyzer()
            sentiments = []
            for comment in comments:
                sentiment = analyzer.analyze_sentiment(comment['text'])
                sentiments.append(sentiment)
            
            # Calculate sentiment percentages
            total = len(sentiments)
            sentiment_stats = {
                'Positive': sentiments.count('Positive') / total * 100,
                'Neutral': sentiments.count('Neutral') / total * 100,
                'Negative': sentiments.count('Negative') / total * 100
            }
            
            return render_template('results.html',
                                comments=comments,
                                keywords=keywords,
                                sentiment_stats=sentiment_stats)
            
        except praw.exceptions.InvalidURL:
            flash('Invalid Reddit URL provided. Please check the URL and try again.', 'error')
            return render_template('index.html')
        except Exception as e:
            flash(f'An error occurred: {str(e)}', 'error')
            return render_template('index.html')
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)