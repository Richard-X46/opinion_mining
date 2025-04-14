from flask import Flask, render_template, request, flash
from flask_limiter.util import get_remote_address
from flask_limiter import Limiter
from flask_wtf.csrf import CSRFProtect
from flask_talisman import Talisman
from dotenv import load_dotenv
import os
import sys
from pathlib import Path
import pandas as pd
import praw
import openai
from psycopg2 import sql, extras

# Add scrapper module path
sys.path.append(str(Path(__file__).parent.parent))

# Project modules
import scrapper.comment_fetcher as cf
from scrapper.keyword_extract import extract_keywords_with_sentiment
from scrapper.sentiment_analysis import SentimentAnalyzer
from scrapper.ls_psql import connect_to_db, upsert_table, query_db
from scrapper.rating_system import calculate_rating

# Load environment variables
load_dotenv()
client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)

#Check
#print("Using Groq API base:", openai.api_base)
#print("Key prefix:", openai.api_key[:5])

# Flask app setup
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'default-secret-key')

csrf = CSRFProtect(app)

talisman = Talisman(
    app,
    content_security_policy={
        "default-src": "'self'",
        "script-src": "'self' 'unsafe-inline'",
        "style-src": "'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com",
    },
    force_https=False
)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "1000 per hour"],
    storage_uri="memory://",
)

# ------------------- Helper Functions -------------------

def get_comment_level(comment):
    """Get the nesting level of a Reddit comment"""
    level = 0
    parent = comment.parent()
    while not isinstance(parent, praw.models.Submission):
        level += 1
        parent = parent.parent()
    return level

def format_comments_with_sentiment(comments_df, sentiment_results):
    """Format comments with sentiment labels and levels"""
    return [
        {
            'text': row['comment'],
            'level': row['comment_level'],
            'sentiment': sentiment_results[i]['sentiment_label']
        }
        for i, row in comments_df.iterrows()
    ]

def analyze_emotions_with_sentiment(comments_text, sentiment_stats):
    """Use LLM to analyze emotions based on sentiment distribution"""
    high_negative = sentiment_stats['Negative'] > 40
    high_positive = sentiment_stats['Positive'] > 40

    prompt = f"""
    Analyze these Reddit comments considering sentiment:
    Negative: {sentiment_stats['Negative']}%
    Neutral: {sentiment_stats['Neutral']}%
    Positive: {sentiment_stats['Positive']}%

    List top emotions (e.g., Joy, Anger, Sadness, Excitement) reflecting the sentiment distribution.
    More negative emotions if negative is high, and vice versa.
    
    Comments:
    {comments_text}
    
    Return only a comma-separated list of keywords, with no additional text.
    Each keyword should optionally include a sentiment qualifier if relevant.
    """

    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "Identify emotions aligned with sentiment distribution."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            temperature=0.3
        )
        return [e.strip() for e in response.choices[0].message.content.strip().split(',')][1:]
    except Exception as e:
        print(f"Emotion analysis error: {e}")
        return ["Unable to analyze emotions"]

def generate_cohesive_summary(comments_df, keywords, sentiment_stats, emotions):
    """Use GPT to generate a cohesive summary based on analysis"""
    comments_text = " ".join(comments_df['comment'].tolist())
    dominant_sentiment = max(sentiment_stats.items(), key=lambda x: x[1])[0]
    emotion_list = ", ".join(emotions)

    prompt = f"""
    Summarize this Reddit discussion in 2–3 sentences, incorporating:
    - Dominant sentiment: {dominant_sentiment} ({sentiment_stats[dominant_sentiment]:.1f}%)
    - Emotions: {emotion_list}
    - Keywords: {', '.join(keywords[:5])}

    Text:
    {comments_text}
    """

    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "Summarize discussions using emotional and sentiment analysis."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Summary generation error: {e}")
        return "Unable to generate summary"

# ------------------- Main Route -------------------

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        search_query = request.form.get("search_query", "").strip()

        if len(search_query) <= 8:
            flash('Please enter a more descriptive query.', 'error')
            return render_template('index.html')

        try:
            # Step 1: DB connection
            conn = connect_to_db(
                os.getenv("HOST"),
                os.getenv("PORT"),
                os.getenv("DATABASE"),
                os.getenv("DB_USER"),
                os.getenv("PASSWORD")
            )
            if conn is None:
                raise Exception("Database connection failed.")

            # Step 2: Reddit post fetch + store
            posts = cf.get_post_details(search_query, post_limit=5)
            if not posts:
                raise Exception("No Reddit posts found for query.")

            posts = sorted(posts, key=lambda x: x['post_score'], reverse=True)
            posts_df = pd.DataFrame([{k: v for k, v in p.items() if k != 'post_comments'} for p in posts])
            posts_df['post_created_utc'] = pd.to_datetime(posts_df['post_created_utc'], unit='s')
            posts_df['search_query'] = search_query
            upsert_table(conn, posts_df, "posts", ["post_id"])

            # Step 3: Comment fetch + store
            comments_data = []
            for post in posts:
                post_comments = cf.get_post_comments(
                    post['post_comments'], search_query, post['post_title']
                )
                if post_comments:
                    comments_data.extend(post_comments)

            if not comments_data:
                raise Exception("No relevant comments found.")

            comments_df = pd.DataFrame(comments_data)
            comments_df['comment_created_utc'] = pd.to_datetime(comments_df['comment_created_utc'], unit='s')
            upsert_table(conn, comments_df, "comments_new", ["comment_id"])

            # Step 4: Load stored comments for query
            query = """
                SELECT * FROM comments_new
                LEFT JOIN posts ON comments_new.post_id = posts.post_id
                WHERE posts.search_query = %s AND comment_score > 1
                ORDER BY comment_score DESC
            """
            comments_df = query_db(conn, query, (search_query,))
            if comments_df is None or comments_df.empty:
                raise Exception("No comments retrieved from DB.")

            all_text = " ".join(comments_df['comment'].tolist())

            # Step 5: Sentiment + Emotion + Keywords
            analyzer = SentimentAnalyzer()
            sentiment_results = []
            for _, row in comments_df.iterrows():
                sentiment = analyzer.analyze_sentiment(row['comment'])
                label_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
                sentiment_results.append({
                    'comment': row['comment'],
                    'sentiment_label': label_map[sentiment]
                })

            total = len(sentiment_results)
            sentiments = [r['sentiment_label'] for r in sentiment_results]
            sentiment_stats = {
                'Positive': sentiments.count(2) / total * 100,
                'Neutral': sentiments.count(1) / total * 100,
                'Negative': sentiments.count(0) / total * 100
            }

            keywords = extract_keywords_with_sentiment(all_text, max(sentiment_stats, key=sentiment_stats.get))
            emotions = analyze_emotions_with_sentiment(all_text, sentiment_stats)
            summary = generate_cohesive_summary(comments_df, keywords, sentiment_stats, emotions)
            rating = calculate_rating(sentiment_results)
            comments = format_comments_with_sentiment(comments_df, sentiment_results)

            return render_template('results.html',
                                   comments=comments,
                                   keywords=keywords,
                                   sentiment_stats=sentiment_stats,
                                   summary=summary,
                                   search_query=search_query,
                                   rating=rating,
                                   emotions=emotions)

        except praw.exceptions.InvalidURL:
            flash('Invalid Reddit URL provided.', 'error')
        except Exception as e:
            flash(f'An error occurred: {str(e)}', 'error')
        finally:
            if 'conn' in locals() and conn:
                conn.close()

    return render_template('index.html')

# ------------------- Run Server -------------------
if __name__ == '__main__':
    debug_mode = os.getenv("FLASK_ENV") != "production"
    app.run(host="0.0.0.0", port=5001, debug=debug_mode)