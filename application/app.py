from flask import Flask, render_template, request, flash
from flask_limiter.util import get_remote_address
from flask_limiter import Limiter
from flask_wtf.csrf import CSRFProtect
from flask_talisman import Talisman
import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import scrapper.comment_fetcher as cf
from scrapper.keyword_extract import extract_keywords_gpt, extract_keywords_with_sentiment
from scrapper.sentiment_analysis import SentimentAnalyzer
from scrapper.ls_psql import connect_to_db, upsert_table, query_db
from dotenv import load_dotenv
import praw.exceptions
import sys
from pathlib import Path
# from transformers import pipeline
from urllib.parse import urlparse
import re
import praw
import pandas as pd
from psycopg2 import sql, extras
import openai
from scrapper.rating_system import calculate_rating

def generate_summary(comments_df, keywords, sentiment_stats):
    # Combine comments into a single text
    comments_text = " ".join(comments_df['comment'].tolist())

    # Prepare the prompt for the LLM
    prompt = f"""
    The following are comments from a Reddit discussion:
    {comments_text}

    The dominant sentiment is {max(sentiment_stats.items(), key=lambda x: x[1])[0].lower()}.
    The key topics discussed are: {', '.join(keywords[:3])}.

    Summarize the discussion in 2-3 sentences, focusing on the key points and sentiment.
    """

    # Call the OpenAI API
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes Reddit discussions."},
                {"role": "user", "content": prompt}
            ],
        max_tokens=100, 
        temperature=0.3, 
    )

    # Extract and return the summary
    summary = response.choices[0].message.content
    return summary

def generate_cohesive_summary(comments_df, keywords, sentiment_stats, emotions):
    """Generate a summary that ties together all sentiment-related components"""
    
    # Combine comments into a single text
    comments_text = " ".join(comments_df['comment'].tolist())
    
    # Determine dominant sentiment and emotions
    dominant_sentiment = max(sentiment_stats.items(), key=lambda x: x[1])[0]
    emotion_list = ", ".join(emotions)
    
    prompt = f"""
    Summarize this Reddit discussion considering ALL of the following analysis:
    
    1. The dominant sentiment is {dominant_sentiment} ({sentiment_stats[dominant_sentiment]:.1f}%)
    2. Key emotions detected: {emotion_list}
    3. Important keywords: {', '.join(keywords[:5])}
    
    Provide a 2-3 sentence summary that weaves together the sentiment, emotions, and key topics 
    in a cohesive way. The summary should reflect the same emotional tone shown in the analysis.
    
    Discussion text:
    {comments_text}
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a discussion summarizer that ensures consistency between sentiment, emotions, and content."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.3 
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in summary generation: {e}")
        return "Unable to generate summary"

# Add this function after generate_summary()
def analyze_emotions(comments_text):
    """Analyze emotions present in the comments using GPT-3.5"""
    prompt = f"""
    Analyze the emotional content in these Reddit comments and identify the main emotions present:
    {comments_text}

    List only the top 3-4 emotions that are most strongly expressed, from this set:
    Frustration, Satisfaction, Joy, Sadness, Anger, Surprise, Disgust, Fear, Interest, Excitement

    Return just the emotions as a comma-separated list, ordered by strength of presence.
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an emotion analysis assistant. Identify emotions from text concisely."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            temperature=0.3
        )
        emotions = response.choices[0].message.content.strip()
        return [e.strip() for e in emotions.split(',')]
    except Exception as e:
        print(f"Error in emotion analysis: {e}")
        return ["Unable to analyze emotions"]

# Add this function after analyze_emotions()
def analyze_emotions_with_sentiment(comments_text, sentiment_stats):
    """Analyze emotions while considering sentiment distribution"""
    
    # Map sentiment ranges to emotion categories
    high_negative = sentiment_stats['Negative'] > 40
    high_positive = sentiment_stats['Positive'] > 40
    
    prompt = f"""
    Analyze the emotional content in these Reddit comments, considering that the sentiment analysis shows:
    Negative: {sentiment_stats['Negative']}%
    Neutral: {sentiment_stats['Neutral']}%
    Positive: {sentiment_stats['Positive']}%

    List only the top 3-4 emotions that are most strongly expressed, ensuring they align with these sentiment percentages.
    Choose from: Frustration, Satisfaction, Joy, Sadness, Anger, Surprise, Disgust, Fear, Interest, Excitement

    The emotions must reflect the sentiment distribution above.
    If negative sentiment is high ({high_negative}), include more negative emotions.
    If positive sentiment is high ({high_positive}), include more positive emotions.
    
    Return just the emotions as a comma-separated list, ordered by strength of presence.
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an emotion analysis assistant that ensures consistency with sentiment analysis."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            temperature=0.3
        )
        emotions = response.choices[0].message.content.strip()
        return [e.strip() for e in emotions.split(',')]
    except Exception as e:
        print(f"Error in emotion analysis: {e}")
        return ["Unable to analyze emotions"]

# Add this function before store_comments_for_url
def get_comment_level(comment):
    """Get the nesting level of a comment"""
    level = 0
    parent = comment.parent()
    while not isinstance(parent, praw.models.Submission):
        level += 1
        parent = parent.parent()
    return level

def format_comments_with_sentiment(comments_df, sentiment_results):
    """Format comments with their sentiment information"""
    formatted_comments = []
    for i, row in comments_df.iterrows():
        formatted_comments.append({
            'text': row['comment'],
            'level': row['comment_level'],
            'sentiment': sentiment_results[i]['sentiment_label']
        })
    return formatted_comments

def store_comments_for_url(url):
    """Store comments from a given URL in the database"""
    try:
        website_id = "reddit.com"
        
        # Retrieve the submission object
        post = reddit.submission(url=url)
        if not post or not hasattr(post, 'comments'):
            raise Exception("Could not fetch Reddit post or comments")
        
        # Replace all MoreComments objects to get comments
        post.comments.replace_more(limit=0)
        
        if not post.comments:
            raise Exception("No comments found for this post")
        
        # Prepare comments data
        comments_data = []
        for comment in list(post.comments)[:10]:  # Limit to 10 comments
            if hasattr(comment, 'body'):  # Verify comment has body attribute
                comments_data.append({
                    "website_id": website_id,
                    "specific_url": url,
                    "comment": comment.body,
                    "comment_level": get_comment_level(comment)
                })
        
        if not comments_data:
            raise Exception("No valid comments found")
        
        # Convert to DataFrame
        comments_df = pd.DataFrame(comments_data)
        
        # Connect and store in database
        conn = connect_to_db(
            os.getenv("HOST"),
            os.getenv("PORT"),
            os.getenv("DATABASE"),
            os.getenv("DB_USER"),
            os.getenv("PASSWORD")
        )
        
        if not conn:
            raise Exception("Database connection failed")
        
        try:
            cur = conn.cursor()
            columns = list(comments_df.columns)
            values = [tuple(x) for x in comments_df.to_numpy()]
            
            insert_sql = sql.SQL(
                "INSERT INTO Comments ({}) VALUES %s ON CONFLICT (specific_url, comment) DO NOTHING"
            ).format(sql.SQL(", ").join(map(sql.Identifier, columns)))
            
            extras.execute_values(cur, insert_sql, values)
            conn.commit()
            cur.close()
            return True
        except Exception as error:
            print(f"Database error: {error}")
            raise error
        finally:
            conn.close()
    except Exception as e:
        print(f"Error in store_comments_for_url: {str(e)}")
        raise e

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'default-secret-key')
csrf = CSRFProtect(app)
talisman = Talisman(
    app,
    content_security_policy={
        "default-src": "'self'",
        "script-src": "'self' 'unsafe-inline'",
        "style-src": "'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com",
    },force_https=True
)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "1000 per hour"],
    storage_uri="memory://",
)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        search_query = request.form["search_query"]

        # search query validation 
        if len(search_query) <= 8:
            flash('Please enter a valid search query.', 'error')
            return render_template('index.html')
        
        try:
            # Connect to database first to verify connection
            conn = connect_to_db(
                os.getenv("HOST"),
                os.getenv("PORT"),
                os.getenv("DATABASE"),
                os.getenv("DB_USER"),
                os.getenv("PASSWORD")
            )
            
            if conn is None:
                raise Exception("Could not connect to database. Please check database configuration.")

            s = cf.get_post_details(search_query, post_limit=5)
            if not s:
                raise Exception("No posts found for the search query.")

            s = sorted(s, key=lambda x: x['post_score'], reverse=True)
            posts_data = [{k:v for k,v in i.items() if k not in ['post_comments']} for i in s]
            
            # Process posts
            posts_df = pd.DataFrame(posts_data)
            posts_df['post_created_utc'] = pd.to_datetime(posts_df['post_created_utc'], unit='s')
            posts_df['search_query'] = search_query
            upsert_table(conn, posts_df, "posts", ["post_id"])

            # Process comments with better error handling
            comments_data = []
            for post in s:
                post_comments = cf.get_post_comments(
                    post['post_comments'],
                    search_query=search_query,
                    post_title=post['post_title']
                )
                if post_comments:  # Only add if we got comments
                    comments_data.extend(post_comments)

            if not comments_data:
                raise Exception("No relevant comments found for the search query.")

            comments_df = pd.DataFrame(comments_data)
            comments_df['comment_created_utc'] = pd.to_datetime(comments_df['comment_created_utc'], unit='s')
            
            # Store filtered comments
            upsert_table(conn, comments_df, "comments_new", ["comment_id"])

            # Fetch comments from database that pertains to the search query
            query = """ 
                select * 
                from comments_new
                left join posts on 
                comments_new.post_id = posts.post_id 
                where posts.search_query = %s and comment_score > 1
                order by comment_score desc
            """
            
            comments_df = query_db(conn, query, (search_query,))

            if comments_df is None or comments_df.empty:
                raise Exception("No comments were retrieved from the database.")
                        
            # Process comments from database
            comments = []
            all_text = " ".join(comments_df['comment'].tolist())
            
            # Unified sentiment analysis
            analyzer = SentimentAnalyzer()
            sentiment_results = []
            for _, row in comments_df.iterrows():
                comment_text = row['comment']
                sentiment = analyzer.analyze_sentiment(comment_text)
                sentiment_label_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
                sentiment_label = sentiment_label_map[sentiment]
                sentiment_results.append({
                    'comment': comment_text,
                    'sentiment_label': sentiment_label
                })

            # Calculate sentiment statistics
            total = len(sentiment_results)
            sentiments = [r['sentiment_label'] for r in sentiment_results]
            sentiment_stats = {
                'Positive': sentiments.count(2) / total * 100,
                'Neutral': sentiments.count(1) / total * 100,
                'Negative': sentiments.count(0) / total * 100
            }
            
            # Determine dominant sentiment
            dominant_sentiment = max(sentiment_stats.items(), key=lambda x: x[1])[0]

            # Extract sentiment-aware keywords
            keywords = extract_keywords_with_sentiment(all_text, dominant_sentiment)
            
            # Analyze emotions with sentiment alignment
            emotions = analyze_emotions_with_sentiment(all_text, sentiment_stats)
            
            # Generate cohesive summary
            summary = generate_cohesive_summary(comments_df, keywords, sentiment_stats, emotions)

            # Calculate website rating (existing code)
            rating = calculate_rating(sentiment_results)

            # Store comment structure for display
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
            flash('Invalid Reddit URL provided. Please check the URL and try again.', 'error')
            return render_template('index.html')
        except Exception as e:
            flash(f'An error occurred: {str(e)}', 'error')
            return render_template('index.html')
        finally:
            if 'conn' in locals() and conn:
                conn.close()
    
    return render_template('index.html')

if __name__ == '__main__':
    debug_mode = os.environ.get("FLASK_ENV") != "production"
    app.run(host="0.0.0.0", debug=debug_mode, port=5001)