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
from loguru import logger
# Load environment variables
load_dotenv()
client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)

#Check
#print("Using Groq API base:", openai.api_base)
#print("Key prefix:", openai.api_key[:5])

# logger setup
logger.add(
    sys.stderr,
    level="INFO",
    format="{time} {level} {message}"
)

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
    formatted_comments = []
    
    for i, row in comments_df.iterrows():
        try:
            # Check if we have a sentiment result for this index
            if i < len(sentiment_results):
                formatted_comments.append({
                    'text': row['comment'],
                    'level': row['comment_level'],
                    'sentiment': sentiment_results[i]['sentiment_label']
                })
            else:
                # If we don't have a sentiment result, use neutral
                logger.warning(f"No sentiment result for comment {i}, using neutral")
                formatted_comments.append({
                    'text': row['comment'],
                    'level': row['comment_level'],
                    'sentiment': 1  # NEUTRAL
                })
        except Exception as e:
            logger.error(f"Error formatting comment {i}: {e}")
            formatted_comments.append({
                'text': row['comment'],
                'level': row.get('comment_level', 0),
                'sentiment': 1  # NEUTRAL
            })
    
    return formatted_comments

def analyze_emotions_with_sentiment(comments_text, sentiment_stats):
    """Use LLM to analyze emotions based on sentiment distribution"""
    # Truncate comments_text to avoid token limit errors
    truncated_text = comments_text[:15000] if len(comments_text) > 15000 else comments_text
    
    # Check if high negative or positive
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
    {truncated_text}
    
    Return only a comma-separated list of keywords, with no additional text.
    Each keyword should optionally include a sentiment qualifier if relevant.
    """

    try:
        logger.info(f"Sending request to analyze emotions with sentiment - comment length: {len(comments_text)}")
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "Identify emotions aligned with sentiment distribution."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            temperature=0.3
        )
        
        # Add defensive check in case response has no choices
        if not response.choices:
            logger.error("Emotion analysis returned no choices")
            return ["Unable to analyze emotions"]
            
        result = response.choices[0].message.content.strip()
        logger.info(f"Raw emotion analysis result: {result}")
        
        # Handle empty result
        if not result:
            logger.error("Emotion analysis returned empty result")
            return ["Unable to analyze emotions"]
            
        # Sanitize result to make sure we're only getting a comma-separated list
        if ":" in result or "\n" in result:
            # Extract only what appears to be the actual list
            parts = result.split(":")
            if len(parts) > 1:
                result = parts[-1].strip()
            else:
                # If splitting by : doesn't work, try to remove any headings/prefixes
                result = result.split("\n")[-1].strip()
                
        # Another defensive check for result after sanitization
        if not result:
            logger.error("Emotion analysis result was empty after sanitization")
            return ["Unable to analyze emotions"]
            
        # Split by comma and filter out empty entries
        emotions = [e.strip() for e in result.split(',') if e.strip()]
        
        # If we still have no emotions, return a default
        if not emotions:
            logger.error("No emotions found after splitting")
            return ["Unable to analyze emotions"]
            
        return emotions
        
    except Exception as e:
        logger.error(f"Emotion analysis error: {e}")
        return ["Unable to analyze emotions"]



def generate_cohesive_summary(comments_df, keywords, sentiment_stats, emotions):
    """Use GPT to generate a cohesive summary based on analysis"""
    # Log the comment count and length of comments_text
    logger.info(f"Number of comments: {len(comments_df)}")

    # Reduce text size by taking only top comments
    if len(comments_df) > 20:
        top_comments = comments_df.sort_values('comment_score', ascending=False).head(20)
        comments_text = " ".join(top_comments['comment'].tolist())
    else:
        comments_text = " ".join(comments_df['comment'].tolist())
    
    # Further truncate if needed
    if len(comments_text) > 10000:
        comments_text = comments_text[:10000] + "..."
    
    logger.info(f"Length of comments text: {len(comments_text)}")
    
    # Add defensive check for empty or None sentiment_stats
    if not sentiment_stats:
        sentiment_stats = {'Neutral': 100}
    
    dominant_sentiment = max(sentiment_stats.items(), key=lambda x: x[1])[0]
    
    # Add defensive check for empty or None emotions
    if not emotions or not isinstance(emotions, list):
        emotions = ["Unable to analyze emotions"]
    
    emotion_list = ", ".join(emotions)
    
    # Add defensive check for empty or None keywords
    if not keywords or not isinstance(keywords, list):
        keywords = ["No keywords available"]
    
    # Safely get up to 5 keywords, or fewer if less are available
    keyword_list = ", ".join(keywords[:5] if len(keywords) >= 5 else keywords)

    prompt = f"""
    Summarize this Reddit discussion in 2–3 sentences, incorporating:
    - Dominant sentiment: {dominant_sentiment} ({sentiment_stats[dominant_sentiment]:.1f}%)
    - Emotions: {emotion_list}
    - Keywords: {keyword_list}

    Text:
    {comments_text}
    """

    try:
        logger.info("Sending summary generation request")
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "Summarize discussions using emotional and sentiment analysis."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.3
        )
        
        if not response.choices:
            logger.error("Summary generation returned no choices")
            return "Unable to generate summary"
            
        summary = response.choices[0].message.content.strip()
        logger.info("Summary generation successful")
        return summary
        
    except Exception as e:
        logger.error(f"Summary generation error: {e}")
        return "Unable to generate summary"

# ------------------- Main Route -------------------

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        logger.info("Received POST request")
        search_query = request.form.get("search_query", "").strip()
        logger.info(f"Search query: {search_query}")

        if len(search_query) <= 8:
            flash('Please enter a more descriptive query.', 'error')
            return render_template('index.html')

        try:
            # Step 1: DB connection
            logger.info("Connecting to database")
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
            posts = cf.get_post_details(search_query, post_limit=3)
            if not posts:
                raise Exception("No Reddit posts found for query.")

            posts = sorted(posts, key=lambda x: x['post_score'], reverse=True)
            posts_df = pd.DataFrame([{k: v for k, v in p.items() if k != 'post_comments'} for p in posts])
            posts_df['post_created_utc'] = pd.to_datetime(posts_df['post_created_utc'], unit='s')
            posts_df['search_query'] = search_query
            logger.info(f"Fetched {len(posts_df)} posts")
            # Store posts in DB

            upsert_table(conn, posts_df, "posts", ["post_id"])
            logger.info("Posts stored in database")
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
            logger.info(f"Fetched and stored {len(comments_df)} comments")
            # Step 4: Load stored comments for query
            query = """
                SELECT * FROM comments_new
                LEFT JOIN posts ON comments_new.post_id = posts.post_id
                WHERE posts.search_query = %s AND comment_score > 1
                ORDER BY comment_score DESC
            """
            comments_df = query_db(conn, query, (search_query,))

            # downsize the comments_df 
            # comments_df = comments_df.sample(frac=0.5, random_state=1)  # downsize to 50% of the original

            if comments_df is None or comments_df.empty:
                logger.warning("No comments retrieved from DB.")
                raise Exception("No comments retrieved from DB.")

            all_text = " ".join(comments_df['comment'].tolist())

            # Step 5: Sentiment + Emotion + Keywords
            logger.info("Analyzing sentiment and emotions")
            analyzer = SentimentAnalyzer()
            comments_list = comments_df['comment'].tolist()
            
            # Process all comments in a single batch
            sentiment_labels = analyzer.analyze_sentiment_batch(comments_list)
            
            # Map sentiments to numerical values
            label_map = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
            sentiment_results = []
            for i, (comment, sentiment) in enumerate(zip(comments_list, sentiment_labels)):
                try:
                    # Check if sentiment is valid
                    if isinstance(sentiment, str) and sentiment in label_map:
                        sentiment_label = label_map[sentiment]
                    else:
                        logger.warning(f"Invalid sentiment value: '{sentiment}', defaulting to NEUTRAL")
                        sentiment_label = 1  # Default to neutral
                        
                    sentiment_results.append({
                        'comment': comment,
                        'sentiment_label': sentiment_label
                    })
                except Exception as e:
                    logger.error(f"Error processing sentiment for comment {i}: {e}")
                    # Add a default neutral sentiment
                    sentiment_results.append({
                        'comment': comment,
                        'sentiment_label': 1  # NEUTRAL
                    })
            total = len(sentiment_results)
            sentiments = [r['sentiment_label'] for r in sentiment_results]
            sentiment_stats = {
                'Positive': sentiments.count(2) / total * 100,
                'Neutral': sentiments.count(1) / total * 100,
                'Negative': sentiments.count(0) / total * 100
            }
            logger.info(f"Sentiment analysis completed: {sentiment_stats}")
            # Step 6: Generate summary and keywords
            logger.info("Generating summary and keywords")
            try:
                logger.info("Extracting keywords with sentiment")
                keywords = extract_keywords_with_sentiment(all_text, max(sentiment_stats, key=sentiment_stats.get))
                
                # Ensure we have valid keywords
                if not keywords or not isinstance(keywords, list):
                    logger.warning("No valid keywords returned, using default")
                    keywords = ["No keywords available"]
                    
                logger.info(f"Extracted {len(keywords)} keywords")
            except Exception as e:
                logger.error(f"Keyword extraction error: {e}")
                keywords = ["No keywords available"]


            emotions = analyze_emotions_with_sentiment(all_text, sentiment_stats)
            summary = generate_cohesive_summary(comments_df, keywords, sentiment_stats, emotions)
            rating = calculate_rating(sentiment_results)
            comments = format_comments_with_sentiment(comments_df, sentiment_results)
            logger.info("Summary and keywords generated and rendered to template") 
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