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
from scrapper.keyword_extract import extract_keywords_gpt
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
        max_tokens=100,  # Limit the response length
        temperature=0.7,  # Control creativity (0 = factual, 1 = creative)
    )

    # Extract and return the summary
    summary = response.choices[0].message.content
    return summary


# Add this function before store_comments_for_url
def get_comment_level(comment):
    """Get the nesting level of a comment"""
    level = 0
    parent = comment.parent()
    while not isinstance(parent, praw.models.Submission):
        level += 1
        parent = parent.parent()
    return level

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

            # --------- ////main from comment fetcher goes here to store the data in posts and comments_new
            s= cf.get_post_details(search_query,post_limit=5)
            s = sorted(s,key=lambda x: x['post_score'],reverse=True) # sort by score
            # removing comment objects from the post data
            posts_data = [{k:v for k,v in i.items() if k not in ['post_comments']} for i in s]
            # convert to dataframe and formating
            posts_df = pd.DataFrame(posts_data)
            posts_df['post_created_utc'] = pd.to_datetime(posts_df['post_created_utc'],unit='s')
            posts_df['search_query'] = search_query # add search query to the dataframe
            # upserting posts data
            upsert_table(conn, posts_df, "posts",["post_id"])

            # ---- /// getting comments based on post data
            comments_data = [cf.get_post_comments(i['post_comments']) for i in s]
            comments_df  = pd.concat([pd.DataFrame(x) for x in comments_data])
            # convert comment_created_utc to datetime
            comments_df['comment_created_utc'] = pd.to_datetime(comments_df['comment_created_utc'],unit='s')

            #upserting comments data
            upsert_table(conn, comments_df, "comments_new",["comment_id"])
            # --------- //// cf ends here


            # Fetch comments from database that pertains to the search query
            query = """ select * 
                        from comments_new
                        left join posts on 
                        comments_new.post_id = posts.post_id 
                        where posts.search_query = %s and comment_score > 1
                        order by comment_score desc
                        
                        """
            
            comments_df= query_db(conn, query, (search_query,))

            if comments_df is None or comments_df.empty:
                raise Exception("No comments were retrieved from the database.")
                        
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
            keywords = extract_keywords_gpt(all_text)
            
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
            
            # Generate summary using our custom function instead of transformer
            summary = generate_summary(comments_df, keywords, sentiment_stats)

            return render_template('results.html',
                                comments=comments,
                                keywords=keywords,
                                sentiment_stats=sentiment_stats,
                                summary=summary,
                                search_query=search_query)  # Add this line                    search_query=search_query)  # Add this line
            
        except praw.exceptions.InvalidURL:
            flash('Invalid Reddit URL provided. Please check the URL and try again.', 'error')
            return render_template('index.html')
        except Exception as e:
            flash(f'An error occurred: {str(e)}', 'error')
            return render_template('index.html')
    
    return render_template('index.html')

if __name__ == '__main__':
    debug_mode = os.environ.get("FLASK_ENV") != "production"
    app.run(host="0.0.0.0", debug=debug_mode, port=5001)


