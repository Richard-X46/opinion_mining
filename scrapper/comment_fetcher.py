import praw
import pandas as pd
from dotenv import load_dotenv
import os
import psycopg2
from psycopg2 import sql, extras

# Load environment variables from .env file
load_dotenv()

# Initialize the Reddit instance
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    username=os.getenv("REDDIT_USERNAME"),
    password=os.getenv("REDDIT_PASSWORD"),
    user_agent=os.getenv("REDDIT_USER_AGENT"),
)

# Function to determine comment level
def get_comment_level(comment):
    level = 1
    parent = comment.parent_id
    while parent.startswith("t1_"):  # If parent is a comment (t1_), increase level
        level += 1
        parent = reddit.comment(parent[3:]).parent_id
    return level

def connect_to_db(host, port, database, user, password):
    try:
        conn = psycopg2.connect(
            host=host, port=port, database=database, user=user, password=password
        )
        print("Connection successful")
        return conn
    except Exception as error:
        print(f"Error: {error}")
        return None

def store_comments_for_url(url):
    """Store comments from a given URL in the database"""
    website_id = "reddit.com"
    
    # Retrieve the submission object
    post = reddit.submission(url=url)
    
    # Replace all MoreComments objects to get comments
    post.comments.replace_more(limit=0)
    
    # Prepare comments data
    comments_data = []
    for comment in list(post.comments)[:10]:  # Limit to 10 comments
        comments_data.append({
            "website_id": website_id,
            "specific_url": url,
            "comment": comment.body,
            "comment_level": get_comment_level(comment)
        })
    
    # Convert to DataFrame
    comments_df = pd.DataFrame(comments_data)
    
    # Connect and store in database
    conn = connect_to_db(
        os.getenv("HOST"),
        os.getenv("PORT"),
        os.getenv("DATABASE"),
        os.getenv("USER"),
        os.getenv("PASSWORD")
    )
    
    if conn:
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
        except Exception as error:
            print(f"Error: {error}")
            raise error
        finally:
            conn.close()

# Remove the hardcoded URL and direct database operations from the main part
if __name__ == "__main__":
    pass