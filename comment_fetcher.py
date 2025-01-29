import praw
import pandas as pd
from sql3 import db_upsert
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Initialize the Reddit instance
reddit = praw.Reddit(
    client_id = os.getenv("REDDIT_CLIENT_ID"),
    client_secret = os.getenv("REDDIT_CLIENT_SECRET"),
    username = os.getenv("REDDIT_USERNAME"),
    password = os.getenv("REDDIT_PASSWORD"),
    user_agent = os.getenv("REDDIT_USER_AGENT")
)

# Define the Reddit post URL
url = 'https://www.reddit.com/r/gaming/comments/1ibol3z/sony_reportedly_developing_new_god_of_war_game/'

# Retrieve the submission object
post = reddit.submission(url=url)

# Replace all MoreComments objects to get all comments
post.comments.replace_more(limit=None)

# Prepare a list of dictionaries for storing comment data
comments_data = []
for comment in post.comments.list():
    comments_data.append({
        'website': 'Reddit',
        'specific_url': url,
        'comment': comment.body
    })

# Convert to a DataFrame
comments_df = pd.DataFrame(comments_data)

# Database path and table name
db_path = "comments_database.sqlite"
table_name = "Comments"
primary_keys = ["website", "specific_url", "comment"]

# Upsert data into the database
db_upsert(comments_df, db_path, table_name, primary_keys)

print(f"Data successfully upserted into {table_name} table in {db_path}.")
