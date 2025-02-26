import praw
import pandas as pd
from dotenv import load_dotenv
import os
from ls_psql import connect_to_db, upsert_comments

# Load environment variables
load_dotenv()

# Initialize the Reddit instance
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    username=os.getenv("REDDIT_USERNAME"),
    password=os.getenv("REDDIT_PASSWORD"),
    user_agent=os.getenv("REDDIT_USER_AGENT"),
)

# Define the Reddit post URL
url = "https://www.reddit.com/r/gaming/comments/1ibol3z/sony_reportedly_developing_new_god_of_war_game/"

# Retrieve the submission object
post = reddit.submission(url=url)

# Replace all MoreComments objects to get all comments
post.comments.replace_more(limit=None)

def get_comment_level(comment):
    """
    Determines the comment level based on its depth.
    Level 1 = main comment, Level 2 = subcomment, Level 3 = sub-subcomment, etc.
    """
    return comment.depth + 1

# Prepare a list of dictionaries for storing comment data
comments_data = []
for comment in post.comments.list():
    comments_data.append(
        {
            "website_id": "reddit.com",
            "specific_url": url,
            "comment": comment.body,
            "comment_level": get_comment_level(comment),
        }
    )

# Convert to a DataFrame
comments_df = pd.DataFrame(comments_data)

# Connect to the database and insert comments
conn = connect_to_db(
    os.getenv("HOST"), os.getenv("PORT"), os.getenv("DATABASE"), os.getenv("USER"), os.getenv("PASSWORD")
)

if conn:
    upsert_comments(conn, comments_df)
    conn.close()


# Test db connection
def test_database_contents():
    host = os.getenv("HOST")
    port = os.getenv("PORT")
    database = os.getenv("DATABASE")
    user = os.getenv("USER")
    password = os.getenv("PASSWORD")
    test_conn = connect_to_db(host, port, database, user, password)
    if test_conn:
        try:
            cur = test_conn.cursor()
            # Get total count
            cur.execute("SELECT COUNT(*) FROM Comments;")
            total_count = cur.fetchone()[0]
            print(f"\nTotal comments in database: {total_count}")

            # Get sample of comments
            cur.execute("""
                SELECT comment_id, website_id, LEFT(comment, 50) as comment_preview, 
                       comment_level 
                FROM Comments 
                LIMIT 5;
            """)
            rows = cur.fetchall()
            print("\nSample of comments:")
            print("ID | Website | Comment Preview | Level")
            print("-" * 70)
            for row in rows:
                print(f"{row[0]} | {row[1]} | {row[2]}... | {row[3]}")

            cur.close()
        except Exception as error:
            print(f"Error testing database: {error}")
        finally:
            test_conn.close()

# Run the test
if __name__ == "__main__":
    test_database_contents()