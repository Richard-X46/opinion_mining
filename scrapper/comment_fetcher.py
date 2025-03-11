import praw
import pandas as pd
from dotenv import load_dotenv
import os
import psycopg2
from psycopg2 import sql, extras
from googlesearch import search
import logging
import scrapper.ls_psql as psql
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Initialize the Reddit instance
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    username=os.getenv("REDDIT_USERNAME"),
    password=os.getenv("REDDIT_PASSWORD"),
    user_agent=os.getenv("REDDIT_USER_AGENT"),)

def get_comment_level(comment):
    level = 1
    parent = comment.parent_id
    while parent.startswith("t1_"):  # If parent is a comment (t1_), increase level
        level += 1
        parent = reddit.comment(parent[3:]).parent_id
    return level

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

def search_reddit(keyword):
    """
    Search for a keyword on Google and return the URLs of the search results from reddit.com.
    """
    # Prepend "site:reddit.com " to the keyword to restrict results to reddit.com
    query = f"site:reddit.com {keyword}"
    search_results = []
    for url in search(query):
        search_results.append(url)
    return search_results

def get_post_details(search_query:str,post_limit: int):
    """
    Get post details for a search query and return them as a DataFrame

    """

    results = search_reddit(search_query)

    results = results[:post_limit]
    # log number of links
    logging.info(f"Found {len(results)} links")


    try:
        if len(results) > 0:

            # Prepare post data
            post_data = []


            for i in results:
                post = reddit.submission(url=i)
                post.comments.replace_more(limit=None)
                # fetch all comments

                post_data.append( {
                    "post_id": post.id,
                    "post_created_utc": post.created_utc,
                    "post_title": post.title,
                    "post_score": post.score,
                    "post_upvote_ratio": post.upvote_ratio,
                    "post_num_comments": post.num_comments,
                    "post_url": post.url,
                    "post_comments": post.comments.list(),
                })
            
            return post_data
        else:
            logging.info("No posts found for the search query")
            return None



    except Exception as e:
        logging.error(f"Error: {e}")
        return None

def get_post_comments(comments:list):
    """
    Get comments for a post and return them as a DataFrame
    """
    try:
        if len(comments) > 0:
            logging.info(f"Found {len(comments)} comments for the post")

            # Prepare comments data
            comments_data = []

            for comment in comments[:25]: # Limit to 25 comments
                comments_data.append({
                    "comment_id": comment.id,
                    "comment_created_utc": comment.created_utc,

                    "comment": comment.body,
                    "comment_score": comment.score,
                    "comment_level": get_comment_level(comment),
                    "comment_parent_id": comment.parent_id,
                    "post_id": comment.submission.id,
                })


            # Convert to DataFrame
           
            return comments_data
        

        else:
            logging.info("No comments found for the post")
            return None
        
        
    except Exception as e:
        logging.error(f"Error: {e}")
        return None

def main():
    # establishing connection to the database
    conn = psql.connect_to_db(host = os.getenv("HOST"),
                                port = os.getenv("PORT"),
                                database = os.getenv("DATABASE"),
                                user = os.getenv("DB_USER"),
                                password = os.getenv("PASSWORD"))



    search_query = "crewai vs n8n"

    # ---- /// getting posts based on search query
    s= get_post_details(search_query,post_limit=5)
    s = sorted(s,key=lambda x: x['post_score'],reverse=True) # sort by score
    # removing comment objects from the post data
    posts_data = [{k:v for k,v in i.items() if k not in ['post_comments']} for i in s]
    # convert to dataframe and formating
    posts_df = pd.DataFrame(posts_data)
    posts_df['post_created_utc'] = pd.to_datetime(posts_df['post_created_utc'],unit='s')
    posts_df['search_query'] = search_query # add search query to the dataframe
    # upserting posts data
    psql.upsert_table(conn, posts_df, "posts",["post_id"])

    # ---- /// getting comments based on post data
    comments_data = [get_post_comments(i['post_comments']) for i in s]
    comments_df  = pd.concat([pd.DataFrame(x) for x in comments_data])
    # convert comment_created_utc to datetime
    comments_df['comment_created_utc'] = pd.to_datetime(comments_df['comment_created_utc'],unit='s')

    #upserting comments data
    psql.upsert_table(conn, comments_df, "comments_new",["comment_id"])
    psql.query_db(conn, "SELECT * FROM comments_new")




# -----/// Creating table for posts and comments
# create_table_query = """
# CREATE TABLE IF NOT EXISTS posts (
#     post_id VARCHAR(255) PRIMARY KEY,
#     post_created_utc TIMESTAMP,
#     post_title TEXT,
#     post_score INTEGER,
#     post_upvote_ratio FLOAT,
#     post_num_comments INTEGER,
#     post_url TEXT,
#     search_query TEXT
# );
# """
# psql.create_tables(conn, create_table_query)
# create_table_query = """
# CREATE TABLE IF NOT EXISTS comments_new (
#     comment_id VARCHAR(255) PRIMARY KEY,
#     comment_created_utc TIMESTAMP,
#     comment TEXT,
#     comment_score INTEGER,
#     comment_level INTEGER,
#     comment_parent_id VARCHAR(255),
#     post_id VARCHAR(255)
# );
# """
# psql.create_tables(conn, create_table_query)


if __name__ == "__main__":
    main()