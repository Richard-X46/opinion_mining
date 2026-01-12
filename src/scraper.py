import praw
from ddgs import DDGS
import logging
from src.config import Config

class RedditScraper:
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=Config.REDDIT_CLIENT_ID,
            client_secret=Config.REDDIT_CLIENT_SECRET,
            user_agent=Config.REDDIT_USER_AGENT,
            username=Config.REDDIT_USERNAME,
            password=Config.REDDIT_PASSWORD,
        )

    def search_posts_ddgs(self, query, max_results=5):
        with DDGS() as ddgs:
            results = ddgs.text(f"site:reddit.com {query}", max_results=max_results)


            # filtering valid urls
            valid_results = {}
            for _ in results:
                # Must be a specific thread (has /comments/), not just a subreddit page
                if "reddit.com/r/" in _["href"] and "/comments/" in _["href"]:
                    valid_results[_["href"]] = _
            return list(valid_results.values())
        
        
    def fetch_post_comments(self, post_url):
        post = self.reddit.submission(url = post_url)

        # need to fix for relevance , get parent comments
        
        post.comment_sort = 'top'
        # limit=0 avoids 429 errors by NOT fetching nested 'more comments'
        post.comments.replace_more(limit=20)
        comments = [comment.body for comment in post.comments.list()]
        return comments
    


if __name__ == "__main__":
    scraper = RedditScraper()
    print(scraper.reddit.user.me()) # its working

    # test ddgs search
    results = scraper.search_posts_ddgs("google antigravity vs vscode 2025", max_results=10)
    for i in results:
        print(i['title'],"\n",i['href'])
        
    comments = scraper.fetch_post_comments(results[0]['href'])
    print(len(comments))
    print(comments[:3])
