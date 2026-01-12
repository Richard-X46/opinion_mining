from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from src.scraper import RedditScraper
from src.analyzer import CommentAnalyzer
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)

# Initialize app
app = FastAPI()
scraper = RedditScraper()
analyzer = CommentAnalyzer()


# Templates
templates = Jinja2Templates(directory="src/templates")

# application routes
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze", response_class=HTMLResponse)
def analyze(request: Request, query: str = Form(...)):
    logging.info(f"Analyzing query: {query}")
    posts = scraper.search_posts_ddgs(query,max_results=10)

    # handling the posts

    if not posts:
        return templates.TemplateResponse("partials/results.html", {"request": request, "query": query, "error": "No posts found"})
    
    # comments
    top_post = posts[0]
    comments = scraper.fetch_post_comments(top_post['href'])
    # analyzer
    # 4. Filter & Return Partial
    results = analyzer.rank_comments(query, comments)
    
    # Filter > 40% similarity (0.4)
    filtered_results = [(c, s) for c, s in results if s >= 0.4]

    if not filtered_results:
         return templates.TemplateResponse("partials/results.html", {
            "request": request, "query": query, "error": "No comments matched your query (>40% similarity)."
        })

    return templates.TemplateResponse("partials/results.html", {
    "request": request, 
    "query": query, 
    "top_post_url": top_post['href'], 
    "comments": filtered_results[:10]})


if __name__ == "__main__":
    uvicorn.run("src.app:app", host="0.0.0.0", port=5001, reload=True)
