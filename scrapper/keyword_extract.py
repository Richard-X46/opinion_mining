import torch
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from scrapper.ls_psql import query_db, connect_to_db
import os
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()
client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)


def extract_keywords_with_sentiment(text, dominant_sentiment):
    """Extract keywords while considering the overall sentiment"""
    
    prompt = f"""
    Given the following text and its dominant sentiment ({dominant_sentiment}), 
    extract exactly 15 keywords/phrases that reflect this sentiment.
    Include both neutral descriptive terms and sentiment-aligned terms.
    
    Text: {text}
    
    Return only a comma-separated list of keywords, with no additional text.
    Each keyword should optionally include a sentiment qualifier if relevant.
    """
    
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are a sentiment-aware keyword extraction assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=100
        )
        
        keywords = [k.strip() for k in response.choices[0].message.content.split(',')][1:]
        return keywords
        
    except Exception as e:
        print(f"Error in sentiment-aware keyword extraction: {str(e)}")