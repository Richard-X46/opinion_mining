import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from scrapper.ls_psql import query_db, connect_to_db
import os
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

# Load pre-trained model tokenizer and model for DistilBERT
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Initialize OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_keywords_bert(text):
    """Extract 15 keywords using DistilBERT model"""
    # Tokenize input text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    
    # Get hidden states from DistilBERT
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the embeddings for the [CLS] token
    cls_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    
    # Use CountVectorizer to get the most common words
    vectorizer = CountVectorizer(stop_words='english', max_features=15)
    X = vectorizer.fit_transform([text])
    word_counts = np.asarray(X.sum(axis=0)).flatten()
    words = vectorizer.get_feature_names_out()
    
    # Get the top 15 words
    top_indices = word_counts.argsort()[-15:][::-1]
    keywords = [words[i] for i in top_indices]
    
    return keywords

def extract_keywords_gpt(text):
    """Extract 15 keywords using GPT-3.5 Turbo"""
    try:
        prompt = f"""Extract exactly 15 key topics or keywords from the following text. 
        Return only the keywords as a comma-separated list, without numbering or explanation:

        Text: {text}"""

        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a keyword extraction assistant. Extract key topics and return them as a comma-separated list only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=100
        )

        # Extract keywords from response and clean them
        keywords_text = response.choices[0].message.content.strip()
        keywords = [k.strip() for k in keywords_text.split(',')][:15]
        
        return keywords
    except Exception as e:
        print(f"Error in GPT keyword extraction: {str(e)}")
        # Fallback to BERT if GPT fails
        return extract_keywords_bert(text)