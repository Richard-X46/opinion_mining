import torch
from transformers import BertTokenizer, BertModel
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from scrapper.ls_psql import query_db, connect_to_db
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load pre-trained model tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load pre-trained model
model = BertModel.from_pretrained('bert-base-uncased')

def extract_keywords(text, top_n=5):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    
    # Get hidden states from BERT
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the embeddings for the [CLS] token
    cls_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    
    # Use CountVectorizer to get the most common words
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform([text])
    word_counts = np.asarray(X.sum(axis=0)).flatten()
    words = vectorizer.get_feature_names_out()
    
    # Get the top N words
    top_indices = word_counts.argsort()[-top_n:][::-1]
    keywords = [words[i] for i in top_indices]
    
    return keywords

if __name__ == "__main__":
    # Connect to database only when running as main script
    conn = connect_to_db(
        os.getenv("HOST"),
        os.getenv("PORT"),
        os.getenv("DATABASE"),
        os.getenv("USER"),
        os.getenv("PASSWORD")
    )
    
    if conn:
        df = query_db(conn, "SELECT * FROM Comments")
        if df is not None:
            list_comments = df['comment'].tolist()
            for comment in list_comments:
                keywords = extract_keywords(comment)
                print(f"Comment: {comment}\nKeywords: {keywords}\n")