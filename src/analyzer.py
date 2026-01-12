from sentence_transformers import SentenceTransformer, util

class CommentAnalyzer:
    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")



    def rank_comments(self, query, comments):
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        
        comment_embeddings = self.embedder.encode(comments, convert_to_tensor=True)
        
        cosine_scores = util.pytorch_cos_sim(query_embedding, comment_embeddings)
        
        comment_score_pairs = list(zip(comments, cosine_scores[0].tolist()))
        
        sorted_comments = sorted(comment_score_pairs, key=lambda x: x[1], reverse=True)
        
        return sorted_comments

if __name__ == "__main__":
    print("running analyzer")
    analyzer = CommentAnalyzer()
    test_query = "google antigravity vs vscode 2025"

    test_comments = ["google antigravity is the best", 
                    "vscode is the best",
                     "google antigravity is the best",
                     "aws has better models",
                     "openai does a better job",
                     "cursor is better ide",

                     "vscode is the best"]
    comments = analyzer.rank_comments(test_query, test_comments)
    for comment, score in comments:
        print(comment, score,"\n")