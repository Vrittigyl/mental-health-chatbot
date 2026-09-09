import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

class DyslexiaQA:
    def __init__(self, posts_path, comments_path, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        print("Loading model...")
        self.model = SentenceTransformer(model_name)
        
        print("Loading datasets...")
        self.posts = pd.read_csv(posts_path)
        self.comments = pd.read_csv(comments_path)
        
        self.prepare_dataset()
        self.generate_embeddings()
        
    def prepare_dataset(self):
        print("Preparing dataset...")
        # Link ID in comments has a 't3_' prefix for posts
        # We need to strip it to match the post ID
        self.comments['post_id'] = self.comments['link_id'].astype(str).str.replace('t3_', '')
        
        # Merge posts and comments
        self.merged_df = pd.merge(
            self.posts, 
            self.comments, 
            left_on='id', 
            right_on='post_id',
            suffixes=('_post', '_comment')
        )
        
        # Fill NaN values with empty string
        self.merged_df['title'] = self.merged_df['title'].fillna('')
        self.merged_df['body_post'] = self.merged_df['body_post'].fillna('')
        self.merged_df['body_comment'] = self.merged_df['body_comment'].fillna('')
        
        # Create the combined text: Question + Answer
        self.merged_df['combined_text'] = (
            "Question Title: " + self.merged_df['title'] + " " +
            "Question Body: " + self.merged_df['body_post'] + " " +
            "Answer: " + self.merged_df['body_comment']
        )
        
        # Keep references for returning results
        self.dataset_texts = self.merged_df['combined_text'].tolist()
        self.answers = self.merged_df['body_comment'].tolist()
        
    def generate_embeddings(self):
        print(f"Generating embeddings for {len(self.dataset_texts)} dataset points...")
        self.embeddings = self.model.encode(self.dataset_texts, convert_to_tensor=True)
        print("Embeddings generated.")
        
    def query(self, query_text, top_k=3):
        query_embedding = self.model.encode(query_text, convert_to_tensor=True)
        
        # Compute cosine similarities
        similarities = util.cos_sim(query_embedding, self.embeddings)[0]
        
        # Get top k indices
        top_k = min(top_k, len(self.dataset_texts))
        top_indices = torch.topk(similarities, k=top_k).indices.tolist()
        
        results = []
        for idx in top_indices:
            results.append({
                "score": similarities[idx].item(),
                "combined_text": self.dataset_texts[idx],
                "answer": self.answers[idx]
            })
            
        return results

if __name__ == "__main__":
    posts_csv = "redditdataset/dyslexia_posts.csv"
    comments_csv = "redditdataset/dyslexia_comments.csv"
    
    qa_system = DyslexiaQA(posts_csv, comments_csv)
    
    sample_query = "I have trouble reading and mix up letters. What should I do?"
    print(f"\nQuery: {sample_query}")
    print("-" * 50)
    
    top_answers = qa_system.query(sample_query, top_k=5)
    
    for i, ans in enumerate(top_answers, 1):
        print(f"Result {i} (Score: {ans['score']:.4f}):")
        print(f"Answer: {ans['combined_text']}")
        print("-" * 50)


# from sentence_transformers import SentenceTransformer

# # 1. Load a pretrained Sentence Transformer model
# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# # The sentences to encode
# sentences = [
#     "The weather is lovely today."
# ]

# # 2. Calculate embeddings by calling model.encode()
# embeddings = model.encode(sentences)
# print(embeddings)
# print(embeddings.shape)
# # [3, 384]

# # 3. Calculate the embedding similarities
# similarities = model.similarity(embeddings, embeddings)
# print(similarities)