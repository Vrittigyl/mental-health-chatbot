import pandas as pd
from keybert import KeyBERT
from rake_nltk import Rake
import nltk
import warnings
warnings.filterwarnings('ignore')

# download nltk stopwords if needed
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

def main():
    print("Loading datasets...")
    posts_csv = "redditdataset/dyslexia_posts.csv"
    comments_csv = "redditdataset/dyslexia_comments.csv"
    
    # Just load first 100 rows to make it fast
    posts = pd.read_csv(posts_csv, nrows=100)
    comments = pd.read_csv(comments_csv, nrows=100)
    
    comments['post_id'] = comments['link_id'].astype(str).str.replace('t3_', '')
    merged_df = pd.merge(posts, comments, left_on='id', right_on='post_id', suffixes=('_post', '_comment'))
    
    merged_df['title'] = merged_df['title'].fillna('')
    merged_df['body_post'] = merged_df['body_post'].fillna('')
    merged_df['body_comment'] = merged_df['body_comment'].fillna('')
    
    merged_df['combined_text'] = (
        "Question Title: " + merged_df['title'] + " " +
        "Question Body: " + merged_df['body_post'] + " " +
        "Answer: " + merged_df['body_comment']
    )
    
    dataset_texts = merged_df['combined_text'].tolist()
    if not dataset_texts:
        print("No matched data. Trying without nrows limit...")
        posts = pd.read_csv(posts_csv)
        comments = pd.read_csv(comments_csv)
        comments['post_id'] = comments['link_id'].astype(str).str.replace('t3_', '')
        merged_df = pd.merge(posts, comments, left_on='id', right_on='post_id', suffixes=('_post', '_comment'))
        merged_df['title'] = merged_df['title'].fillna('')
        merged_df['body_post'] = merged_df['body_post'].fillna('')
        merged_df['body_comment'] = merged_df['body_comment'].fillna('')
        merged_df['combined_text'] = (
            "Question Title: " + merged_df['title'] + " " +
            "Question Body: " + merged_df['body_post'] + " " +
            "Answer: " + merged_df['body_comment']
        )
        dataset_texts = merged_df['combined_text'].tolist()
    
    print("Initializing KeyBERT...")
    kw_model = KeyBERT()
    
    print("Initializing RAKE...")
    r = Rake()
    
    sample_query = "I have trouble reading and mix up letters. What should I do?"
    
    print("\n" + "="*50)
    print("--- QUERY KEYWORDS ---")
    print(f"Original Query: {sample_query}")
    print("\nKeyBERT (top 5, ngram 1-2):")
    query_keywords_kb = kw_model.extract_keywords(sample_query, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5)
    for kw, score in query_keywords_kb:
        print(f"  - {kw}: {score:.4f}")
    
    print("\nRAKE (top 5 phrases):")
    r.extract_keywords_from_text(sample_query)
    for score, kw in r.get_ranked_phrases_with_scores()[:5]:
        print(f"  - {kw}: {score:.4f}")
    print("="*50)
    
    print("\n--- DATASET KEYWORDS (First 2 examples) ---")
    for i in range(min(2, len(dataset_texts))):
        text = dataset_texts[i][:1000] # truncate for speed if too long
        print(f"\nExample {i+1} Text (truncated): {text[:350]}...")
        
        print("\nKeyBERT (top 5, ngram 1-2):")
        doc_keywords_kb = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5)
        for kw, score in doc_keywords_kb:
            print(f"  - {kw}: {score:.4f}")
        
        print("\nRAKE (top 5 phrases):")
        r.extract_keywords_from_text(text)
        for score, kw in r.get_ranked_phrases_with_scores()[:5]:
            print(f"  - {kw}: {score:.4f}")
        print("-" * 50)

if __name__ == "__main__":
    main()
