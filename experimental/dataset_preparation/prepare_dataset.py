import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import re
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)
    # Remove numbers, punctuations, and unwanted characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize and lemmatize
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(lemmatized_words)

def process_reddit_data(submissions_file, comments_file, disease_name, kw_model):
    print(f"Processing {disease_name}...")
    try:
        # Load datasets
        subs = pd.read_csv(submissions_file)
        comms = pd.read_csv(comments_file)
        
        # Fill NaNs
        subs['title'] = subs['title'].fillna('')
        subs['body'] = subs['body'].fillna('')
        comms['body'] = comms['body'].fillna('')
        
        # Format the question: Title + "\n\n" + Body
        subs['question(post)'] = subs.apply(
            lambda x: f"{x['title']}\n\n{x['body']}".strip(), axis=1
        )
        
        # Clean link_id in comments to match submission id
        if 'link_id' in comms.columns:
            comms['link_id'] = comms['link_id'].astype(str).str.replace('t3_', '', regex=False)
        else:
            print(f"Warning: No link_id in {comments_file}")
            return pd.DataFrame()
            
        # Filter out empty comments if any
        comms = comms[comms['body'].str.strip() != '']
        
        # Sort by score descending and take top 5 per post
        if 'score' in comms.columns:
            comms['score'] = pd.to_numeric(comms['score'], errors='coerce').fillna(0)
            comms = comms.sort_values(by=['link_id', 'score'], ascending=[True, False])
            
        top_comms = comms.groupby('link_id').head(5)
        
        # Group comments by post ID
        grouped_comms = top_comms.groupby('link_id')['body'].apply(list).reset_index()
        grouped_comms = grouped_comms.rename(columns={'link_id': 'id', 'body': 'answers(comments)'})
        
        # Merge submissions and comments
        merged = pd.merge(subs, grouped_comms, on='id', how='inner') # Use inner to only keep posts with comments
        
        # If any posts have empty lists somehow, filter them out
        merged = merged[merged['answers(comments)'].map(len) > 0]
        
        # Add disease label
        merged['disease'] = disease_name
        
        # Keep only required columns
        final_df = merged[['question(post)', 'answers(comments)', 'disease']].copy()
        
        # Extract keyphrases
        print(f"Extracting keyphrases for {disease_name}...")
        
        def extract_kws(text):
            if not str(text).strip(): return []
            preprocessed = preprocess_text(text)
            if not preprocessed.strip(): return []
            return [kw[0] for kw in kw_model.extract_keywords(preprocessed, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=5)]
            
        final_df['keyphrases'] = final_df['question(post)'].apply(extract_kws)
        
        print(f"Found {len(final_df)} Q&A pairs for {disease_name}.")
        return final_df
        
    except Exception as e:
        print(f"Error processing {disease_name}: {e}")
        return pd.DataFrame()

def process_quora_autism(file_path, disease_name, kw_model):
    print(f"Processing {disease_name}...")
    try:
        df = pd.read_csv(file_path)
        
        # Fill NaNs
        df['question'] = df['question'].fillna('')
        df['answer'] = df['answer'].fillna('')
        df['comment_answer'] = df['comment_answer'].fillna('')
        
        df = df[df['question'].str.strip() != '']
        
        # Combine answer and comment_answer into a single list of (text, score) for each row
        def get_answers(row):
            ans = []
            if str(row['answer']).strip():
                try:
                    score = float(row.get('upvotes', 0))
                    if np.isnan(score): score = 0.0
                except:
                    score = 0.0
                ans.append((str(row['answer']).strip(), score))
            if str(row['comment_answer']).strip():
                try:
                    score = float(row.get('comment_upvotes', 0))
                    if np.isnan(score): score = 0.0
                except:
                    score = 0.0
                ans.append((str(row['comment_answer']).strip(), score))
            return ans
            
        df['row_answers'] = df.apply(get_answers, axis=1)
        
        # Group by question, flatten the answers list, sort by score and take top 5
        def process_grouped_answers(x):
            all_ans = [item for sublist in x for item in sublist]
            all_ans.sort(key=lambda item: item[1], reverse=True)
            return [item[0] for item in all_ans][:5]

        grouped = df.groupby('question')['row_answers'].apply(process_grouped_answers).reset_index()
        
        grouped = grouped.rename(columns={'question': 'question(post)', 'row_answers': 'answers(comments)'})
        grouped = grouped[grouped['answers(comments)'].map(len) > 0]
        
        grouped['disease'] = disease_name
        
        final_df = grouped[['question(post)', 'answers(comments)', 'disease']].copy()
        
        # Extract keyphrases
        print(f"Extracting keyphrases for {disease_name}...")
        
        def extract_kws(text):
            if not str(text).strip(): return []
            preprocessed = preprocess_text(text)
            if not preprocessed.strip(): return []
            return [kw[0] for kw in kw_model.extract_keywords(preprocessed, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=5)]
            
        final_df['keyphrases'] = final_df['question(post)'].apply(extract_kws)
        
        print(f"Found {len(final_df)} Q&A pairs for {disease_name}.")
        return final_df
        
    except Exception as e:
        print(f"Error processing {disease_name}: {e}")
        return pd.DataFrame()

from keybert import KeyBERT

def main():
    print("Loading KeyBERT model...")
    kw_model = KeyBERT()
    
    datasets = []
    
    # Process Schizophrenia
    schizo_df = process_reddit_data(
        'redditdataset/schizophrenia_submissions.csv', 
        'redditdataset/schizophrenia_comments.csv', 
        'schizophrenia',
        kw_model
    )
    if not schizo_df.empty: datasets.append(schizo_df)
    
    # Process OCD
    ocd_df = process_reddit_data(
        'redditdataset/OCD_submissions.csv', 
        'redditdataset/OCD_comments.csv', 
        'ocd',
        kw_model
    )
    if not ocd_df.empty: datasets.append(ocd_df)
    
    # Process ADHD
    adhd_df = process_reddit_data(
        'redditdataset/adhdindia_submissions.csv', 
        'redditdataset/adhdindia_comments.csv', 
        'adhd',
        kw_model
    )
    if not adhd_df.empty: datasets.append(adhd_df)
    
    # Process Dyslexia
    dyslexia_df = process_reddit_data(
        'redditdataset/dyslexia_posts.csv', 
        'redditdataset/dyslexia_comments.csv', 
        'dyslexia',
        kw_model
    )
    if not dyslexia_df.empty: datasets.append(dyslexia_df)
        
    # Process Autism
    autism_df = process_quora_autism('redditdataset/IRE_Autism .csv', 'autism', kw_model)
    if not autism_df.empty: datasets.append(autism_df)
        
    # Split each dataset and combine
    train_dfs = []
    test_dfs = []
    
    print("\nSplitting data into 90% train and 10% test...")
    for df in datasets:
        disease = df.iloc[0]['disease']
        if len(df) < 2:
            print(f"Not enough data for {disease} to split. Skipping.")
            continue
            
        train, test = train_test_split(df, test_size=0.1, random_state=42)
        train_dfs.append(train)
        test_dfs.append(test)
        print(f"{disease.capitalize()} -> Train: {len(train)}, Test: {len(test)}")
        
    if train_dfs and test_dfs:
        train_dataset = pd.concat(train_dfs, ignore_index=True)
        test_dataset = pd.concat(test_dfs, ignore_index=True)
        
        # Shuffle datasets
        train_dataset = train_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
        test_dataset = test_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"\nFinal Train dataset size: {len(train_dataset)}")
        print(f"Final Test dataset size: {len(test_dataset)}")
        
        # Save to JSONL
        train_file = 'train_dataset.jsonl'
        test_file = 'test_dataset.jsonl'
        
        train_dataset.to_json(train_file, orient='records', lines=True, force_ascii=False)
        test_dataset.to_json(test_file, orient='records', lines=True, force_ascii=False)
        
        print(f"Successfully saved to {train_file} and {test_file}")
    else:
        print("No data available to create datasets.")

if __name__ == "__main__":
    main()
