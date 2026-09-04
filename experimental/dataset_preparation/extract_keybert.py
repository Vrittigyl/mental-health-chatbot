import pandas as pd
from keybert import KeyBERT
from collections import defaultdict
import time
import os

def process_disease(disease_name, files_info, kw_model, output_dir):
    phrase_scores = defaultdict(float)
    
    for file_info in files_info:
        file_path = file_info['path']
        columns = file_info['columns']
        
        if not os.path.exists(file_path):
            print(f"File {file_path} not found. Skipping...")
            continue
            
        print(f"Processing {file_path} for {disease_name}...")
        try:
            for chunk in pd.read_csv(file_path, chunksize=50000, lineterminator='\n', on_bad_lines='skip'):
                text_list = []
                for col in columns:
                    if col in chunk.columns:
                        col_texts = chunk[col].dropna().astype(str)
                        col_texts = col_texts[~col_texts.isin(['[deleted]', '[removed]', 'nan', ''])]
                        text_list.extend(col_texts.tolist())
                
                # Process in batches
                batch_size = 2000
                for i in range(0, len(text_list), batch_size):
                    batch = " . ".join(text_list[i:i+batch_size])
                    # max keyphrase length 3
                    keywords = kw_model.extract_keywords(
                        batch, 
                        keyphrase_ngram_range=(1, 3), 
                        stop_words='english',
                        top_n=50
                    )
                    
                    for phrase, score in keywords:
                        phrase_scores[phrase] += score
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    sorted_phrases = sorted(phrase_scores.items(), key=lambda x: x[1], reverse=True)
    out_file = os.path.join(output_dir, f"{disease_name}_keybert_keywords.csv")
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write("phrase,score\n")
        # Save top 50,000 phrases to prevent massive output files
        for phrase, score in sorted_phrases[:50000]:
            clean_phrase = str(phrase).replace('"', '""')
            f.write(f'"{clean_phrase}",{score:.4f}\n')
    
    print(f"Saved {disease_name} keywords to {out_file} (Top 50000)")

def main():
    diseases = {
        "autism": [
            {"path": "redditdataset/IRE_Autism .csv", "columns": ["question", "answer", "comment_answer"]}
        ],
        "adhd": [
            {"path": "redditdataset/adhdindia_submissions.csv", "columns": ["title", "body"]},
            {"path": "redditdataset/adhdindia_comments.csv", "columns": ["body"]}
        ],
        "dyslexia": [
            {"path": "redditdataset/dyslexia_posts.csv", "columns": ["title", "body"]},
            {"path": "redditdataset/dyslexia_comments.csv", "columns": ["body"]}
        ],
        "schizophrenia": [
            {"path": "redditdataset/schizophrenia_submissions.csv", "columns": ["title", "body"]},
            {"path": "redditdataset/schizophrenia_comments.csv", "columns": ["body"]}
        ],
        "ocd": [
            {"path": "redditdataset/OCD_submissions.csv", "columns": ["title", "body"]},
            {"path": "redditdataset/OCD_comments.csv", "columns": ["body"]}
        ]
    }

    # Combine those files in one folder in redditdataset
    output_dir = "redditdataset/keybert_keywords"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Initializing KeyBERT model...")
    kw_model = KeyBERT()

    start_time = time.time()
    for d, info in diseases.items():
        process_disease(d, info, kw_model, output_dir)
    print(f"Finished all in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
