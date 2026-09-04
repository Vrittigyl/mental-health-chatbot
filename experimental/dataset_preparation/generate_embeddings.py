import json
import time
import os
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

def get_keyphrase_string(row):
    keyphrases = row.get("keyphrases", [])
    if isinstance(keyphrases, list) and len(keyphrases) > 0:
        cleaned = [str(kp).strip() for kp in keyphrases if str(kp).strip()]
        if cleaned:
            return ", ".join(cleaned)
    # Fallback to question post title/body if keyphrases list is empty
    question = row.get("question(post)", "")
    return str(question).strip() if question else ""

def generate_embeddings_for_dataset(jsonl_path, output_pt_path, model, batch_size=512):
    print(f"\n--- Processing {jsonl_path} ---")
    start_time = time.time()
    
    if not os.path.exists(jsonl_path):
        print(f"Error: File {jsonl_path} not found!")
        return
        
    print("Loading JSONL dataset...")
    df = pd.read_json(jsonl_path, lines=True)
    total_samples = len(df)
    print(f"Loaded {total_samples} records from {jsonl_path}")
    
    print("Preparing keyphrase text strings...")
    texts = df.apply(get_keyphrase_string, axis=1).tolist()
    
    print(f"Generating embeddings using model '{MODEL_NAME}' in batches of {batch_size}...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True
    )
    
    print(f"Embedding tensor shape: {embeddings.shape}")
    print(f"Saving embeddings to {output_pt_path}...")
    # Move to CPU before saving if on GPU/MPS
    torch.save(embeddings.cpu(), output_pt_path)
    
    elapsed = time.time() - start_time
    file_size_mb = os.path.getsize(output_pt_path) / (1024 * 1024)
    print(f"Successfully saved {output_pt_path} ({file_size_mb:.2f} MB) in {elapsed:.2f} seconds.")

def main():
    print(f"Initializing SentenceTransformer with model: {MODEL_NAME}...")
    # Auto-select device: MPS if available on Mac, CUDA if GPU, else CPU
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        
    print(f"Using device: {device}")
    model = SentenceTransformer(MODEL_NAME, device=device)
    
    # 1. Process test dataset first (smaller, fast validation)
    test_jsonl = "test_dataset.jsonl"
    test_pt = "test_keyphrase_embeddings.pt"
    generate_embeddings_for_dataset(test_jsonl, test_pt, model, batch_size=512)
    
    # 2. Process train dataset
    train_jsonl = "train_dataset.jsonl"
    train_pt = "train_keyphrase_embeddings.pt"
    generate_embeddings_for_dataset(train_jsonl, train_pt, model, batch_size=512)
    
    print("\n[SUCCESS] Keyphrase embeddings generation complete for both datasets!")

if __name__ == "__main__":
    main()
