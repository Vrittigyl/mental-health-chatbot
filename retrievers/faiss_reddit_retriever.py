"""
FAISS-based Reddit Q&A Retrieval
================================

Replaces the brute-force cosine similarity search in Bert/retrieve.py 
with a FAISS IVF index for faster nearest-neighbor search using clustering.

Usage:
    python -m Bert.faiss_retrieve
"""

# Fix macOS OpenMP conflict between FAISS and PyTorch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import csv
import json
import sys
import numpy as np
import faiss

# ── Configuration ────────────────────────────────────────────
csv.field_size_limit(sys.maxsize)

from config.settings import REDDIT_OUTPUT_DIR
OUTPUT_DIR = REDDIT_OUTPUT_DIR

TOP_K = 10  # Number of similar results to return

TRAIN_FILES = [
    "train_autism.csv",
    "train_adhd.csv",
    "train_ocd.csv",
    "train_schizophrenia.csv",
    "train_dyslexia.csv",
]


def load_training_data():
    """Load all training CSVs and parse their embeddings."""
    all_data = []

    for fname in TRAIN_FILES:
        filepath = os.path.join(OUTPUT_DIR, fname)
        if not os.path.exists(filepath):
            print(f"  ⚠️  Skipping {fname} (not found)")
            continue

        print(f"  📖 Loading {fname}...")
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                try:
                    embedding = json.loads(row.get("embedding", "[]"))
                    if not embedding:
                        continue

                    # Determine the answers key dynamically
                    answers_key = None
                    for key in row:
                        if key.startswith("top_") and key.endswith("_answers"):
                            answers_key = key
                            break

                    answers = json.loads(row.get(answers_key, "[]")) if answers_key else []

                    all_data.append({
                        "question": row.get("question", ""),
                        "answers": answers,
                        "disease": row.get("disease", ""),
                        "embedding": np.array(embedding, dtype=np.float32),
                    })
                    count += 1
                except (json.JSONDecodeError, ValueError):
                    continue

        print(f"       Loaded {count:,} rows")

    return all_data


def build_faiss_index(data):
    """
    Build a FAISS IVF index from all Reddit embeddings.
    
    Uses IndexIVFFlat (Inverted File Index):
      1. Runs K-Means to split all vectors into clusters
      2. At search time, only checks the nearest cluster(s) instead of all vectors
    
    This is faster than brute force because it skips ~80% of vectors.
    """
    # Stack all embeddings into a numpy matrix
    all_embeddings = np.stack([item["embedding"] for item in data])
    
    # L2 normalize so that inner product = cosine similarity
    faiss.normalize_L2(all_embeddings)
    
    # Get the dimensionality (e.g., 384 for all-MiniLM-L6-v2)
    dimension = all_embeddings.shape[1]
    n_vectors = all_embeddings.shape[0]
    
    # Number of clusters (rule of thumb: sqrt of total vectors)
    n_clusters = int(np.sqrt(n_vectors))  # ~143 clusters for 20,607 vectors
    
    # Number of clusters to search per query (more = slower but more accurate)
    n_probe = 10  # Search the 10 nearest clusters (~1,400 vectors instead of 20,607)
    
    # Build the IVF index
    # Step 1: Create a "quantizer" that assigns vectors to clusters
    quantizer = faiss.IndexFlatIP(dimension)
    
    # Step 2: Create the IVF index on top of the quantizer
    index = faiss.IndexIVFFlat(quantizer, dimension, n_clusters, faiss.METRIC_INNER_PRODUCT)
    
    # Step 3: Train the index (runs K-Means to find cluster centers)
    print(f"  🔧 Training K-Means with {n_clusters} clusters...")
    index.train(all_embeddings)
    
    # Step 4: Add all vectors to their respective clusters
    index.add(all_embeddings)
    
    # Step 5: Set how many clusters to probe at search time
    index.nprobe = n_probe
    
    print(f"  🔍 FAISS IVF Index built:")
    print(f"     Vectors: {index.ntotal:,}")
    print(f"     Clusters: {n_clusters}")
    print(f"     Probe: {n_probe} clusters per query (~{n_vectors // n_clusters * n_probe:,} vectors searched)")
    
    return index


def find_similar_faiss(query_embedding, index, data, top_k=TOP_K):
    """
    Find the top-k most similar items using FAISS index.
    
    Instead of:  np.dot(all_embeddings, query)     ← Python loop over 20,607 vectors
    We do:       index.search(query, top_k)        ← FAISS C++ optimized search
    """
    # Prepare query: reshape to (1, dimension) and L2 normalize
    query = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(query)
    
    # FAISS search — returns distances (scores) and indices
    scores, indices = index.search(query, top_k)
    
    # Build results
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:  # FAISS returns -1 for empty slots
            continue
        results.append({
            "question": data[idx]["question"],
            "answers": data[idx]["answers"],
            "disease": data[idx]["disease"],
            "similarity": float(score),
        })
    
    return results


def display_results(results, query):
    """Pretty-print the retrieval results."""
    print("\n" + "=" * 70)
    print(f"  🔍 QUERY: {query}")
    print("=" * 70)

    if not results:
        print("  No results found.")
        print("=" * 70)
        return

    for i, result in enumerate(results):
        sim_pct = result["similarity"] * 100
        print(f"\n  ── Result #{i+1} (similarity: {sim_pct:.1f}%) ──")
        print(f"  📌 Disease:  {result['disease']}")
        print(f"  ❓ Question: {result['question']}")

        answers = result["answers"]
        if answers:
            for j, ans in enumerate(answers):
                answer_text = ans.get("answer", "")
                score = ans.get("score", 0)
                safe_preview = answer_text.encode("ascii", "ignore").decode("ascii")
                print(f"     #{j+1} (score: {score}): {safe_preview}")

    print("\n" + "=" * 70)


# ── CLI ──────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Reddit Q&A Retriever (FAISS-powered)")
    print("=" * 70)

    # ── Load training data ──
    print("\n📂 Loading training data...")
    data = load_training_data()

    if not data:
        print("\n❌ No training data found! Run reddit_qa_bert.py first.")
        return

    print(f"\n✅ Loaded {len(data):,} Q&A pairs from {len(TRAIN_FILES)} disease datasets")

    # ── Build FAISS Index ──
    print("\n🔧 Building FAISS index...")
    index = build_faiss_index(data)

    # ── Load sentence-transformer model ──
    print("\n🧠 Loading sentence-transformer model (all-MiniLM-L6-v2)...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ Model loaded! Ready for queries.\n")

    # ── Interactive loop ──
    while True:
        try:
            user_input = input("  🔍 Enter your question (or 'quit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! 👋")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("\nGoodbye! 👋")
            break

        # Encode the query
        query_embedding = model.encode(user_input)

        # Search with FAISS
        results = find_similar_faiss(query_embedding, index, data, top_k=10)

        # Display results
        display_results(results, user_input)


if __name__ == "__main__":
    main()
