#!/usr/bin/env python3
"""
BERT-based Q&A Retrieval
========================
Takes a user question, encodes it using the same sentence-transformer model
(all-MiniLM-L6-v2), and retrieves the top 10 most similar Q&A pairs from
the training dataset using cosine similarity.

Usage:
    python -m Bert.retrieve
    # or from inside the Bert folder:
    python retrieve.py
"""

import csv
import json
import os
import sys
import numpy as np

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


def cosine_similarity(vec_a, vec_b):
    """Compute cosine similarity between two vectors."""
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def find_similar(query_embedding, data, top_k=TOP_K):
    """Find the top-k most similar items by cosine similarity."""
    # Stack all embeddings into a matrix for fast computation
    all_embeddings = np.stack([item["embedding"] for item in data])

    # Compute cosine similarities in batch
    # Normalize query
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
    # Normalize all embeddings
    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True) + 1e-10
    all_normed = all_embeddings / norms

    # Dot product gives cosine similarity
    similarities = np.dot(all_normed, query_norm)

    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "question": data[idx]["question"],
            "answers": data[idx]["answers"],
            "disease": data[idx]["disease"],
            "similarity": float(similarities[idx]),
        })

    return results


def display_results(results, query):
    """Pretty-print the retrieval results."""
    print("\n" + "=" * 70)
    print(f"  🔍 QUERY: {query}")
    # {'...' if len(query) > 100 else ''}
    print("=" * 70)

    if not results:
        print("  No relevant results found.")
        return

    for i, result in enumerate(results):
        sim_pct = result["similarity"] * 100
        print(f"\n  ── Result #{i+1} (similarity: {sim_pct:.1f}%) ──")
        print(f"  📌 Disease:  {result['disease']}")
        print(f"  ❓ Question: {result['question']}")
        # '...' if len(result['question']) > 150 else ''}

        answers = result["answers"]
        if answers:
            print(f"  💬 Top Answers ({len(answers)} total):")
            for j, ans in enumerate(answers):  # Show top 3 answers for readability
                answer_text = ans.get("answer", "")
                score = ans.get("score", 0)
                preview = answer_text
                # if len(answer_text) > 200:
                #     preview += "..."
                # Use safe encoding to prevent terminal crashes on weird characters
                safe_preview = preview.encode("ascii", "ignore").decode("ascii")
                print(f"     #{j+1} (score: {score}): {safe_preview}")
        #     if len(answers) > 3:
        #         print(f"     ... and {len(answers) - 3} more answers")
        # else:
        #     print("  💬 No answers available.")

    print("\n" + "=" * 70)


def main():
    print("=" * 70)
    print("  🧠 BERT-based Q&A Retrieval System")
    print("=" * 70)

    # ── Load training data ──
    print("\n📂 Loading training data...")
    data = load_training_data()

    if not data:
        print("\n❌ No training data found! Run reddit_qa_bert.py first.")
        return

    print(f"\n✅ Loaded {len(data):,} Q&A pairs from {len(TRAIN_FILES)} disease datasets")

    # ── Load sentence-transformer model ──
    print("\n🧠 Loading sentence-transformer model (all-MiniLM-L6-v2)...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ Model loaded! Ready for queries.\n")

    # ── Interactive loop ──
    while True:
        print("-" * 70)
        print("Enter your question (or 'quit' to exit):")
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if not query:
            print("Please enter a question.")
            continue

        if query.lower() in ("quit", "exit", "q"):
            print("Exiting...")
            break

        # Encode the query
        query_embedding = model.encode(query)

        # Find similar Q&A pairs
        results = find_similar(query_embedding, data, top_k=TOP_K)

        # Display results
        display_results(results, query)


if __name__ == "__main__":
    main()







