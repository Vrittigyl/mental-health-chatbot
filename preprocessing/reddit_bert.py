#!/usr/bin/env python3
"""
Reddit Mental Health Q&A Dataset Processor with BERT Embeddings
===============================================================
Processes datasets for 5 conditions: Autism, ADHD, Schizophrenia, OCD, Dyslexia.

For each question (submission title), finds the TOP 10 answers ranked by
score (upvotes - downvotes) and stores them as an array.

Splits 90% train / 10% test.
Extracts BERT embeddings using sentence-transformers.

Output:
  - test_combined_with_embeddings.csv  → 10% test, ALL diseases combined
  - train_autism.csv                 → 90% train, Autism only
  - train_adhd.csv                   → 90% train, ADHD only
  - train_schizophrenia.csv          → 90% train, Schizophrenia only
  - train_ocd.csv                    → 90% train, OCD only
  - train_dyslexia.csv               → 90% train, Dyslexia only
"""

import csv
import json
import os
import random
import sys
import time
import re
from collections import defaultdict

# ── Configuration ────────────────────────────────────────────
csv.field_size_limit(sys.maxsize)

from config.settings import REDDIT_RAW_DIR, REDDIT_OUTPUT_DIR
DATA_DIR = REDDIT_RAW_DIR
OUTPUT_DIR = REDDIT_OUTPUT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_QA_PER_DISEASE = 5000      # Max Q&A pairs to keep per disease
MAX_COMMENTS_READ = 300000     # Max comment rows to read per file (memory safety)
TOP_N_ANSWERS = 10             # Top N answers per question (by score)
TEST_RATIO = 0.10              # 10% test, 90% train
RANDOM_SEED = 42
MIN_ANSWER_LENGTH = 30         # Minimum answer text length

random.seed(RANDOM_SEED)

def clean_text(text):
    """Basic text cleaning."""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ── 1. Process Reddit-style datasets (submissions + comments) ──

def process_comments(comment_files):
    """
    Read comment files. For each submission (link_id), collect all
    direct-reply comments with their scores. Keep top N by score.

    Returns: dict[link_id] -> [(score, body), ...] sorted desc by score, max N
    """
    comments_by_post = defaultdict(list)

    for cfile in comment_files:
        filepath = os.path.join(DATA_DIR, cfile)
        print(f"    📖 Reading comments: {cfile}...")
        count = 0
        direct_replies = 0

        if not os.path.exists(filepath):
            print(f"       ⚠️ File not found: {filepath}")
            continue

        with open(filepath, "r", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if count >= MAX_COMMENTS_READ:
                    break
                count += 1

                link_id = row.get("link_id", "")
                parent_id = row.get("parent_id", "")
                body = (row.get("body", "") or "").strip()

                # Only keep DIRECT replies to the submission (not reply-to-reply)
                if parent_id != link_id:
                    continue

                # Filter out deleted/removed/short content
                if body in ("[deleted]", "[removed]", "") or len(body) < MIN_ANSWER_LENGTH:
                    continue

                try:
                    score = int(float(row.get("score", 0)))
                except (ValueError, TypeError):
                    score = 0

                direct_replies += 1
                comments_by_post[link_id].append((score, body))

        print(f"       Read {count:,} rows → {direct_replies:,} direct replies → {len(comments_by_post):,} submissions")

    # Sort each submission's comments by score (desc) and keep top N
    for lid in comments_by_post:
        comments_by_post[lid] = sorted(
            comments_by_post[lid], key=lambda x: -x[0]
        )[:TOP_N_ANSWERS]

    return dict(comments_by_post)


def process_submissions(submission_files, comments_by_post):
    """
    Read submission files. Match each submission with its top comments.
    Question = submission title, Answers = top comments by score.

    Returns: list of {question, top_10_answers: [{answer, score}, ...]}
    """
    qa_pairs = []

    for sfile in submission_files:
        filepath = os.path.join(DATA_DIR, sfile)
        print(f"    📖 Reading submissions: {sfile}...")
        count = 0
        matched = 0

        if not os.path.exists(filepath):
            print(f"       ⚠️ File not found: {filepath}")
            continue

        with open(filepath, "r", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if len(qa_pairs) >= MAX_QA_PER_DISEASE:
                    break
                count += 1

                sid = "t3_" + (row.get("id", "") or "")
                title = (row.get("title", "") or "").strip()

                # Skip deleted/empty titles
                if not title or title in ("[deleted]", "[removed]"):
                    continue

                # Check if this submission has matching comments
                if sid in comments_by_post and comments_by_post[sid]:
                    answers = []
                    for score, body in comments_by_post[sid]:
                        answers.append({"answer": clean_text(body), "score": score})

                    qa_pairs.append({
                        "question": clean_text(title),
                        f"top_{TOP_N_ANSWERS}_answers": answers,
                    })
                    matched += 1

        print(f"       Scanned {count:,} submissions → {matched:,} matched with comments")

    print(f"    ✅ Total Q&A pairs: {len(qa_pairs)}")
    return qa_pairs


# ── 2. Process Autism dataset (already Q&A format) ──

def process_autism():
    """
    The Autism dataset (IRE_Autism .csv) is already in Q&A format.
    Columns: question, answer, upvotes, comment_answer, comment_upvotes

    Multiple rows may share the same question with different answers.
    We group by question and collect top unique answers by upvotes.
    """
    filepath = os.path.join(DATA_DIR, "IRE_Autism .csv")
    print(f"    📖 Reading: IRE_Autism .csv...")

    if not os.path.exists(filepath):
        print(f"       ⚠️ File not found: {filepath}")
        return []

    # Group: question -> list of (score, answer_text)
    qa_map = defaultdict(list)

    with open(filepath, "r", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = (row.get("question", "") or "").strip()
            if not question:
                continue

            # Main answer
            answer = (row.get("answer", "") or "").strip()
            if answer and answer not in ("[deleted]", "[removed]") and len(answer) >= MIN_ANSWER_LENGTH:
                try:
                    upvotes = int(float(row.get("upvotes", 0)))
                except (ValueError, TypeError):
                    upvotes = 0
                qa_map[question].append((upvotes, answer))

            # Comment answer
            comment_answer = (row.get("comment_answer", "") or "").strip()
            if comment_answer and comment_answer not in ("[deleted]", "[removed]") and len(comment_answer) >= MIN_ANSWER_LENGTH:
                try:
                    c_upvotes = int(float(row.get("comment_upvotes", 0)))
                except (ValueError, TypeError):
                    c_upvotes = 0
                qa_map[question].append((c_upvotes, comment_answer))

    # Build Q&A pairs: for each question, top unique answers
    qa_pairs = []
    for question, answer_list in qa_map.items():
        if len(qa_pairs) >= MAX_QA_PER_DISEASE:
            break

        # Deduplicate and sort by score
        seen = set()
        unique_answers = []
        for score, body in sorted(answer_list, key=lambda x: -x[0]):
            body_key = body[:200]  # use first 200 chars for dedup
            if body_key not in seen:
                seen.add(body_key)
                unique_answers.append({"answer": clean_text(body), "score": score})
            if len(unique_answers) >= TOP_N_ANSWERS:
                break

        if unique_answers:
            qa_pairs.append({
                "question": clean_text(question),
                f"top_{TOP_N_ANSWERS}_answers": unique_answers,
            })

    print(f"    ✅ Total Q&A pairs: {len(qa_pairs)}")
    return qa_pairs


# ── 3. BERT Embeddings extraction ──

def extract_embeddings_all(all_items):
    """
    Extract BERT embeddings for each Q&A pair using sentence-transformers.
    Combines question + best answer text for richer context.
    """
    from sentence_transformers import SentenceTransformer

    print("\n  🧠 Loading sentence-transformers model (all-MiniLM-L6-v2)...")
    st_model = SentenceTransformer("all-MiniLM-L6-v2")

    total = len(all_items)
    print(f"  🧠 Extracting embeddings for {total:,} items...")
    start_time = time.time()

    for i, item in enumerate(all_items):
        if (i + 1) % 200 == 0 or i == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(f"     {i+1:,}/{total:,} ({rate:.1f} items/sec, ETA: {eta:.0f}s)")

        # Combine question + top answer for embedding
        text = item["question"]
        answers_key = f"top_{TOP_N_ANSWERS}_answers"
        if item.get(answers_key):
            # Use first (best-scored) answer, truncated to avoid very long texts
            best_answer = item[answers_key][0]["answer"][:500]
            text += " " + best_answer

        try:
            embedding = st_model.encode(text)
            item["embedding"] = embedding.tolist()
        except Exception as e:
            item["embedding"] = []

    elapsed = time.time() - start_time
    print(f"  ✅ Embeddings extracted in {elapsed:.1f}s")


# ── 4. Save output files ──

def save_csv(data, filepath):
    """Save Q&A data to CSV with columns: question, top_10_answers, disease, embedding"""
    answers_key = f"top_{TOP_N_ANSWERS}_answers"
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question", answers_key, "disease", "embedding"])
        for item in data:
            writer.writerow([
                item["question"],
                json.dumps(item.get(answers_key, []), ensure_ascii=False),
                item["disease"],
                json.dumps(item.get("embedding", []), ensure_ascii=False),
            ])
    print(f"    💾 Saved {len(data):,} rows → {os.path.basename(filepath)}")


# ── MAIN ──

def main():
    print("=" * 70)
    print("  Reddit Mental Health Q&A Dataset Processor with BERT")
    print(f"  Top {TOP_N_ANSWERS} answers per question (ranked by score)")
    print("=" * 70)

    all_qa = {}  # disease_name -> list of qa dicts

    # ────────────────────────────────────────────────
    # Process each disease
    # ────────────────────────────────────────────────

    # 1. AUTISM
    print("\n🧩 [1/5] AUTISM")
    all_qa["autism"] = process_autism()

    # 2. ADHD (main + India subreddit)
    print("\n🧠 [2/5] ADHD")
    print("  Step A: Reading comments...")
    adhd_comments = process_comments(["ADHD_comments.csv", "adhdindia_comments.csv"])
    print("  Step B: Matching with submissions...")
    all_qa["adhd"] = process_submissions(
        ["ADHD_submissions.csv", "adhdindia_submissions.csv"], adhd_comments
    )
    del adhd_comments  # free memory

    # 3. OCD
    print("\n🔄 [3/5] OCD")
    print("  Step A: Reading comments...")
    ocd_comments = process_comments(["OCD_comments.csv"])
    print("  Step B: Matching with submissions...")
    all_qa["ocd"] = process_submissions(["OCD_submissions.csv"], ocd_comments)
    del ocd_comments

    # 4. SCHIZOPHRENIA
    print("\n🌀 [4/5] SCHIZOPHRENIA")
    print("  Step A: Reading comments...")
    schiz_comments = process_comments(["schizophrenia_comments.csv"])
    print("  Step B: Matching with submissions...")
    all_qa["schizophrenia"] = process_submissions(
        ["schizophrenia_submissions.csv"], schiz_comments
    )
    del schiz_comments

    # 5. DYSLEXIA
    print("\n📚 [5/5] DYSLEXIA")
    print("  Step A: Reading comments...")
    dys_comments = process_comments(["dyslexia_comments.csv"])
    print("  Step B: Matching with submissions...")
    all_qa["dyslexia"] = process_submissions(["dyslexia_posts.csv"], dys_comments)
    del dys_comments

    # ────────────────────────────────────────────────
    # Summary
    # ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  📊 Q&A PAIRS SUMMARY")
    print("=" * 70)
    total = 0
    for disease, pairs in all_qa.items():
        print(f"    {disease:20s}: {len(pairs):,} pairs")
        total += len(pairs)
    print(f"    {'TOTAL':20s}: {total:,} pairs")

    # ────────────────────────────────────────────────
    # Split 90% train / 10% test (stratified per disease)
    # ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  ✂️  SPLITTING {int((1-TEST_RATIO)*100)}% TRAIN / {int(TEST_RATIO*100)}% TEST (per disease)")
    print("=" * 70)

    train_data = {}   # disease -> list
    test_data = []    # combined list

    for disease, pairs in all_qa.items():
        random.shuffle(pairs)
        split_idx = int(len(pairs) * (1 - TEST_RATIO))

        train_portion = pairs[:split_idx]
        test_portion = pairs[split_idx:]

        # Tag each item with its disease
        for p in train_portion:
            p["disease"] = disease
        for p in test_portion:
            p["disease"] = disease

        train_data[disease] = train_portion
        test_data.extend(test_portion)

        print(f"    {disease:20s}: train={len(train_portion):,}  |  test={len(test_portion):,}")

    total_train = sum(len(v) for v in train_data.values())
    print(f"    {'TOTAL TRAIN':20s}: {total_train:,}")
    print(f"    {'TOTAL TEST':20s}: {len(test_data):,}")

    # ────────────────────────────────────────────────
    # BERT embeddings extraction
    # ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  🧠 BERT EMBEDDINGS EXTRACTION")
    print("=" * 70)

    # Combine all items for batch processing
    all_items = []
    for disease, pairs in train_data.items():
        all_items.extend(pairs)
    all_items.extend(test_data)

    extract_embeddings_all(all_items)

    # ────────────────────────────────────────────────
    # Save output files
    # ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  💾 SAVING OUTPUT FILES")
    print("=" * 70)

    # Test file: combined all diseases with embeddings
    save_csv(test_data, os.path.join(OUTPUT_DIR, "test_combined_with_embeddings.csv"))

    # Train files: one per disease
    for disease, pairs in train_data.items():
        save_csv(pairs, os.path.join(OUTPUT_DIR, f"train_{disease}.csv"))

    # ────────────────────────────────────────────────
    # Final report
    # ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ✅ ALL DONE!")
    print("=" * 70)
    print(f"\n  Output directory: {OUTPUT_DIR}/")
    print(f"\n  Files created:")
    print(f"    📁 test_combined_with_embeddings.csv")
    print(f"       → {len(test_data):,} rows ({int(TEST_RATIO*100)}% test, ALL diseases, with embeddings)")
    for disease, pairs in train_data.items():
        print(f"    📁 train_{disease}.csv")
        print(f"       → {len(pairs):,} rows ({int((1-TEST_RATIO)*100)}% train, {disease} only, with embeddings)")

    print(f"\n  CSV columns: question | top_{TOP_N_ANSWERS}_answers (JSON array) | disease | embedding (JSON array)")
    print(f"\n  How to identify test vs train questions:")
    print(f"    • Test questions → in test_combined_with_embeddings.csv")
    print(f"    • Train questions → in train_<disease>.csv files")
    print(f"    • The split is random but reproducible (seed={RANDOM_SEED})")

if __name__ == "__main__":
    main()
