"""
Centralized Configuration
=========================

All model names, file paths, and hyperparameters in one place.
Change a value here and every module picks it up automatically.
"""

import os

# ── Project Root ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Data Paths ──
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REDDIT_RAW_DIR = os.path.join(DATA_DIR, "reddit", "raw")
REDDIT_OUTPUT_DIR = os.path.join(DATA_DIR, "reddit", "processed", "bert")
REDDIT_KEYBERT_OUTPUT_DIR = os.path.join(DATA_DIR, "reddit", "processed", "keybert")

# ── Embedding Model ──
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"

# ── Summarizer Models ──
BART_MODEL_NAME = "facebook/bart-large-cnn"
BIGBIRD_MODEL_NAME = "google/bigbird-pegasus-large-pubmed"
PEGASUS_X_MODEL_NAME = "pszemraj/pegasus-x-large-book-summary"
T5_MODEL_NAME = "pszemraj/long-t5-tglobal-base-16384-book-summary"
GEMMA_OLLAMA_MODEL = "gemma3:4b"

# ── Retrieval Hyperparameters ──
BOOK_TOP_K = 5              # Number of textbook paragraphs to retrieve
REDDIT_TOP_K = 5            # Number of Reddit discussions to retrieve
SBERT_SIMILARITY_POOL = 10  # Candidates to fetch before MMR re-ranking
MMR_LAMBDA = 0.7            # MMR trade-off: 1.0 = pure relevance, 0.0 = pure diversity
SOURCE_BOOST = 0.06         # Similarity boost for detected disease books

# ── FAISS ──
FAISS_N_PROBE = 10          # Number of clusters to search per query

# ── Summarizer Hyperparameters ──
BART_CHUNK_SIZE = 900       # Tokens per chunk
BART_CHUNK_OVERLAP = 100    # Overlap between chunks
BART_MAX_OUTPUT = 300       # Max generated tokens
BIGBIRD_MAX_INPUT = 4096    # BigBird max input tokens
BIGBIRD_MAX_OUTPUT = 200    # BigBird max generated tokens
