"""
Knowledge Graph Builder — Multi-Book Edition
=============================================

Extracts text from multiple mental health books (PDFs),
builds a SINGLE unified co-occurrence graph, and saves it for retrieval.

Each paragraph is tagged with its source book so results can show
which book the information came from.

Usage:
    python -m knowledge.ingest

Books:
    - OCD
    - ADHD
    - Autism
    - Dyslexia
    - Schizophrenia
"""

import re
import json
import os
from collections import defaultdict
from itertools import combinations
from functools import lru_cache

import nltk
import pdfplumber
from nltk.stem import WordNetLemmatizer

# Ensure wordnet data is available
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)

# Singleton lemmatizer instance
_lemmatizer = WordNetLemmatizer()

@lru_cache(maxsize=10000)
def cached_lemmatize(word):
    """Lemmatize with multiple POS tags, picking the shortest result."""
    lemmas = [
        _lemmatizer.lemmatize(word, pos='n'),  # noun
        _lemmatizer.lemmatize(word, pos='v'),  # verb
        _lemmatizer.lemmatize(word, pos='a'),  # adjective
    ]
    return min(lemmas, key=len)

SYNONYM_MAP = {
    "asd": "autism",
    "adhd": "attention_deficit",
    "ocd": "obsessive_compulsive",
    "cbt": "cognitive_behavioral_therapy",
    "dbt": "dialectical_behavior_therapy",
    "ptsd": "post_traumatic_stress",
    "ssri": "antidepressant",
    "behavioural": "behavioral",
    "behaviour": "behavior",
    "analyse": "analyze",
    "generalised": "generalized",
}

DOMAIN_BIGRAMS = {
    "obsessive compulsive": "obsessive_compulsive",
    "attention deficit": "attention_deficit",
    "cognitive behavioral": "cognitive_behavioral",
    "social anxiety": "social_anxiety",
    "panic disorder": "panic_disorder",
    "bipolar disorder": "bipolar_disorder",
    "executive function": "executive_function",
    "sensory processing": "sensory_processing",
    "working memory": "working_memory",
}

def is_valid_word(word):
    """Filter out noise and PDF artifacts."""
    if len(set(word)) == 1:        # e.g., "aaa", "bbb"
        return False
    if len(word) > 25:             # Probably a PDF artifact
        return False
    if not any(c in 'aeiouy' for c in word):  # No vowels = not English
        return False
    return True


def is_junk_page(text):
    """
    Detect pure index pages that add noise to the knowledge graph.
    
    Only filters pages that are clearly NOT content:
    - Index pages with high ratios of page numbers (e.g., "Term, 123, 456–789")
    - Pages with an explicit "INDEX" header
    
    We do NOT filter bibliography/reference pages because academic textbooks
    often mix real content with citations on the same page.
    """
    if not text or len(text.strip()) < 50:
        return True

    words = text.split()

    # Index page detection: high ratio of standalone page numbers
    # Index entries look like: "Term, 123, 456–789, 234"
    page_refs = re.findall(r'\b\d{1,4}\b', text)
    if len(words) > 20:
        number_ratio = len(page_refs) / len(words)
        if number_ratio > 0.30:
            return True

    # Explicit "INDEX" in first 100 chars with many page numbers
    first_chunk = text.strip()[:100].upper()
    if "INDEX" in first_chunk and len(page_refs) > 5:
        return True

    return False


# ──────────────────────────────────────────────────────────────
# Stop Words
# ──────────────────────────────────────────────────────────────

ENGLISH_STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "it", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
    "that", "these", "those", "am", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "having", "do", "does", "did", "doing",
    "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between",
    "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "s", "t", "can", "will", "just", "don", "should", "now", "d",
    "ll", "m", "o", "re", "ve", "y", "ain", "aren", "couldn", "didn",
    "doesn", "hadn", "hasn", "haven", "isn", "ma", "mightn", "mustn",
    "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn",
}

# Domain stop words (common in academic books but not meaningful)
DOMAIN_STOP_WORDS = {
    "also", "may", "however", "although", "would", "could", "one",
    "two", "three", "four", "five", "first", "second", "third",
    "used", "using", "use", "many", "much", "often", "see",
    "noted", "found", "reported", "described", "suggest", "suggested",
    "chapter", "section", "figure", "table", "page", "vol",
    "new", "york", "press", "journal", "eds", "london", "university",
    "p1", "mrm", "ikj", "qc", "abe", "t1", "wu038",  # PDF artifacts
    "et", "al", "pp", "doi", "http", "https", "www", "com", "org",
    "copyright", "published", "wiley", "springer", "elsevier",
    "isbn", "edited", "editor", "editors", "handbook", "manual",
}

# Minimum word length to include in the graph
MIN_WORD_LENGTH = 3


# ──────────────────────────────────────────────────────────────
# Config — All 5 Books
# ──────────────────────────────────────────────────────────────

BOOKS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "books")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Each book: (filename, display_name, skip_before, skip_after)
# skip_before = number of front-matter pages to skip (title, copyright, contents)
# skip_after  = page number after which to stop (references, index)
BOOKS = [
    ("ocd.pdf",      "OCD",            12,  430),
    ("adhd.pdf",     "ADHD",            0,   None),
    ("autism.pdf",   "Autism",          15,  None),
    ("dyslexia.pdf", "Dyslexia",       10,  None),
    ("sch.pdf",      "Schizophrenia",  15,  None),
]


# ──────────────────────────────────────────────────────────────
# Step 1: Extract text from a single PDF
# ──────────────────────────────────────────────────────────────

def extract_text_from_pdf(
    pdf_path: str,
    skip_before: int = 10,
    skip_after: int | None = None,
) -> list[str]:
    """
    Extract text from each page of a PDF.
    Returns a list of page texts (one string per page).
    """
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        end_page = skip_after if skip_after is not None else total

        for i, page in enumerate(pdf.pages):
            if i < skip_before or i >= end_page:
                continue

            text = page.extract_text()
            if text and len(text.strip()) > 50 and not is_junk_page(text):
                pages.append(text)

    return pages


# ──────────────────────────────────────────────────────────────
# Step 2: Split into overlapping chunks (sliding window)
# ──────────────────────────────────────────────────────────────

CHUNK_SIZE = 300     # words per chunk
CHUNK_OVERLAP = 75   # words of overlap between consecutive chunks

def split_into_paragraphs(pages: list[str], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split page texts into overlapping chunks using a sliding window.

    Instead of splitting on blank lines (which can cut related content
    across two paragraphs), we:
      1. Clean all pages and join them into one continuous text.
      2. Slide a window of `chunk_size` words forward by
         `chunk_size - overlap` words each step.

    This guarantees that the last `overlap` words of chunk N are also
    the first `overlap` words of chunk N+1, so no context is ever
    lost at a boundary.

    Args:
        pages:      List of raw page texts from the PDF.
        chunk_size: Number of words per chunk (default 300).
        overlap:    Number of overlapping words between chunks (default 75).

    Returns:
        List of text chunks (strings).
    """
    # Clean and merge all pages into one continuous text
    cleaned_pages = []
    for page_text in pages:
        cleaned = re.sub(r'P1:MRM.*?CharCount=\d+', '', page_text)
        cleaned = re.sub(r'WU038.*?\d{4}\s+\d{2}:\d{2}', '', cleaned)
        cleaned = cleaned.strip()
        if cleaned:
            cleaned_pages.append(cleaned)

    full_text = " ".join(cleaned_pages)

    # Normalize whitespace
    full_text = re.sub(r'\s+', ' ', full_text).strip()
    words = full_text.split()

    if not words:
        return []

    # Sliding window
    chunks = []
    step = chunk_size - overlap  # how far to advance each iteration
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_text = " ".join(words[start:end])

        # Only keep chunks with meaningful content (> 30 chars)
        if len(chunk_text) > 30:
            chunks.append(chunk_text)

        start += step

    return chunks


# ──────────────────────────────────────────────────────────────
# Step 3: Clean and tokenize
# ──────────────────────────────────────────────────────────────

def clean_and_tokenize(text: str, stop_words: set) -> list[str]:
    """
    Clean a paragraph and return a list of meaningful words.

    Applies bigram merging, synonym mapping, and advanced lemmatization
    (nouns, verbs, adjectives) to normalize terms.
    """
    text = text.lower()

    # Fix PDF concatenation issues (camelCase splits)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text).lower()

    # Remove punctuation, numbers, special characters
    text = re.sub(r'[^a-z\s]', ' ', text)

    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Apply bigram replacement BEFORE splitting
    for bigram, replacement in DOMAIN_BIGRAMS.items():
        text = text.replace(bigram, replacement)

    # Tokenize, filter, and lemmatize
    words = text.split()
    filtered = []
    for w in words:
        if w in stop_words or w in DOMAIN_STOP_WORDS or len(w) < MIN_WORD_LENGTH:
            continue
            
        if not is_valid_word(w):
            continue

        # Map synonyms
        w = SYNONYM_MAP.get(w, w)

        # Lemmatize (cached, multi-POS)
        # Don't lemmatize bigrams containing underscores to avoid messing them up
        if '_' not in w:
            w = cached_lemmatize(w)
            
        filtered.append(w)

    return filtered


# ──────────────────────────────────────────────────────────────
# Step 4: Build co-occurrence graph from ALL books
# ──────────────────────────────────────────────────────────────

def build_cooccurrence_graph(
    paragraphs: list[dict],
    stop_words: set,
) -> tuple[dict, dict, dict]:
    """
    Build a unified word co-occurrence graph from all paragraphs across all books.

    Args:
        paragraphs: List of {"text": str, "source": str} dicts.
        stop_words: Set of stop words to filter out.

    Returns:
        graph:              {word1: {word2: frequency, ...}, ...}
        word_freq:          {word: total_count}
        word_to_paragraphs: {word: [paragraph_indices]}
    """
    graph: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    word_freq: dict[str, int] = defaultdict(int)
    word_to_paragraphs: dict[str, list[int]] = defaultdict(list)

    for para_idx, para_data in enumerate(paragraphs):
        words = clean_and_tokenize(para_data["text"], stop_words)

        # Count word frequencies
        unique_words = set(words)
        for word in unique_words:
            word_freq[word] += 1
            word_to_paragraphs[word].append(para_idx)

        # Build co-occurrence edges (every pair of unique words)
        for w1, w2 in combinations(unique_words, 2):
            graph[w1][w2] += 1
            graph[w2][w1] += 1

    # Convert defaultdicts to regular dicts for JSON
    graph = {k: dict(v) for k, v in graph.items()}
    word_freq = dict(word_freq)
    word_to_paragraphs = {k: v for k, v in word_to_paragraphs.items()}

    print(f"  Graph nodes (unique words): {len(graph)}")
    print(f"  Graph edges (co-occurrences): {sum(len(v) for v in graph.values()) // 2}")

    return graph, word_freq, word_to_paragraphs


# ──────────────────────────────────────────────────────────────
# Step 5: Save everything
# ──────────────────────────────────────────────────────────────

def save_knowledge_graph(
    graph: dict,
    word_freq: dict,
    paragraphs: list[dict],
    word_to_paragraphs: dict,
    output_dir: str,
):
    """Save all data to disk."""
    os.makedirs(output_dir, exist_ok=True)

    # Save graph
    with open(os.path.join(output_dir, "graph.json"), "w") as f:
        json.dump(graph, f)
    size_kb = os.path.getsize(os.path.join(output_dir, "graph.json")) / 1024
    print(f"  Saved graph.json ({size_kb:.0f} KB)")

    # Save word frequencies
    with open(os.path.join(output_dir, "word_freq.json"), "w") as f:
        json.dump(word_freq, f)
    print(f"  Saved word_freq.json")

    # Save paragraphs with source tags
    with open(os.path.join(output_dir, "paragraphs.json"), "w") as f:
        json.dump(paragraphs, f)
    print(f"  Saved paragraphs.json ({len(paragraphs)} paragraphs)")

    # Save word → paragraph index
    with open(os.path.join(output_dir, "word_to_paragraphs.json"), "w") as f:
        json.dump(word_to_paragraphs, f)
    print(f"  Saved word_to_paragraphs.json")


# ──────────────────────────────────────────────────────────────
# Step 6: Print top connections (verification)
# ──────────────────────────────────────────────────────────────

def print_top_connections(graph: dict, word_freq: dict, top_n: int = 15):
    """Print the most connected words and their top neighbors."""
    sorted_words = sorted(graph.items(), key=lambda x: sum(x[1].values()), reverse=True)

    print(f"\n  Top {top_n} most connected words:")
    print(f"  {'Word':<20} {'Freq':<8} {'Top 5 Neighbors (weight)'}")
    print(f"  {'─'*20} {'─'*8} {'─'*45}")

    for word, neighbors in sorted_words[:top_n]:
        top_neighbors = sorted(neighbors.items(), key=lambda x: x[1], reverse=True)[:10]
        neighbor_str = ", ".join(f"{n}({w})" for n, w in top_neighbors)
        print(f"  {word:<20} {word_freq.get(word, 0):<8} {neighbor_str}")


def print_book_stats(paragraphs: list[dict]):
    """Print per-book paragraph counts."""
    from collections import Counter
    source_counts = Counter(p["source"] for p in paragraphs)

    print(f"\n  Paragraphs per book:")
    print(f"  {'Book':<20} {'Paragraphs':<12}")
    print(f"  {'─'*20} {'─'*12}")
    for source, count in sorted(source_counts.items()):
        print(f"  {source:<20} {count:<12}")
    print(f"  {'─'*20} {'─'*12}")
    print(f"  {'TOTAL':<20} {len(paragraphs):<12}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    print("\nBuilding Unified Knowledge Graph from 5 books...")
    print("=" * 60)

    stop_words = ENGLISH_STOP_WORDS
    all_paragraphs = []  # List of {"text": str, "source": str}

    # ── Step 1 & 2: Extract and split each book ──
    for filename, book_name, skip_before, skip_after in BOOKS:
        pdf_path = os.path.join(BOOKS_DIR, filename)

        if not os.path.exists(pdf_path):
            print(f"\n  Skipping {book_name}: {filename} not found in {BOOKS_DIR}")
            continue

        print(f"\n[{book_name}] Extracting text from {filename}...")
        pages = extract_text_from_pdf(pdf_path, skip_before, skip_after)
        print(f"  Extracted {len(pages)} content pages")

        paragraphs = split_into_paragraphs(pages)
        print(f"  Split into {len(paragraphs)} paragraphs")

        # Tag each paragraph with its source book
        for para_text in paragraphs:
            all_paragraphs.append({
                "text": para_text,
                "source": book_name,
            })

    if not all_paragraphs:
        print("\nNo paragraphs extracted! Check that PDF files exist in data/books/")
        return

    # ── Stats ──
    print_book_stats(all_paragraphs)

    # ── Step 3 & 4: Build unified graph ──
    print(f"\n[Graph] Building unified co-occurrence graph from {len(all_paragraphs)} paragraphs...")
    graph, word_freq, word_to_paragraphs = build_cooccurrence_graph(
        all_paragraphs, stop_words
    )

    # ── Step 5: Save ──
    print("\n[Save] Saving to disk...")
    save_knowledge_graph(graph, word_freq, all_paragraphs, word_to_paragraphs, OUTPUT_DIR)

    # ── Step 6: Verify ──
    print("\n[Verify] Top connections in unified graph:")
    print_top_connections(graph, word_freq)

    print("\n" + "=" * 60)
    print("Unified knowledge graph built successfully!")
    print(f"   Files saved in: {os.path.abspath(OUTPUT_DIR)}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
