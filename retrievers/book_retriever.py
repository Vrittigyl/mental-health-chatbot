"""
Knowledge Graph Retrieval — Multi-Book Hybrid Search + MMR Re-Ranking
======================================================================

Pipeline:
    User Query → SBERT Embed → Match Graph Nodes → Expand via Edges
    → Find Candidate Paragraphs → SBERT Similarity (Top 10)
    → MMR Re-Rank (Top 5) → Return Results with Source Attribution
"""

import json
import os
import re
import torch
from sentence_transformers import SentenceTransformer, util
from typing import List, Tuple

from retrievers.reranker import mmr_rerank, RankedResult

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_NAME = "all-MiniLM-L6-v2"

# ──────────────────────────────────────────────────────────────
# Symptom → Disease Mapping (for queries without explicit disease names)
# ──────────────────────────────────────────────────────────────

SYMPTOM_DISEASE_MAP = {
    # ─── OCD ───
    "checking": "OCD", "washing hands": "OCD", "contamination": "OCD",
    "intrusive thought": "OCD", "ritual": "OCD", "compulsion": "OCD",
    "compulsive": "OCD", "obsession": "OCD", "obsessive": "OCD",
    "locked": "OCD", "clean": "OCD", "germs": "OCD",
    "over and over": "OCD", "reassurance": "OCD", "counting": "OCD",
    "arranging": "OCD", "symmetry": "OCD", "hoarding": "OCD",
    "unwanted thought": "OCD", "can't stop thinking": "OCD",
    "keep checking": "OCD", "hand washing": "OCD",
    "need to check": "OCD", "fear of contamination": "OCD",
    "repeating": "OCD", "harm ocd": "OCD",
    "intrusive": "OCD", "doubt": "OCD",

    # ─── ADHD ───
    "can't focus": "ADHD", "attention": "ADHD", "hyperactive": "ADHD",
    "impulsive": "ADHD", "fidget": "ADHD", "distracted": "ADHD",
    "procrastinat": "ADHD", "forgetful": "ADHD", "concentrate": "ADHD",
    "restless": "ADHD", "disorganized": "ADHD", "can't sit still": "ADHD",
    "easily bored": "ADHD", "zoning out": "ADHD", "time blind": "ADHD",

    # ─── Schizophrenia ───
    "hearing voices": "Schizophrenia", "voices": "Schizophrenia",
    "hallucination": "Schizophrenia", "delusion": "Schizophrenia",
    "paranoid": "Schizophrenia", "paranoia": "Schizophrenia",
    "psychosis": "Schizophrenia", "psychotic": "Schizophrenia",
    "seeing things": "Schizophrenia", "thought insertion": "Schizophrenia",

    # ─── Dyslexia ───
    "reading difficulty": "Dyslexia", "spelling": "Dyslexia",
    "dyslexic": "Dyslexia", "can't read": "Dyslexia",
    "mix up letters": "Dyslexia", "words move": "Dyslexia",
    "reading slow": "Dyslexia", "phonological": "Dyslexia",

    # ─── Autism ───
    "spectrum": "Autism", "sensory overload": "Autism",
    "stimming": "Autism", "meltdown": "Autism",
    "eye contact": "Autism", "social cue": "Autism",
    "nonverbal": "Autism", "special interest": "Autism",
    "routine change": "Autism", "sensory": "Autism",
}

# Graph terms to inject when a disease is detected from symptoms
DISEASE_GRAPH_TERMS = {
    "OCD": ["obsessive_compulsive", "compulsion", "obsession", "ritual", "anxiety", "exposure", "cognitive_behavioral_therapy"],
    "ADHD": ["attention_deficit", "hyperactivity", "impulsivity", "executive_function", "stimulant", "dopamine"],
    "Schizophrenia": ["schizophrenia", "psychosis", "hallucination", "delusion", "antipsychotic", "dopamine"],
    "Dyslexia": ["dyslexia", "reading", "phonological", "orthographic", "decoding", "literacy"],
    "Autism": ["autism", "autistic", "social", "sensory_processing", "developmental", "spectrum"],
}

# Similarity boost for paragraphs from detected disease books
SOURCE_BOOST = 0.06


def detect_diseases_from_query(query: str) -> List[str]:
    """
    Detect likely diseases from a user query using symptom keyword matching.
    Returns a list of detected disease book names (e.g., ["OCD", "ADHD"]).
    """
    query_lower = query.lower()
    disease_scores: dict[str, int] = {}

    for symptom, disease in SYMPTOM_DISEASE_MAP.items():
        if symptom in query_lower:
            disease_scores[disease] = disease_scores.get(disease, 0) + 1

    if not disease_scores:
        return []

    # Return diseases sorted by number of matching symptoms (desc)
    sorted_diseases = sorted(disease_scores.items(), key=lambda x: -x[1])
    return [d for d, _ in sorted_diseases]


class HybridRetriever:
    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir

        # Load raw data
        raw_paragraphs = self._load_json("paragraphs.json")
        self.word_to_paragraphs = self._load_json("word_to_paragraphs.json")
        self.graph = self._load_json("graph.json")

        # Handle both old format (list of strings) and new format (list of dicts)
        if raw_paragraphs and isinstance(raw_paragraphs[0], dict):
            self.paragraphs = [p["text"] for p in raw_paragraphs]
            self.sources = [p["source"] for p in raw_paragraphs]
        else:
            self.paragraphs = raw_paragraphs
            self.sources = ["Unknown"] * len(raw_paragraphs)

        # Load embeddings
        print("Loading embeddings...")
        self.node_embed_dict = torch.load(os.path.join(self.data_dir, "node_embeddings.pt"))
        self.para_embs = torch.load(os.path.join(self.data_dir, "paragraph_embeddings.pt"))

        # Prepare node tensors for fast matrix multiplication
        self.node_words = list(self.node_embed_dict.keys())
        self.node_tensor = torch.stack([self.node_embed_dict[w] for w in self.node_words])

        # Load model
        print("Loading sentence transformer model...")
        self.model = SentenceTransformer(MODEL_NAME)

        print(f"Ready! {len(self.paragraphs)} paragraphs from {len(set(self.sources))} books")

    def _load_json(self, filename: str):
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Missing {filename}.")
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def _get_candidate_indices(
        self, query_emb: torch.Tensor, query: str, top_nodes: int = 3
    ) -> List[int]:
        """
        Steps 2–4: Semantic node matching → graph expansion → candidate filtering.
        Returns a list of valid paragraph indices.
        """
        # 2. Semantic Node Matching
        cos_scores = util.cos_sim(query_emb, self.node_tensor)[0]
        top_node_results = torch.topk(cos_scores, k=top_nodes)

        matched_nodes = []
        for score, idx in zip(top_node_results[0], top_node_results[1]):
            word = self.node_words[idx]
            matched_nodes.append((word, score.item()))

        print(f"\n[Search] Top Semantic Graph Nodes: {', '.join([w for w, s in matched_nodes])}")

        # 3. Graph Expansion
        expanded_nodes = set(w for w, _ in matched_nodes)
        for word, _ in matched_nodes:
            if word in self.graph:
                neighbors = sorted(
                    self.graph[word].items(), key=lambda x: x[1], reverse=True
                )[:2]
                for neighbor, weight in neighbors:
                    expanded_nodes.add(neighbor)

        # 3b. Symptom-based Disease Expansion
        detected_diseases = detect_diseases_from_query(query)
        if detected_diseases:
            print(f"[Search] Detected diseases from symptoms: {', '.join(detected_diseases)}")
            for disease in detected_diseases:
                for term in DISEASE_GRAPH_TERMS.get(disease, []):
                    if term in self.word_to_paragraphs:
                        expanded_nodes.add(term)

        print(f"[Search] Expanded Concepts via Graph: {', '.join(expanded_nodes)}")

        # 4. Candidate Paragraph Selection
        candidate_para_indices = set()
        for word in expanded_nodes:
            if word in self.word_to_paragraphs:
                candidate_para_indices.update(self.word_to_paragraphs[word])

        if not candidate_para_indices:
            print("[Search] No candidate paragraphs found.")
            return []

        # FILTER: Remove garbage paragraphs and strictly filter by target books if mentioned in query
        target_books = [book for book in ["Autism", "ADHD", "OCD", "Dyslexia", "Schizophrenia"] if book.lower() in query.lower()]
        if target_books:
            print(f"[Search] Filtered strictly to books: {target_books}")

        valid_indices = []
        for idx in candidate_para_indices:
            text = self.paragraphs[idx]
            
            # Enforce book filter if a book is explicitly named in the query
            if target_books and self.sources[idx] not in target_books:
                continue
                
            if "REFERENCES" in text[:50]:
                continue
            words = text.split()
            if len(words) > 0:
                avg_word_length = sum(len(w) for w in words) / len(words)
                if avg_word_length > 10:
                    continue
            valid_indices.append(idx)

        return valid_indices

    def retrieve_top_k(
        self,
        query: str,
        k: int = 5,
        top_nodes: int = 3,
        similarity_pool: int = 10,
        lambda_param: float = 0.7,
    ) -> List[dict]:
        """
        Two-stage retrieval:
            Stage 1 — SBERT Similarity: Get top `similarity_pool` candidates (default 10)
            Stage 2 — MMR Re-Ranking:   From those, pick top `k` (default 5)

        Args:
            query:            User search query string.
            k:                Final number of results after MMR re-ranking.
            top_nodes:        Number of graph nodes to match.
            similarity_pool:  How many candidates to fetch in Stage 1.
            lambda_param:     MMR trade-off: 1.0 = pure relevance, 0.0 = pure diversity.

        Returns:
            List of dicts with keys: index, text, source, score, relevance, diversity
        """
        if not query.strip():
            return []

        # 1. Embed user query
        query_emb = self.model.encode(query, convert_to_tensor=True).cpu()

        # 2–4. Get candidate paragraph indices via knowledge graph
        candidate_indices = self._get_candidate_indices(query_emb, query, top_nodes)

        if not candidate_indices:
            return []

        # ── Stage 1: SBERT Similarity → Top N ──
        candidate_embs = self.para_embs[candidate_indices]
        cos_scores = util.cos_sim(query_emb, candidate_embs)[0]

        # Apply source boost for symptom-detected diseases
        detected_diseases = detect_diseases_from_query(query)
        if detected_diseases:
            for i, idx in enumerate(candidate_indices):
                if self.sources[idx] in detected_diseases:
                    cos_scores[i] += SOURCE_BOOST

        pool_size = min(similarity_pool, len(candidate_indices))
        top_sim_results = torch.topk(cos_scores, k=pool_size)

        top_n_indices = [candidate_indices[int(idx.item())] for idx in top_sim_results.indices]
        top_n_embs = self.para_embs[top_n_indices]
        top_n_texts = [self.paragraphs[int(idx)] for idx in top_n_indices]

        print(f"[Stage 1] SBERT Similarity → Top {pool_size} from {len(candidate_indices)} candidates")

        # ── Stage 2: MMR Re-Ranking → Top K ──
        ranked_results = mmr_rerank(
            query_emb=query_emb,
            candidate_embs=top_n_embs,
            candidate_texts=top_n_texts,
            top_k=k,
            lambda_param=lambda_param,
        )

        print(f"[Stage 2] MMR Re-Ranking → Top {len(ranked_results)} (λ={lambda_param})")

        # Build final results with source attribution
        final_results = []
        for result in ranked_results:
            global_idx = top_n_indices[result.index]
            final_results.append({
                "index": global_idx,
                "text": result.text,
                "source": self.sources[global_idx],
                "score": result.final_score,
                "relevance": result.relevance_score,
                "diversity": result.diversity_score,
            })

        return final_results


# ──────────────────────────────────────────────────────────────
# Output Helpers
# ──────────────────────────────────────────────────────────────

def clean_output_text(text: str) -> str:
    """Cleans raw PDF text for readable terminal output."""
    text = re.sub(r'P[0-9]:[A-Z/]+\s*.*?\n', '', text)
    text = re.sub(r'CharCount=\d+\n?', '', text)
    text = re.sub(r'-\n\s*', '', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def main():
    import sys

    args = sys.argv[1:]

    if "--help" in args or "-h" in args:
        print("""
Usage: python -m knowledge.retrieve [QUERY]

Examples:
  python -m knowledge.retrieve "What causes OCD?"
  python -m knowledge.retrieve "treatment for anxiety"
  python -m knowledge.retrieve "ADHD symptoms in children"
        """)
        return

    if args:
        query = " ".join(args)
    else:
        print("Enter your question (paste your text, then press Enter twice to submit):")
        lines = []
        while True:
            try:
                line = input()
                if not line.strip():  # Stop if the user enters an empty line
                    break
                lines.append(line)
            except EOFError:
                break
        query = "\n".join(lines)

    print(f"\nSearching for: '{query}'")

    try:
        retriever = HybridRetriever()
        results = retriever.retrieve_top_k(query, k=5, similarity_pool=10)

        if not results:
            print("No relevant information found.")
            return

        print(f"\nTop {len(results)} results:\n")
        for i, result in enumerate(results, 1):
            cleaned_text = clean_output_text(result["text"])
            print(f"--- Result {i} [{result['source']}] (Score: {result['score']:.4f}) ---")
            print(f"    Relevance: {result['relevance']:.4f} | Diversity: {result['diversity']:.4f}")
            print(cleaned_text)
            print()

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you run these first:")
        print("  python -m knowledge.ingest")
        print("  python -m knowledge.ingest_embeddings")


if __name__ == "__main__":
    main()
